"""
Multi-Stream Audio Processor for ARACHNE-X
Separates audio into three streams: lip-sync, prosody/emotion, and head movement.
Enables fine-grained control over avatar animation synthesis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np
from scipy import signal


class LipSyncAnalyzer(nn.Module):
    """
    Extracts high-frequency lip-sync information from audio.
    Focuses on frequencies 18-24 Hz (phoneme articulation patterns).
    """
    def __init__(self, audio_dim: int = 768, output_dim: int = 512):
        super().__init__()
        self.audio_dim = audio_dim
        self.output_dim = output_dim
        
        # Temporal convolution for phoneme detection
        self.phoneme_conv = nn.Sequential(
            nn.Conv1d(audio_dim, 256, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Vowel/consonant classifier (guides mouth opening)
        self.vowel_classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 5)  # A, E, I, O, U
        )
        
        # Output projection
        self.output_proj = nn.Linear(128, output_dim)
        
    def forward(self, audio_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_embeddings: [B, T, audio_dim] from wav2vec2
            
        Returns:
            lip_sync_features: [B, T//2, output_dim]
            vowel_logits: [B, T//2, 5]
        """
        B, T, D = audio_embeddings.shape
        
        # Transpose for conv1d: [B, D, T]
        x = audio_embeddings.transpose(1, 2)
        
        # Extract phoneme patterns
        phoneme_feat = self.phoneme_conv(x)  # [B, 128, T//2]
        
        # Classify vowels
        vowel_logits = self.vowel_classifier(
            phoneme_feat.transpose(1, 2)
        )  # [B, T//2, 5]
        
        # Project to output dimension
        lip_sync_features = self.output_proj(
            phoneme_feat.transpose(1, 2)
        )  # [B, T//2, output_dim]
        
        return lip_sync_features, vowel_logits


class ProsodyAnalyzer(nn.Module):
    """
    Extracts emotion/prosody information from audio.
    Focuses on frequencies 4-6 Hz (intonation, stress patterns).
    Drives facial expressions (smile, concern, etc).
    """
    def __init__(self, audio_dim: int = 768, output_dim: int = 512):
        super().__init__()
        self.audio_dim = audio_dim
        self.output_dim = output_dim
        
        # Low-frequency extraction via pooling
        self.low_freq_pool = nn.AdaptiveAvgPool1d(128)
        
        # Prosody encoder
        self.prosody_encoder = nn.Sequential(
            nn.Linear(audio_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Emotion classifier (AUs - Action Units)
        self.emotion_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 12)  # 12 primary facial action units
        )
        
        # Output projection
        self.output_proj = nn.Linear(256, output_dim)
        
    def forward(self, audio_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_embeddings: [B, T, audio_dim]
            
        Returns:
            prosody_features: [B, T//4, output_dim] (downsampled for stability)
            emotion_logits: [B, T//4, 12] (Action Unit activations)
        """
        B, T, D = audio_embeddings.shape
        
        # Downsample for low-frequency analysis
        x = audio_embeddings.transpose(1, 2)  # [B, D, T]
        x = self.low_freq_pool(x)  # [B, D, 128]
        x = x.transpose(1, 2)  # [B, 128, D]
        
        # Average pooling to get frame-level prosody
        x = x.mean(dim=1, keepdim=True)  # [B, 1, D]
        
        # Encode prosody
        prosody_encoded = self.prosody_encoder(x.squeeze(1))  # [B, 256]
        
        # Classify emotions (AUs)
        emotion_logits = self.emotion_classifier(prosody_encoded)  # [B, 12]
        emotion_logits = emotion_logits.unsqueeze(1).expand(B, T//4, 12)  # Broadcast to time
        
        # Project to output
        prosody_features = self.output_proj(prosody_encoded)  # [B, output_dim]
        prosody_features = prosody_features.unsqueeze(1).expand(B, T//4, -1)  # [B, T//4, output_dim]
        
        return prosody_features, emotion_logits


class HeadMovementAnalyzer(nn.Module):
    """
    Extracts head movement patterns from audio pacing.
    Focuses on frequencies 1-2 Hz (head nods, turns during speech).
    """
    def __init__(self, audio_dim: int = 768, output_dim: int = 256):
        super().__init__()
        self.audio_dim = audio_dim
        self.output_dim = output_dim
        
        # Very low frequency extraction
        self.ultra_low_freq_pool = nn.AdaptiveAvgPool1d(32)
        
        # Head motion encoder
        self.head_motion_encoder = nn.Sequential(
            nn.Linear(audio_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # 3D rotation/translation predictor (pitch, yaw, roll + x,y,z translation)
        self.head_pose_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)  # [pitch, yaw, roll, tx, ty, tz]
        )
        
        # Output projection
        self.output_proj = nn.Linear(128, output_dim)
        
    def forward(self, audio_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_embeddings: [B, T, audio_dim]
            
        Returns:
            head_movement_features: [B, T//8, output_dim]
            head_pose_6dof: [B, T//8, 6] (pitch, yaw, roll, tx, ty, tz)
        """
        B, T, D = audio_embeddings.shape
        
        # Ultra-low frequency pooling
        x = audio_embeddings.transpose(1, 2)  # [B, D, T]
        x = self.ultra_low_freq_pool(x)  # [B, D, 32]
        x = x.transpose(1, 2)  # [B, 32, D]
        
        # Encode head motion
        head_encoded = self.head_motion_encoder(x)  # [B, 32, 128]
        
        # Predict 6-DoF head pose
        head_pose_6dof = self.head_pose_predictor(head_encoded)  # [B, 32, 6]
        
        # Project to output features
        head_movement_features = self.output_proj(head_encoded)  # [B, 32, output_dim]
        
        # Smooth 6-DoF predictions with temporal filtering
        head_pose_6dof_smooth = self._smooth_pose_predictions(head_pose_6dof)
        
        return head_movement_features, head_pose_6dof_smooth
    
    def _smooth_pose_predictions(self, poses: torch.Tensor, window_size: int = 3) -> torch.Tensor:
        """Apply temporal smoothing to 6-DoF predictions."""
        B, T, D = poses.shape
        smoothed = poses.clone()
        
        for b in range(B):
            for d in range(D):
                # Apply Savitzky-Golay filter for smoothing
                smoothed[b, :, d] = torch.tensor(
                    signal.savgol_filter(poses[b, :, d].cpu().numpy(), window_size, 2),
                    device=poses.device, dtype=poses.dtype
                )
        
        return smoothed


class MultiStreamAudioProcessor(nn.Module):
    """
    Complete multi-stream audio processor combining all three streams.
    Integrates with existing Wav2Vec2ModelWrapper.
    """
    def __init__(
        self,
        audio_embedding_dim: int = 768,
        lip_sync_dim: int = 512,
        prosody_dim: int = 512,
        head_movement_dim: int = 256,
        use_wav2vec_embeddings: bool = True
    ):
        super().__init__()
        self.audio_embedding_dim = audio_embedding_dim
        self.use_wav2vec_embeddings = use_wav2vec_embeddings
        
        # Three analysis streams
        self.lip_sync_analyzer = LipSyncAnalyzer(audio_embedding_dim, lip_sync_dim)
        self.prosody_analyzer = ProsodyAnalyzer(audio_embedding_dim, prosody_dim)
        self.head_movement_analyzer = HeadMovementAnalyzer(audio_embedding_dim, head_movement_dim)
        
        # Fusion layer to combine all streams
        total_dim = lip_sync_dim + prosody_dim + head_movement_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024)
        )
        
    def forward(
        self,
        audio_embeddings: torch.Tensor,
        sample_rate: int = 16000,
        fps: int = 30
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            audio_embeddings: [B, T, 768] from Wav2Vec2ModelWrapper
            sample_rate: audio sample rate
            fps: target video frame rate
            
        Returns:
            dict containing:
                - lip_sync_features: [B, T//2, 512]
                - lip_sync_vowels: [B, T//2, 5]
                - prosody_features: [B, T//4, 512]
                - emotion_logits: [B, T//4, 12]
                - head_movement_features: [B, T//8, 256]
                - head_pose_6dof: [B, T//8, 6]
                - fused_embeddings: [B, min_T, 1024]
        """
        B, T, D = audio_embeddings.shape
        
        # Extract lip-sync stream
        lip_sync_feat, lip_sync_vowels = self.lip_sync_analyzer(audio_embeddings)
        
        # Extract prosody stream
        prosody_feat, emotion_logits = self.prosody_analyzer(audio_embeddings)
        
        # Extract head movement stream
        head_move_feat, head_pose_6dof = self.head_movement_analyzer(audio_embeddings)
        
        # Synchronize to common time dimension (use coarsest: T//8)
        min_t = min(lip_sync_feat.shape[1], prosody_feat.shape[1], head_move_feat.shape[1])
        
        lip_sync_feat_sync = lip_sync_feat[:, :min_t]
        prosody_feat_sync = prosody_feat[:, :min_t]
        head_move_feat_sync = head_move_feat[:, :min_t]
        
        # Fuse all streams
        fused = torch.cat([lip_sync_feat_sync, prosody_feat_sync, head_move_feat_sync], dim=-1)
        fused_embeddings = self.fusion(fused)  # [B, min_t, 1024]
        
        return {
            'lip_sync_features': lip_sync_feat,
            'lip_sync_vowels': lip_sync_vowels,
            'prosody_features': prosody_feat,
            'emotion_logits': emotion_logits,
            'head_movement_features': head_move_feat,
            'head_pose_6dof': head_pose_6dof,
            'fused_embeddings': fused_embeddings,
        }
