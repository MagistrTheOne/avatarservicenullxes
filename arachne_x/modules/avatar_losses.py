"""
Avatar-Specific Loss Functions for ARACHNE-X
Multi-objective loss stack optimized for hyper-realistic avatar generation:
- Lip-sync accuracy
- Identity consistency
- Temporal coherence
- Facial expression control
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class LipSyncLoss(nn.Module):
    """
    Measures sync between audio features and mouth region in generated video.
    Uses contrastive learning between synchronized and non-synchronized pairs.
    """
    def __init__(self, embedding_dim: int = 256, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.embedding_dim = embedding_dim
        
        # Audio feature projector for sync space
        self.audio_projector = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
        
        # Video mouth region projector
        self.mouth_projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
        
    def forward(
        self,
        audio_features: torch.Tensor,  # [B, T, 768]
        mouth_features: torch.Tensor,   # [B, T, 512]
        video_frames: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            audio_features: wav2vec embeddings
            mouth_features: extracted mouth region features from generated video
            video_frames: optional ground truth frames for supervision
            
        Returns:
            lip_sync_loss: scalar loss value
        """
        B, T, _ = audio_features.shape
        
        # Project to sync embedding space
        audio_emb = self.audio_projector(audio_features)  # [B, T, embedding_dim]
        mouth_emb = self.mouth_projector(mouth_features)  # [B, T, embedding_dim]
        
        # Normalize embeddings
        audio_emb = F.normalize(audio_emb, p=2, dim=-1)
        mouth_emb = F.normalize(mouth_emb, p=2, dim=-1)
        
        # Compute similarity matrix: [B, T, T]
        # Diagonal should be high (sync), off-diagonal should be low
        sim_matrix = torch.bmm(mouth_emb, audio_emb.transpose(1, 2)) / self.temperature
        
        # Create targets: diagonal is positive (1), off-diagonal is negative (0)
        targets = torch.eye(T, device=audio_features.device).unsqueeze(0).expand(B, -1, -1)
        
        # Contrastive loss (cross-entropy)
        loss = F.binary_cross_entropy_with_logits(sim_matrix, targets)
        
        # DTW-based loss for temporal alignment (higher weight for synchronized frames)
        dtw_loss = self._compute_dtw_loss(audio_emb, mouth_emb)
        
        return loss + 0.5 * dtw_loss
    
    def _compute_dtw_loss(self, audio_emb: torch.Tensor, mouth_emb: torch.Tensor) -> torch.Tensor:
        """Simplified DTW loss for temporal alignment."""
        # Compute frame-wise distances
        distances = torch.cdist(audio_emb, mouth_emb, p=2)  # [B, T, T]
        
        # Extract diagonal (synchronized pairs)
        diag_distances = torch.diagonal(distances, dim1=1, dim2=2)  # [B, T]
        
        # Smooth with neighbors (enforce temporal coherence)
        temporal_penalty = torch.sum((diag_distances[:, 1:] - diag_distances[:, :-1]) ** 2, dim=1)
        
        return torch.mean(diag_distances) + 0.1 * torch.mean(temporal_penalty)


class IdentityPreservationLoss(nn.Module):
    """
    Preserves identity consistency across generated frames using face embeddings.
    Uses ArcFace-style embeddings for identity verification.
    """
    def __init__(self, embedding_dim: int = 512, margin: float = 0.5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.margin = margin
        
        # Simple face feature extractor (would be replaced with pretrained ArcFace)
        self.face_feature_extractor = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_dim)
        )
        
        self.loss_fn = nn.CosineSimilarity(dim=-1)
        
    def forward(
        self,
        generated_face_features: torch.Tensor,  # [B, T, 2048]
        reference_face_feature: torch.Tensor     # [B, 2048]
    ) -> torch.Tensor:
        """
        Args:
            generated_face_features: face features from generated frames
            reference_face_feature: face feature from reference image
            
        Returns:
            identity_loss: scalar loss value
        """
        B, T, D = generated_face_features.shape
        
        # Extract embeddings
        generated_emb = self.face_feature_extractor(generated_face_features)  # [B, T, embedding_dim]
        reference_emb = self.face_feature_extractor(reference_face_feature)   # [B, embedding_dim]
        
        # Normalize
        generated_emb = F.normalize(generated_emb, p=2, dim=-1)
        reference_emb = F.normalize(reference_emb, p=2, dim=-1)
        
        # Expand reference for comparison
        reference_emb = reference_emb.unsqueeze(1).expand(B, T, -1)  # [B, T, embedding_dim]
        
        # Compute cosine similarity
        similarity = self.loss_fn(generated_emb, reference_emb)  # [B, T]
        
        # Loss: minimize distance from identity (maximize similarity)
        # Use margin-based loss
        loss = torch.clamp(self.margin - similarity, min=0).mean()
        
        return loss


class TemporalCoherenceLoss(nn.Module):
    """
    Ensures smooth temporal transitions between frames.
    Uses optical flow consistency and feature smoothness.
    """
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        latents: torch.Tensor,  # [B, C, T, H, W]
        optical_flow: Optional[torch.Tensor] = None  # [B, 2, T-1, H, W]
    ) -> torch.Tensor:
        """
        Args:
            latents: generated latent features across time
            optical_flow: estimated optical flow between consecutive frames
            
        Returns:
            temporal_loss: scalar loss value
        """
        B, C, T, H, W = latents.shape
        
        # Temporal smoothness: penalize large changes between consecutive frames
        frame_diff = latents[:, :, 1:] - latents[:, :, :-1]  # [B, C, T-1, H, W]
        temporal_smoothness = torch.mean(torch.abs(frame_diff))
        
        # Optical flow warping consistency (if available)
        if optical_flow is not None:
            # Warp frame t using optical flow to predict frame t+1
            # This is a simplified version - full implementation would use grid_sample
            flow_consistency = torch.mean(torch.abs(optical_flow))
        else:
            flow_consistency = 0.0
        
        # Temporal variance penalty (avoid static regions)
        temporal_var = torch.var(latents, dim=2).mean()
        temporal_var_penalty = -torch.log(torch.clamp(temporal_var, min=1e-6))
        
        # Combined loss
        loss = temporal_smoothness + 0.1 * flow_consistency + 0.05 * temporal_var_penalty
        
        return loss


class ExpressionControlLoss(nn.Module):
    """
    Controls facial expressions through Action Unit (AU) guidance.
    Ensures generated expressions match intended emotional state.
    """
    def __init__(self, num_aus: int = 12):
        super().__init__()
        self.num_aus = num_aus
        
        # AU classifier for generated video
        self.au_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_aus),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        face_features: torch.Tensor,       # [B, T, 512]
        target_emotion_logits: torch.Tensor,  # [B, T, num_aus]
        emotion_weight: float = 1.0
    ) -> torch.Tensor:
        """
        Args:
            face_features: extracted facial features
            target_emotion_logits: target AU activations from audio prosody
            emotion_weight: weight for expression loss
            
        Returns:
            expression_loss: scalar loss value
        """
        B, T, D = face_features.shape
        
        # Predict AUs from generated face
        predicted_aus = self.au_classifier(face_features)  # [B, T, num_aus]
        
        # Convert logits to probabilities
        target_aus = torch.sigmoid(target_emotion_logits)  # [B, T, num_aus]
        
        # MSE loss between predicted and target AUs
        au_loss = F.mse_loss(predicted_aus, target_aus)
        
        # Temporal consistency of AUs (smooth changes)
        au_temporal_diff = torch.abs(predicted_aus[:, 1:] - predicted_aus[:, :-1])
        au_temporal_smoothness = torch.mean(au_temporal_diff)
        
        loss = au_loss + 0.1 * au_temporal_smoothness
        
        return emotion_weight * loss


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features or similar.
    Ensures generated faces are realistic and match reference.
    """
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        
        # Simplified feature extractor (would use pretrained VGG/LPIPS)
        self.feature_extractor = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, feature_dim)
        )
        
    def forward(
        self,
        generated_features: torch.Tensor,  # [B, T, 2048]
        reference_features: torch.Tensor   # [B, 2048]
    ) -> torch.Tensor:
        """
        Args:
            generated_features: features from generated frames
            reference_features: features from reference frame
            
        Returns:
            perceptual_loss: scalar loss value
        """
        # Extract features
        gen_feat = self.feature_extractor(generated_features)  # [B, T, feature_dim]
        ref_feat = self.feature_extractor(reference_features)  # [B, feature_dim]
        
        # Expand reference
        ref_feat = ref_feat.unsqueeze(1).expand_as(gen_feat)
        
        # L2 distance in feature space
        loss = F.mse_loss(gen_feat, ref_feat)
        
        return loss


class ARACHNEAvatarLossModule(nn.Module):
    """
    Complete loss module combining all components with learned weighting.
    """
    def __init__(
        self,
        lip_sync_weight: float = 0.25,
        identity_weight: float = 0.15,
        temporal_weight: float = 0.10,
        expression_weight: float = 0.10,
        perceptual_weight: float = 0.40,
    ):
        super().__init__()
        
        self.lip_sync_weight = lip_sync_weight
        self.identity_weight = identity_weight
        self.temporal_weight = temporal_weight
        self.expression_weight = expression_weight
        self.perceptual_weight = perceptual_weight
        
        # Initialize loss components
        self.lip_sync_loss = LipSyncLoss()
        self.identity_loss = IdentityPreservationLoss()
        self.temporal_loss = TemporalCoherenceLoss()
        self.expression_loss = ExpressionControlLoss()
        self.perceptual_loss = PerceptualLoss()
        
    def forward(
        self,
        audio_features: torch.Tensor,
        mouth_features: torch.Tensor,
        generated_face_features: torch.Tensor,
        reference_face_feature: torch.Tensor,
        latents: torch.Tensor,
        face_features: torch.Tensor,
        target_emotion_logits: torch.Tensor,
        optical_flow: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses and return both individual components and total.
        
        Returns:
            dict with keys: lip_sync, identity, temporal, expression, perceptual, total
        """
        # Compute individual losses
        lip_sync = self.lip_sync_loss(audio_features, mouth_features)
        identity = self.identity_loss(generated_face_features, reference_face_feature)
        temporal = self.temporal_loss(latents, optical_flow)
        expression = self.expression_loss(face_features, target_emotion_logits)
        perceptual = self.perceptual_loss(generated_face_features, reference_face_feature)
        
        # Weighted sum
        total_loss = (
            self.lip_sync_weight * lip_sync +
            self.identity_weight * identity +
            self.temporal_weight * temporal +
            self.expression_weight * expression +
            self.perceptual_weight * perceptual
        )
        
        return {
            'lip_sync': lip_sync.detach(),
            'identity': identity.detach(),
            'temporal': temporal.detach(),
            'expression': expression.detach(),
            'perceptual': perceptual.detach(),
            'total': total_loss
        }
