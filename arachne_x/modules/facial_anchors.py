"""
Facial Keypoint Anchoring Module for ARACHNE-X
Provides 68-point facial landmark anchoring for stable face generation
and high-frequency detail preservation.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np
import os

logger = logging.getLogger(__name__)


class FacialAnchorEmbedder(nn.Module):
    """
    Embeds 68 facial landmarks (MediaPipe/DLIB format) into high-dimensional space.
    These anchors are used to constrain the diffusion process in facial regions.
    """
    def __init__(self, hidden_size: int = 1024, num_landmarks: int = 68):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_landmarks = num_landmarks
        
        # Each landmark has (x, y, confidence)
        self.landmark_input_dim = num_landmarks * 2  # x, y only
        
        # MLP for landmark encoding
        self.landmark_encoder = nn.Sequential(
            nn.Linear(self.landmark_input_dim, hidden_size * 2),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        # Per-landmark attention weights (learned per region)
        self.landmark_region_attn = nn.Sequential(
            nn.Linear(hidden_size, num_landmarks),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, landmarks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            landmarks: [B, 68, 2] normalized facial keypoints (0-1 range)
            
        Returns:
            anchor_embed: [B, hidden_size] aggregated landmark embedding
            region_attn: [B, 68] attention weights per landmark region
        """
        B = landmarks.shape[0]
        
        # Flatten landmarks for encoder
        landmarks_flat = landmarks.reshape(B, -1)  # [B, 136]
        
        # Encode landmarks
        anchor_embed = self.landmark_encoder(landmarks_flat)  # [B, hidden_size]
        
        # Compute per-landmark attention
        region_attn = self.landmark_region_attn(anchor_embed)  # [B, 68]
        
        return anchor_embed, region_attn


class FacialAnchorConstrainer(nn.Module):
    """
    Applies facial landmark constraints to latent features during diffusion.
    Prevents unrealistic face deformations and maintains identity consistency.
    """
    def __init__(self, hidden_size: int = 1024, latent_channels: int = 16):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_channels = latent_channels
        
        # Compute warp field from anchors
        self.warp_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 2)  # 2D warp displacement
        )
        
        # Feature masking for face regions
        self.face_mask_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self, 
        latents: torch.Tensor, 
        anchor_embed: torch.Tensor,
        region_attn: torch.Tensor,
        spatial_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Args:
            latents: [B, C, T, H, W] latent features
            anchor_embed: [B, hidden_size] landmark embeddings
            region_attn: [B, 68] per-landmark attention
            spatial_shape: (H, W) of latent space
            
        Returns:
            constrained_latents: [B, C, T, H, W] with facial constraints applied
        """
        B, C, T, H, W = latents.shape
        
        # Generate face region mask
        face_mask = self.face_mask_generator(anchor_embed)  # [B, 1]
        face_mask = face_mask.view(B, 1, 1, 1, 1).expand(B, 1, T, H, W)
        
        # Generate warp displacement field
        warp_disp = self.warp_generator(anchor_embed)  # [B, 2]
        
        # Apply constraints: blend original with constrained version
        # Using face mask to selectively apply constraints
        constrained_latents = latents.clone()
        
        # Soft constraint: reduce variation in face region
        face_region_feature = latents * face_mask
        constrained_latents = constrained_latents * (1 - face_mask) + face_region_feature * face_mask
        
        return constrained_latents


class KalmanFilter2D:
    """Simple 2D Kalman filter for (x,y) with velocity state for temporal smoothing."""
    def __init__(self, device='cpu', process_var: float = 1e-3, measure_var: float = 1e-2):
        self.device = device
        # State: [x, y, vx, vy]
        self.F = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32, device=device)
        self.H = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.float32, device=device)
        self.Q = torch.eye(4, device=device) * process_var
        self.R = torch.eye(2, device=device) * measure_var

    def init_state(self, x: float, y: float):
        self.x = torch.tensor([x, y, 0.0, 0.0], dtype=torch.float32, device=self.device)
        self.P = torch.eye(4, dtype=torch.float32, device=self.device)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.t() + self.Q

    def update(self, meas_x: float, meas_y: float):
        z = torch.tensor([meas_x, meas_y], dtype=torch.float32, device=self.device)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.t() + self.R
        K = self.P @ self.H.t() @ torch.inverse(S)
        self.x = self.x + K @ y
        I = torch.eye(self.P.shape[0], device=self.device)
        self.P = (I - K @ self.H) @ self.P

    def state(self):
        return self.x[0].item(), self.x[1].item()


class LandmarkDetectorV2(nn.Module):
    """Robust landmark detector with MediaPipe primary and optional DLIB fallback. Returns coords and per-landmark confidence."""
    def __init__(self, device: str = "cuda", allow_zero_fallback: bool = False):
        super().__init__()
        self.device = device
        self.allow_zero_fallback = allow_zero_fallback
        self.use_mediapipe = False
        self.use_dlib = False
        self._warned_no_detector = False
        try:
            import mediapipe as mp
            self.mp = mp
            self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            )
            self.use_mediapipe = True
        except Exception:
            self.use_mediapipe = False

        if not self.use_mediapipe:
            try:
                import dlib
                import cv2
                self.dlib = dlib
                self.cv2 = cv2
                # shape predictor must be provided by deployment; try local file
                predictor_path = "shape_predictor_68_face_landmarks.dat"
                if os.path.exists(predictor_path):
                    self.shape_predictor = dlib.shape_predictor(predictor_path)
                    self.detector = dlib.get_frontal_face_detector()
                    self.use_dlib = True
                else:
                    self.use_dlib = False
            except Exception:
                self.use_dlib = False

        if not self.use_mediapipe and not self.use_dlib:
            msg = (
                "LandmarkDetectorV2 has no available backend (MediaPipe/DLIB). "
                "Install `mediapipe` or provide `shape_predictor_68_face_landmarks.dat` for dlib."
            )
            if self.allow_zero_fallback:
                logger.warning("%s Falling back to zero landmarks.", msg)
            else:
                raise RuntimeError(msg)

    def forward(self, frame_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            frame_rgb: HxW x3 uint8 RGB image
        Returns:
            lmks: (68,2) normalized coords (0..1), confidences: (68,) floats 0..1
        """
        H, W, _ = frame_rgb.shape
        if self.use_mediapipe:
            results = self.mp_face_mesh.process(frame_rgb)
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0].landmark
                lmks = np.array([[lm.x, lm.y] for lm in face_landmarks[:68]], dtype=np.float32)
                # MediaPipe doesn't expose per-point confidence; use overall detection confidence approximation
                det_conf = float(results.multi_face_landmarks[0].landmark[0].visibility) if hasattr(results.multi_face_landmarks[0].landmark[0], 'visibility') else 0.9
                confidences = np.clip(np.ones(68, dtype=np.float32) * det_conf, 0.0, 1.0)
                return lmks, confidences
            else:
                return np.zeros((68, 2), dtype=np.float32), np.zeros(68, dtype=np.float32)

        if self.use_dlib:
            gray = self.cv2.cvtColor(frame_rgb, self.cv2.COLOR_RGB2GRAY)
            dets = self.detector(gray, 1)
            if len(dets) > 0:
                d = dets[0]
                shape = self.shape_predictor(gray, d)
                coords = np.zeros((68, 2), dtype=np.float32)
                for i in range(68):
                    coords[i, 0] = shape.part(i).x / W
                    coords[i, 1] = shape.part(i).y / H
                confidences = np.ones(68, dtype=np.float32) * 0.8
                return coords, confidences
            else:
                return np.zeros((68, 2), dtype=np.float32), np.zeros(68, dtype=np.float32)

        # Explicit fallback path only when allow_zero_fallback=True.
        if not self._warned_no_detector:
            logger.warning("LandmarkDetectorV2 fallback returning zero landmarks (no detector available).")
            self._warned_no_detector = True
        return np.zeros((68, 2), dtype=np.float32), np.zeros(68, dtype=np.float32)


class TemporalLandmarkTracker:
    """Tracks and smooths landmarks across frames using per‑point Kalman filters and exponential smoothing of confidence."""
    def __init__(self, device='cpu', num_landmarks: int = 68, smooth_alpha: float = 0.6):
        self.device = device
        self.num_landmarks = num_landmarks
        self.smooth_alpha = smooth_alpha
        self.initialized = False
        self.kalman_filters = [KalmanFilter2D(device=device) for _ in range(num_landmarks)]
        self.prev_conf = np.zeros(num_landmarks, dtype=np.float32)

    def reset(self):
        self.initialized = False
        self.prev_conf = np.zeros(self.num_landmarks, dtype=np.float32)
        self.kalman_filters = [KalmanFilter2D(device=self.device) for _ in range(self.num_landmarks)]

    def smooth(self, landmarks: np.ndarray, confidences: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        landmarks: (T, 68, 2) numpy
        confidences: (T, 68) numpy
        Returns smoothed landmarks and smoothed confidences of same shape (T,68,2) and (T,68)
        """
        T = landmarks.shape[0]
        out_lms = np.zeros_like(landmarks, dtype=np.float32)
        out_conf = np.zeros_like(confidences, dtype=np.float32)

        # Initialize filters with first frame if not initialized
        if not self.initialized and T > 0:
            first = landmarks[0]
            first_conf = confidences[0]
            for i in range(self.num_landmarks):
                x0, y0 = float(first[i, 0]), float(first[i, 1])
                if first_conf[i] > 0:
                    self.kalman_filters[i].init_state(x0, y0)
            self.prev_conf = first_conf.copy()
            self.initialized = True

        for t in range(T):
            for i in range(self.num_landmarks):
                meas_x, meas_y = float(landmarks[t, i, 0]), float(landmarks[t, i, 1])
                conf = float(confidences[t, i])
                kf = self.kalman_filters[i]
                if conf > 0 and hasattr(kf, 'x'):
                    kf.predict()
                    kf.update(meas_x, meas_y)
                    sx, sy = kf.state()
                    # exponential smoothing with measured value weighted by confidence
                    prev_x = out_lms[t-1, i, 0] if t > 0 else sx
                    prev_y = out_lms[t-1, i, 1] if t > 0 else sy
                    alpha = self.smooth_alpha * conf + 0.01
                    sm_x = alpha * sx + (1 - alpha) * prev_x
                    sm_y = alpha * sy + (1 - alpha) * prev_y
                    out_lms[t, i, 0] = sm_x
                    out_lms[t, i, 1] = sm_y
                else:
                    # no measurement: predict only or copy previous
                    if hasattr(kf, 'x'):
                        kf.predict()
                        px, py = kf.state()
                        out_lms[t, i, 0] = px
                        out_lms[t, i, 1] = py
                    else:
                        out_lms[t, i, :] = landmarks[t, i, :]

                # smooth confidence
                prev_conf = out_conf[t-1, i] if t > 0 else self.prev_conf[i]
                out_conf[t, i] = prev_conf * (1 - 0.5) + conf * 0.5

        # clamp coords to [0,1]
        out_lms = np.clip(out_lms, 0.0, 1.0)
        return out_lms, out_conf


class FacialAnchorModule(nn.Module):
    """
    Complete facial anchoring system combining extraction, embedding, and constraints.
    """
    def __init__(
        self, 
        hidden_size: int = 1024,
        latent_channels: int = 16,
        num_landmarks: int = 68,
        anchor_weight: float = 0.15
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.anchor_weight = anchor_weight
        
        self.embedder = FacialAnchorEmbedder(hidden_size, num_landmarks)
        self.constrainer = FacialAnchorConstrainer(hidden_size, latent_channels)
        # New detector + temporal tracker
        self.landmark_detector = LandmarkDetectorV2(device='cuda' if torch.cuda.is_available() else 'cpu')
        self.landmark_tracker = TemporalLandmarkTracker(device='cuda' if torch.cuda.is_available() else 'cpu', num_landmarks=num_landmarks)
        
    def forward(
        self,
        latents: torch.Tensor,
        video_frames: Optional[torch.Tensor] = None,
        landmarks: Optional[torch.Tensor] = None,
        spatial_shape: Tuple[int, int] = (60, 104)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            latents: [B, C, T, H, W] latent features
            video_frames: [B, T, H, W, 3] RGB frames (if None, use pre-extracted landmarks)
            landmarks: [B, T, 68, 2] pre-extracted facial landmarks
            spatial_shape: shape of latent space
            
        Returns:
            constrained_latents: [B, C, T, H, W]
            anchor_embed: [B, hidden_size]
            region_attn: [B, 68]
        """
        B, C, T, H, W = latents.shape
        
        # Extract or use provided landmarks
        if landmarks is None and video_frames is not None:
            # video_frames: [B, T, H, W, 3]
            Bf, Tf, Hf, Wf, Cf = video_frames.shape
            landmarks_np = np.zeros((Bf, Tf, 68, 2), dtype=np.float32)
            confs_np = np.zeros((Bf, Tf, 68), dtype=np.float32)
            for b in range(Bf):
                for t in range(Tf):
                    frame = (video_frames[b, t] * 255).byte().cpu().numpy()
                    lmks, confs = self.landmark_detector(frame)
                    landmarks_np[b, t] = lmks
                    confs_np[b, t] = confs

            # Temporal smoothing per batch item
            smoothed_landmarks = np.zeros_like(landmarks_np)
            smoothed_confs = np.zeros_like(confs_np)
            for b in range(Bf):
                sm_lm, sm_cf = self.landmark_tracker.smooth(landmarks_np[b], confs_np[b])
                smoothed_landmarks[b] = sm_lm
                smoothed_confs[b] = sm_cf

            landmarks = torch.from_numpy(smoothed_landmarks).to(latents.device)
            confs = torch.from_numpy(smoothed_confs).to(latents.device)

        elif landmarks is None:
            # No landmarks provided, return unmodified latents
            anchor_embed = torch.zeros(B, self.hidden_size, device=latents.device)
            region_attn = torch.ones(B, 68, device=latents.device) / 68
            return latents, anchor_embed, region_attn

        # landmarks: [B, T, 68, 2]
        # confs: [B, T, 68] or default to ones
        if 'confs' not in locals():
            # if landmarks provided externally but no confidences, assume full confidence
            confs = torch.ones(landmarks.shape[0], landmarks.shape[1], landmarks.shape[2], device=latents.device)

        # Compute confidence‑weighted average across time for stability
        # weights: [B, T, 68, 1]
        weights = confs.unsqueeze(-1)
        landmarks_weighted = landmarks * weights
        landmarks_sum = landmarks_weighted.sum(dim=1)  # [B, 68, 2]
        weights_sum = weights.sum(dim=1).clamp(min=1e-6)
        landmarks_avg = landmarks_sum / weights_sum

        # Compute region attention from mean confidences
        region_attn = confs.mean(dim=1)  # [B, 68]
        # Normalize region attention
        region_attn = region_attn / (region_attn.sum(dim=1, keepdim=True) + 1e-6)

        # Dynamic anchor weight scaling: higher when overall confidence high, lower when noisy
        mean_conf = confs.mean(dim=(1,2))  # [B]
        # scale between 0.05..0.6 around base anchor_weight
        dynamic_anchor_weights = (self.anchor_weight * (0.5 + mean_conf)).clamp(min=0.05, max=0.6)

        # Embed landmarks
        anchor_embed, _ = self.embedder(landmarks_avg)  # [B, hidden_size]

        # Apply constraints to latents
        constrained_latents = self.constrainer(
            latents, anchor_embed, region_attn, spatial_shape
        )

        # Blend: per-batch dynamic anchor weight
        out = torch.empty_like(latents)
        for b in range(B):
            aw = dynamic_anchor_weights[b].item() if dynamic_anchor_weights.numel() > 1 else float(dynamic_anchor_weights)
            out[b:b+1] = aw * constrained_latents[b:b+1] + (1 - aw) * latents[b:b+1]

        return out, anchor_embed, region_attn
