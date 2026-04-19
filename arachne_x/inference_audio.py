"""
Shared audio embedding windowing for avatar inference / export (matches scripts/infer.py logic).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import librosa
import numpy as np
import torch

if TYPE_CHECKING:
    from arachne_x.pipeline_arachne_x_video_avatar import ArachneXVideoAvatarPipeline


def build_avatar_windowed_audio_emb(
    pipe: "ArachneXVideoAvatarPipeline",
    audio_path: str,
    num_frames: int,
    device: Union[str, torch.device],
    sample_rate: int = 16000,
) -> torch.Tensor:
    """
    Load wav, run ``get_audio_embedding``, build [1, T, W, S, C] windows (same as ``scripts/infer._build_audio_emb``).
    """
    speech_array, sr = librosa.load(audio_path, sr=sample_rate)
    audio_stride = int(getattr(pipe, "vae_scale_factor_temporal", 4))
    audio_stride = max(audio_stride, 1)
    full_audio_emb = pipe.get_audio_embedding(
        speech_array,
        fps=16 * audio_stride,
        device=device,
        sample_rate=sr,
    )
    audio_window = int(getattr(pipe.dit, "audio_window", 5))
    audio_window = max(1, 2 * (audio_window // 2) + 1)
    indices = torch.arange(audio_window, device=full_audio_emb.device) - (audio_window // 2)
    center_indices = torch.arange(
        0,
        audio_stride * num_frames,
        audio_stride,
        device=full_audio_emb.device,
    ).unsqueeze(1) + indices.unsqueeze(0)
    center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb.shape[0] - 1)
    return full_audio_emb[center_indices][None, ...].to(device)
