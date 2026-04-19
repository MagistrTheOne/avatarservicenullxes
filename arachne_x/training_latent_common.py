"""
Shared helpers for ARACHNE-X latent training: sample validation and batch collation.
Used by ``scripts/train.py`` and ``scripts/train_lora_avatar.py``.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List

import torch


def validate_latent_sample(sample: Dict[str, Any], *, require_audio: bool, source: str = "") -> Dict[str, torch.Tensor]:
    required = {"latents", "prompt_embeds", "prompt_mask", "timesteps", "noise"}
    missing = required - set(sample.keys())
    if missing:
        raise KeyError(f"{source}missing keys: {sorted(missing)}")
    if require_audio and "audio_embs" not in sample:
        raise KeyError(f"{source}avatar training requires audio_embs")
    return sample  # type: ignore[return-value]


def normalize_prompt_embeds_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Export often stores ``prompt_embeds`` as ``[1, 1, S, D]``. Collate yields ``[B, 1, 1, S, D]``.
    ``CaptionEmbedder`` / y_embedder expects 4D ``[B, 1, S, D]``.
    """
    while x.dim() > 4 and x.size(1) == 1:
        x = x.squeeze(1)
    if x.dim() > 4:
        raise ValueError(
            f"prompt_embeds shape {tuple(x.shape)} cannot be reduced to 4D [B,1,S,D] for CaptionEmbedder "
            "(expected extra axes to be singleton at dim 1, e.g. [B,1,1,S,D] from stacking [1,1,S,D])."
        )
    if x.dim() == 3:
        x = x.unsqueeze(1)
    if x.dim() != 4:
        raise ValueError(f"prompt_embeds must be 4D [B,1,S,D] after normalize, got {tuple(x.shape)}")
    return x


def squeeze_collated_singleton_batch_dim(x: torch.Tensor) -> torch.Tensor:
    """
    ``default_collate`` / ``torch.stack`` turns per-sample ``[1, C, T, H, W]`` into ``[B, 1, C, T, H, W]``.
    LongCat DiT expects ``[B, C, T, H, W]`` (same for stacked ``audio_embs`` when dim 1 is a lone 1).
    """
    if x.dim() == 6 and x.size(1) == 1:
        return x.squeeze(1)
    return x


_KEYS_SQUEEZE = frozenset({"latents", "noise", "audio_embs"})


def collate_latent_samples(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack per-field tensors into a batch dict (same layout as default DataLoader for tensor values)."""
    if not samples:
        raise ValueError("empty batch")
    out: Dict[str, torch.Tensor] = {}
    for k in samples[0]:
        vals = [s[k] for s in samples]
        t = torch.stack(vals, dim=0)
        if k in _KEYS_SQUEEZE:
            t = squeeze_collated_singleton_batch_dim(t)
        elif k == "prompt_embeds":
            t = normalize_prompt_embeds_batch(t)
        out[k] = t
    return out


def decode_wds_sample_pt(sample: Dict[str, Any], *, require_audio: bool) -> Dict[str, torch.Tensor]:
    """Decode WebDataset sample with binary field ``sample.pt`` (torch.save bytes)."""
    raw = sample.get("sample.pt")
    if raw is None:
        raise KeyError("WebDataset sample missing sample.pt")
    buf = io.BytesIO(raw) if isinstance(raw, (bytes, bytearray)) else io.BytesIO(bytes(raw))
    try:
        obj = torch.load(buf, map_location="cpu", weights_only=False)
    except TypeError:
        buf.seek(0)
        obj = torch.load(buf, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"sample.pt must be a dict, got {type(obj)}")
    return validate_latent_sample(obj, require_audio=require_audio, source="")
