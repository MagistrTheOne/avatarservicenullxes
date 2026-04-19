"""
Build one ``LatentDataset`` / ``train.py --mode base`` sample dict from a video file.

Uses full-clip VAE encode → normalized latents ``z0``, then flow-matching ``scale_noise``
(aligned with ``training_latent_export.build_avatar_latent_training_sample`` but without audio).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from arachne_x.pipeline_arachne_x_video import retrieve_latents
from arachne_x.training_latent_common import validate_latent_sample

if TYPE_CHECKING:
    from arachne_x.pipeline_arachne_x_video import LongCatVideoPipeline


def _read_video_rgb_pils(path: str, num_frames: int) -> Tuple[List[Image.Image], Tuple[int, int]]:
    """Sample up to ``num_frames`` RGB PIL frames (evenly spaced); pad with last frame if short."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        raise ValueError(f"No frames in video: {path}")

    nf = int(num_frames)
    if nf <= 0:
        cap.release()
        raise ValueError("num_frames must be positive")

    if n >= nf:
        idxs = np.linspace(0, n - 1, num=nf, dtype=np.int64)
    else:
        idxs = np.arange(n, dtype=np.int64)

    frames: List[np.ndarray] = []
    for i in idxs.tolist():
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            continue
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise ValueError(f"Failed to read frames from {path}")

    native_h, native_w = frames[0].shape[0], frames[0].shape[1]
    while len(frames) < nf:
        frames.append(frames[-1].copy())
    frames = frames[:nf]

    pils = [Image.fromarray(f) for f in frames]
    return pils, (native_h, native_w)


@torch.inference_mode()
def build_base_latent_training_sample(
    pipe: "LongCatVideoPipeline",
    *,
    video_path: str,
    prompt: str,
    negative_prompt: str = "",
    resolution: str = "480p",
    num_frames: int = 93,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    vae_sample_mode: Literal["sample", "argmax"] = "argmax",
) -> Dict[str, torch.Tensor]:
    """
    Return CPU float tensors: latents, noise, timesteps, prompt_embeds, prompt_mask (no audio_embs).

    ``latents`` is the noisy tensor at timestep ``t``; ``noise`` is epsilon for MSE against DiT output.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    scale_factor_spatial = pipe.vae_scale_factor_spatial * 2
    if pipe.dit.cp_split_hw is not None:
        scale_factor_spatial *= max(pipe.dit.cp_split_hw)

    nf = int(num_frames)
    if nf % pipe.vae_scale_factor_temporal != 1:
        adj = nf // pipe.vae_scale_factor_temporal * pipe.vae_scale_factor_temporal + 1
        nf = adj

    pils, _ = _read_video_rgb_pils(video_path, nf)
    height, width = pipe.get_condition_shape(pils[0], resolution, scale_factor_spatial=scale_factor_spatial)
    pipe.check_inputs(prompt, negative_prompt, height, width, scale_factor_spatial)

    dit_dtype = pipe.dit.dtype
    prompt_embeds, prompt_attention_mask, _, _ = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=False,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        dtype=dit_dtype,
        device=device,
    )

    video_bcthw = pipe.video_processor.preprocess_video(pils, height=height, width=width)
    video_bcthw = video_bcthw.to(device=device, dtype=pipe.vae.dtype)

    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(int(seed))

    encoded = pipe.vae.encode(video_bcthw)
    z_raw = retrieve_latents(encoded, generator=gen, sample_mode=vae_sample_mode)
    z0 = pipe.normalize_latents(z_raw.float())

    eps = torch.randn(z0.shape, device=z0.device, dtype=torch.float32, generator=gen)
    sched = pipe.scheduler
    n_sched = int(sched.timesteps.shape[0])
    idx_gen = torch.Generator()
    if seed is not None:
        idx_gen.manual_seed(int(seed) + 1)
    idx = int(torch.randint(0, n_sched, (1,), generator=idx_gen).item())
    t = sched.timesteps[idx].view(1).to(device=device, dtype=torch.float32)
    noisy = sched.scale_noise(z0, t, eps)

    sample = {
        "latents": noisy.cpu().to(torch.float32),
        "noise": eps.cpu().to(torch.float32),
        "timesteps": t.cpu().to(torch.float32),
        "prompt_embeds": prompt_embeds.cpu().to(torch.float32),
        "prompt_mask": prompt_attention_mask.cpu(),
    }
    return validate_latent_sample(sample, require_audio=False, source="")


@torch.inference_mode()
def export_base_latent_training_pt(
    pipe: "LongCatVideoPipeline",
    *,
    video_path: str,
    prompt: str,
    output_path: str,
    negative_prompt: str = "",
    resolution: str = "480p",
    num_frames: int = 93,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    vae_sample_mode: Literal["sample", "argmax"] = "argmax",
) -> None:
    """Build sample and ``torch.save`` to ``output_path``."""
    import os

    sample = build_base_latent_training_sample(
        pipe,
        video_path=video_path,
        prompt=prompt,
        negative_prompt=negative_prompt,
        resolution=resolution,
        num_frames=num_frames,
        seed=seed,
        device=device,
        vae_sample_mode=vae_sample_mode,
    )
    out_abs = os.path.abspath(output_path)
    parent = os.path.dirname(out_abs)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(sample, out_abs)
    print(f"Saved {out_abs} keys={list(sample.keys())} latent_shape={tuple(sample['latents'].shape)}")
