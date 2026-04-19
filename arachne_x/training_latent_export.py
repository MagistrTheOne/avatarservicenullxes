"""
Build one ``LatentDataset`` / ``train.py`` sample dict for avatar (audio-conditioned DiT).

Shared by ``scripts/export_latent_training_sample.py`` and URL batch exporters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import torch
from PIL import Image

from arachne_x.inference_audio import build_avatar_windowed_audio_emb

if TYPE_CHECKING:
    from arachne_x.pipeline_arachne_x_video_avatar import ArachneXVideoAvatarPipeline


@torch.inference_mode()
def build_avatar_latent_training_sample(
    pipe: "ArachneXVideoAvatarPipeline",
    *,
    image: Union[Image.Image, Any],
    audio_path: str,
    prompt: str,
    negative_prompt: str = "",
    resolution: str = "480p",
    num_frames: int = 93,
    seed: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Return CPU float tensors: latents, noise, timesteps, prompt_embeds, prompt_mask, audio_embs.
    ``audio_path`` must be a local path (wav readable by librosa in ``build_avatar_windowed_audio_emb``).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    scale_factor_spatial = pipe.vae_scale_factor_spatial * 2
    if pipe.dit.cp_split_hw is not None:
        scale_factor_spatial *= max(pipe.dit.cp_split_hw)

    height, width = pipe.get_condition_shape(image, resolution, scale_factor_spatial=scale_factor_spatial)
    pipe.check_inputs(prompt, negative_prompt, height, width, scale_factor_spatial)

    nf = int(num_frames)
    if nf % pipe.vae_scale_factor_temporal != 1:
        adj = nf // pipe.vae_scale_factor_temporal * pipe.vae_scale_factor_temporal + 1
        print(f"Adjusted num_frames {nf} -> {adj} (vae_scale_factor_temporal={pipe.vae_scale_factor_temporal})")
        nf = adj

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

    img_t = pipe.video_processor.preprocess(image, height=height, width=width, resize_mode="crop")
    img_t = img_t.to(device=device, dtype=prompt_embeds.dtype)

    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(int(seed))

    z0 = pipe.prepare_latents(
        image=img_t,
        batch_size=1,
        num_channels_latents=pipe.dit.config.in_channels,
        height=height,
        width=width,
        num_frames=nf,
        num_cond_frames=1,
        dtype=torch.float32,
        device=device,
        generator=gen,
    )

    eps = torch.randn(z0.shape, device=z0.device, dtype=torch.float32, generator=gen)
    sched = pipe.scheduler
    n_sched = int(sched.timesteps.shape[0])
    idx_gen = torch.Generator()
    if seed is not None:
        idx_gen.manual_seed(int(seed) + 1)
    idx = int(torch.randint(0, n_sched, (1,), generator=idx_gen).item())
    t = sched.timesteps[idx].view(1).to(device=device, dtype=torch.float32)
    noisy = sched.scale_noise(z0, t, eps)

    wav = build_avatar_windowed_audio_emb(pipe, audio_path, nf, device)
    audio_embs = pipe._prepare_audio_emb_for_dit(
        wav,
        num_frames=nf,
        batch_size=1,
        num_videos_per_prompt=1,
        device=device,
    )

    return {
        "latents": noisy.cpu().to(torch.float32),
        "noise": eps.cpu().to(torch.float32),
        "timesteps": t.cpu().to(torch.float32),
        "prompt_embeds": prompt_embeds.cpu().to(torch.float32),
        "prompt_mask": prompt_attention_mask.cpu(),
        "audio_embs": audio_embs.cpu().to(torch.float32),
    }


@torch.inference_mode()
def export_avatar_latent_training_pt(
    pipe: "ArachneXVideoAvatarPipeline",
    *,
    image: Union[Image.Image, Any],
    audio_path: str,
    prompt: str,
    output_path: str,
    negative_prompt: str = "",
    resolution: str = "480p",
    num_frames: int = 93,
    seed: Optional[int] = None,
    device: Optional[str] = None,
) -> None:
    """Build sample and ``torch.save`` to ``output_path``."""
    import os

    sample = build_avatar_latent_training_sample(
        pipe,
        image=image,
        audio_path=audio_path,
        prompt=prompt,
        negative_prompt=negative_prompt,
        resolution=resolution,
        num_frames=num_frames,
        seed=seed,
        device=device,
    )
    out_abs = os.path.abspath(output_path)
    parent = os.path.dirname(out_abs)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(sample, out_abs)
    print(f"Saved {out_abs} keys={list(sample.keys())} latent_shape={tuple(sample['latents'].shape)}")
