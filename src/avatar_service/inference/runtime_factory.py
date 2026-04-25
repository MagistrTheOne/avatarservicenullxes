"""Selects the runtime implementation based on Settings.arachne_mode."""

from __future__ import annotations

from ..config import Settings
from .runtime_base import AvatarRuntime


def create_runtime(settings: Settings) -> AvatarRuntime:
    if settings.arachne_mode == "real":
        from .arachne_runtime import ArachneRuntime

        return ArachneRuntime(
            weights_dir=settings.arachne_weights_dir,
            cuda_device=settings.arachne_cuda_device,
            resolution=settings.arachne_resolution,
            warmup_blocks=settings.arachne_warmup_blocks,
            warmup_frames=settings.arachne_warmup_frames,
            default_num_inference_steps=settings.arachne_num_inference_steps,
        )
    if settings.arachne_mode == "infinitetalk":
        from .infinitetalk_runtime import InfiniteTalkRuntime

        return InfiniteTalkRuntime(
            repo_dir=settings.infinitetalk_repo_dir,
            python_bin=settings.infinitetalk_python_bin,
            ckpt_dir=settings.infinitetalk_ckpt_dir,
            wav2vec_dir=settings.infinitetalk_wav2vec_dir,
            infinitetalk_model_dir=settings.infinitetalk_model_dir,
            quant_dir=settings.infinitetalk_quant_dir or None,
            size=settings.infinitetalk_size,
            sample_steps=settings.infinitetalk_sample_steps,
            frame_num=settings.infinitetalk_frame_num,
            motion_frame=settings.infinitetalk_motion_frame,
            mode=settings.infinitetalk_mode,
            text_guidance_scale=settings.infinitetalk_text_guidance_scale,
            audio_guidance_scale=settings.infinitetalk_audio_guidance_scale,
            temp_dir=settings.infinitetalk_temp_dir,
        )
    from ..stub.fake_arachne import FakeArachneRuntime

    return FakeArachneRuntime(resolution=settings.arachne_resolution)
