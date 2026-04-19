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
    from ..stub.fake_arachne import FakeArachneRuntime

    return FakeArachneRuntime(resolution=settings.arachne_resolution)
