"""Abstract runtime interface.

ARACHNE-X-ULTRA-AVATAR works in **block-streaming mode** (see
``arachne_x.pipeline_arachne_x_video_avatar.generate_streaming_ai2v``):

- one call generates a block of ``num_frames`` (default 93) frames at
  **16 FPS** (not 30),
- accepts an ``audio_stream`` generator yielding float32 chunks at 16 kHz,
- yields ``np.ndarray[H, W, 3] uint8`` frames one by one after the denoising
  loop finishes — per-frame latency inside a block is ~30 ms (streaming VAE
  decode), first-frame latency of a block is ~500-900 ms on an H200 with 8
  distilled steps.

The abstract base below mirrors that API. Both the real and stub runtimes
implement it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

import numpy as np

from .identity_bank import IdentityTokens


@dataclass
class GeneratedFrame:
    """Output of one streaming step: an RGB uint8 image + metadata."""

    image_rgb: np.ndarray  # (H, W, 3) uint8
    frame_index_in_block: int
    block_index: int
    inference_ms: float = 0.0
    is_idle: bool = False  # True if the input audio chunk was silence


@dataclass
class BlockRequest:
    """One block of frames to generate.

    The runtime will consume ``audio_pcm16_16k`` (int16 mono 16 kHz), convert
    it to float32, and feed it to the pipeline as the streaming input. The
    block occupies ``num_frames / 16`` seconds of output video.
    """

    audio_pcm16_16k: np.ndarray
    identity_tokens: IdentityTokens
    prompt: str
    num_frames: int = 93
    num_inference_steps: int = 8
    text_guidance_scale: float = 4.0
    audio_guidance_scale: float = 4.0
    resolution: str = "480p"  # "480p" or "720p"
    emotion: str | None = None
    emotion_intensity: float = 0.0


class AvatarRuntime(ABC):
    """Abstract runtime shared by real and stub implementations."""

    mode: str = "abstract"

    # --------------------------------------------------------------- lifecycle
    @abstractmethod
    async def load(self) -> None:
        """Load weights and run warm-up passes. Must be awaited before inference."""

    @abstractmethod
    async def aclose(self) -> None:
        """Release GPU memory and any background threads."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...

    @property
    @abstractmethod
    def audio_sample_rate(self) -> int: ...

    @property
    @abstractmethod
    def output_fps(self) -> int: ...

    # --------------------------------------------------------------- identity
    @abstractmethod
    async def prepare_identity(
        self,
        avatar_key: str,
        reference_image_rgb: np.ndarray,
    ) -> IdentityTokens:
        """Pre-compute identity tokens for ``avatar_key`` from a reference portrait."""

    # --------------------------------------------------------------- inference
    @abstractmethod
    def infer_block(self, request: BlockRequest) -> AsyncIterator[GeneratedFrame]:
        """Generate one video block.

        Returns an async iterator yielding frames in playback order. The
        implementation must run the actual synchronous tensor work in a
        worker thread so the asyncio event loop can drive the SFU publisher
        while inference is hot on the GPU.
        """
