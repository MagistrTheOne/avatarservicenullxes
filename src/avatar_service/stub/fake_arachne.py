"""CPU-only dev backend for the avatar runtime.

Mirrors :class:`avatar_service.inference.arachne_runtime.ArachneRuntime` so
the rest of the pipeline (OpenAI peer, audio bridge, SFU peer, NVENC, aiortc
video track) can run end-to-end on a developer laptop without an H200.

The fake runtime renders a composite image per frame:

- An identity portrait either loaded from the reference image or derived
  deterministically from ``avatar_key`` (hashed into a gradient).
- A live audio waveform drawn across the bottom that reacts to the
  corresponding audio window. When the window is silence we draw a flat
  baseline and flag the frame ``is_idle=True``.

Output matches the real runtime: **16 FPS, 16 kHz mono audio input**, block
streaming semantics via :meth:`infer_block`.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import AsyncIterator

import numpy as np

from ..inference.identity_bank import IdentityTokens
from ..inference.runtime_base import AvatarRuntime, BlockRequest, GeneratedFrame

FAKE_AUDIO_SR = 16_000
FAKE_OUTPUT_FPS = 16


def _gradient_from_key(width: int, height: int, avatar_key: str) -> np.ndarray:
    """Deterministic RGB gradient keyed by `avatar_key`."""

    h = hashlib.sha256(avatar_key.encode("utf-8")).digest()
    c1 = np.array([h[0], h[1], h[2]], dtype=np.float32)
    c2 = np.array([h[3], h[4], h[5]], dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, height, dtype=np.float32).reshape(-1, 1, 1)
    colour = (1.0 - ramp) * c1 + ramp * c2
    frame = np.broadcast_to(colour, (height, width, 3)).astype(np.uint8).copy()
    return frame


def _resize_nn(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if img.shape[0] == h and img.shape[1] == w:
        return img.astype(np.uint8).copy()
    ys = (np.linspace(0, img.shape[0] - 1, h)).astype(np.int64)
    xs = (np.linspace(0, img.shape[1] - 1, w)).astype(np.int64)
    return img[ys[:, None], xs[None, :], :].astype(np.uint8)


class FakeArachneRuntime(AvatarRuntime):
    """In-process fake runtime. No GPU. No flash_attn. No weights."""

    mode = "stub"

    def __init__(self, resolution: str = "480p") -> None:
        self._resolution = resolution
        self._width, self._height = (832, 480) if resolution == "480p" else (1280, 720)
        self._loaded = False

    # ------------------------------------------------------------- lifecycle
    async def load(self) -> None:
        dummy_audio = np.zeros(FAKE_AUDIO_SR // FAKE_OUTPUT_FPS, dtype=np.int16)
        base = _gradient_from_key(self._width, self._height, "warmup")
        for _ in range(2):
            self._draw_waveform(base, dummy_audio)
        self._loaded = True

    async def aclose(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def audio_sample_rate(self) -> int:
        return FAKE_AUDIO_SR

    @property
    def output_fps(self) -> int:
        return FAKE_OUTPUT_FPS

    # ------------------------------------------------------------- identity
    async def prepare_identity(
        self,
        avatar_key: str,
        reference_image_rgb: np.ndarray,
    ) -> IdentityTokens:
        if reference_image_rgb is not None and reference_image_rgb.size > 0:
            base = _resize_nn(reference_image_rgb, self._width, self._height)
        else:
            base = _gradient_from_key(self._width, self._height, avatar_key)
        return IdentityTokens(avatar_key=avatar_key, payload=base)

    # ------------------------------------------------------------- inference
    def infer_block(self, request: BlockRequest) -> AsyncIterator[GeneratedFrame]:
        return _FakeBlockIterator(self, request)

    # ------------------------------------------------------------- drawing
    def _draw_waveform(self, frame: np.ndarray, pcm: np.ndarray) -> None:
        h, w, _ = frame.shape
        if pcm.size == 0:
            return
        pcm_f = pcm.astype(np.float32) / 32768.0
        if pcm_f.size >= w:
            idx = np.linspace(0, pcm_f.size - 1, w).astype(np.int64)
            y = pcm_f[idx]
        else:
            reps = int(np.ceil(w / max(1, pcm_f.size)))
            y = np.tile(pcm_f, reps)[:w]
        band_h = max(8, h // 6)
        center_y = h - band_h // 2 - 4
        amplitude = (np.clip(y, -1.0, 1.0) * (band_h // 2)).astype(np.int32)
        for x in range(w):
            yy = center_y + int(amplitude[x])
            yy = max(0, min(h - 1, yy))
            frame[yy, x] = (255, 255, 255)
        frame[center_y, :] = np.minimum(frame[center_y, :].astype(np.int32) + 80, 255).astype(np.uint8)


class _FakeBlockIterator:
    """Async iterator that paces frames at the configured FPS."""

    def __init__(self, runtime: FakeArachneRuntime, request: BlockRequest) -> None:
        self._runtime = runtime
        self._request = request
        self._emitted = 0
        self._period = 1.0 / runtime.output_fps
        self._next_at = time.monotonic()
        self._base_image = request.identity_tokens.payload  # ndarray
        # Slice the block audio into per-frame windows so the waveform "dances".
        self._samples_per_frame = runtime.audio_sample_rate // runtime.output_fps
        self._audio = request.audio_pcm16_16k.astype(np.int16)

    def __aiter__(self) -> "_FakeBlockIterator":
        return self

    async def __anext__(self) -> GeneratedFrame:
        if self._emitted >= self._request.num_frames:
            raise StopAsyncIteration
        # Pace so the fake path feels like a realistic 16 FPS stream.
        now = time.monotonic()
        if self._next_at > now:
            await asyncio.sleep(self._next_at - now)
        self._next_at += self._period

        start = self._emitted * self._samples_per_frame
        end = start + self._samples_per_frame
        window = self._audio[start:end] if end <= self._audio.size else self._audio[start:]
        if window.size < self._samples_per_frame:
            window = np.pad(window, (0, self._samples_per_frame - window.size))

        frame = self._base_image.copy()
        self._runtime._draw_waveform(frame, window)
        is_idle = bool(np.max(np.abs(window.astype(np.int32))) < 200)
        out = GeneratedFrame(
            image_rgb=frame,
            frame_index_in_block=self._emitted,
            block_index=0,
            inference_ms=self._period * 1000.0,
            is_idle=is_idle,
        )
        self._emitted += 1
        return out
