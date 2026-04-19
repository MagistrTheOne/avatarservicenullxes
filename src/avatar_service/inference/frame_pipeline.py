"""Block-driven frame pipeline.

ARACHNE-X works in block mode: one call generates ``num_frames`` frames at
16 FPS. To keep the avatar continuously speaking we chain blocks back to
back, each one conditioned on the next chunk of agent-audio pulled from
the audio ring.

Responsibilities
----------------

- Read the next ``block_duration_seconds`` worth of 16 kHz mono PCM16 from
  the ring buffer (provided upstream by the TTS publisher after resampling
  from OpenAI's 24 kHz to 16 kHz).
- Issue one :class:`BlockRequest` against the configured runtime.
- Push every yielded :class:`GeneratedFrame` into the
  :class:`AvatarVideoTrack` so aiortc can pace it onto the SFU.
- Record per-frame latency for the session snapshot & Prometheus.

The pipeline runs forever; the session manager stops it via ``cancel()``.
"""

from __future__ import annotations

import asyncio
import math
import time

from ..encode.video_track import AvatarVideoTrack
from ..logging import get_logger
from ..media.audio_ring import AudioRing
from ..sync.av_clock import LatencyEWMA
from .identity_bank import IdentityTokens
from .runtime_base import AvatarRuntime, BlockRequest

_log = get_logger(__name__)


class FramePipeline:
    def __init__(
        self,
        runtime: AvatarRuntime,
        tts_audio_ring: AudioRing,
        video_track: AvatarVideoTrack,
        *,
        identity_tokens: IdentityTokens,
        prompt: str,
        resolution: str = "480p",
        num_frames_per_block: int = 93,
        num_inference_steps: int = 8,
        text_guidance_scale: float = 4.0,
        audio_guidance_scale: float = 4.0,
        emotion: str | None = None,
        emotion_intensity: float = 0.0,
    ) -> None:
        if tts_audio_ring.sample_rate != runtime.audio_sample_rate:
            raise ValueError(
                f"tts_audio_ring.sample_rate ({tts_audio_ring.sample_rate}) != "
                f"runtime.audio_sample_rate ({runtime.audio_sample_rate})"
            )
        self._runtime = runtime
        self._ring = tts_audio_ring
        self._cursor = tts_audio_ring.new_reader(start_at_latest=True)
        self._video_track = video_track
        self._identity = identity_tokens
        self._prompt = prompt
        self._resolution = resolution
        self._num_frames = num_frames_per_block
        self._steps = num_inference_steps
        self._text_cfg = text_guidance_scale
        self._audio_cfg = audio_guidance_scale
        self._emotion = emotion
        self._emotion_intensity = emotion_intensity

        self._task: asyncio.Task[None] | None = None
        self._cancelled = False

        # Metrics.
        self.latency_ewma = LatencyEWMA(alpha=0.2, window=512)
        self.frames_generated = 0
        self.blocks_generated = 0
        self.audio_underruns = 0
        self.first_frame_at: float | None = None

    # ---------- lifecycle ----------
    def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name="frame-pipeline")

    async def stop(self) -> None:
        self._cancelled = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def update_prompt(self, prompt: str) -> None:
        self._prompt = prompt

    def update_emotion(self, emotion: str | None, intensity: float = 0.0) -> None:
        self._emotion = emotion
        self._emotion_intensity = intensity

    # ---------- main loop ----------
    async def _run(self) -> None:
        fps = self._runtime.output_fps
        sr = self._runtime.audio_sample_rate
        block_samples = int(math.ceil(self._num_frames * sr / fps))

        while not self._cancelled:
            # Block on enough audio; if it doesn't arrive within 3x block duration,
            # we synthesize silence so the avatar falls into idle rendering.
            timeout = max(3.0, 3.0 * (self._num_frames / fps))
            pcm = await self._cursor.read_exactly(block_samples, timeout=timeout)
            if pcm is None:
                if self._cancelled:
                    return
                self.audio_underruns += 1
                _log.warning("frame_pipeline.audio_underrun", requested=block_samples)
                # Inject silence block so the avatar doesn't freeze visually.
                import numpy as np

                pcm = np.zeros(block_samples, dtype=np.int16)

            request = BlockRequest(
                audio_pcm16_16k=pcm,
                identity_tokens=self._identity,
                prompt=self._prompt,
                num_frames=self._num_frames,
                num_inference_steps=self._steps,
                text_guidance_scale=self._text_cfg,
                audio_guidance_scale=self._audio_cfg,
                resolution=self._resolution,
                emotion=self._emotion,
                emotion_intensity=self._emotion_intensity,
            )

            block_t0 = time.perf_counter()
            frame_count = 0
            async for gf in self._runtime.infer_block(request):
                if self._cancelled:
                    return
                await self._video_track.push(gf.image_rgb, inference_ms=gf.inference_ms)
                self.frames_generated += 1
                frame_count += 1
                self.latency_ewma.observe(gf.inference_ms)
                if self.first_frame_at is None:
                    self.first_frame_at = time.monotonic()
            self.blocks_generated += 1
            _log.info(
                "frame_pipeline.block_done",
                block=self.blocks_generated,
                frames=frame_count,
                elapsed_ms=round((time.perf_counter() - block_t0) * 1000.0, 1),
                p50_ms=self.latency_ewma.p50(),
                p95_ms=self.latency_ewma.p95(),
            )
