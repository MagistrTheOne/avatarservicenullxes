"""aiortc MediaStreamTrack that publishes frames produced by the inference loop.

Design
------
The frame pipeline writes completed frames into an `asyncio.Queue`. The aiortc
PeerConnection pulls from this track via `recv()` at its own (codec-driven)
pace. We therefore run the inference loop on its own clock (30 FPS wall-clock,
driven by `AVClock`) and let this track pace output by time, duplicating the
last frame if the inference loop falls behind — which is exactly what a
real-time video source should do.

Timestamps
----------
aiortc expects `pts` in units of the track's time base. We report 90 kHz clock
(WebRTC convention) and compute `pts = elapsed_sec * 90000`.
"""

from __future__ import annotations

import asyncio
import fractions
from dataclasses import dataclass

import av
import numpy as np
from aiortc import VideoStreamTrack

from ..logging import get_logger

_log = get_logger(__name__)

# WebRTC video clock: 90 kHz per RFC 3550.
VIDEO_CLOCK_RATE = 90_000


@dataclass
class _QueuedFrame:
    rgb: np.ndarray
    inference_ms: float


class AvatarVideoTrack(VideoStreamTrack):
    """Video track backed by an async queue of RGB frames.

    The frame pipeline pushes via `push()`. aiortc calls `recv()` to pull the
    next `VideoFrame`. If the queue stalls, we emit a copy of the last frame
    so the peer sees a continuous stream — crucial for jitter buffers and
    keyframe cadence. The track is FPS-agnostic: PTS is derived from the
    configured fps regardless of the inference backend's actual throughput.
    """

    kind = "video"

    def __init__(self, width: int, height: int, fps: int) -> None:
        super().__init__()
        self._width = width
        self._height = height
        self._fps = fps
        self._frame_period = 1.0 / fps
        self._queue: asyncio.Queue[_QueuedFrame] = asyncio.Queue(maxsize=8)
        self._last_frame: _QueuedFrame | None = None
        self._frame_count = 0
        self._next_pts = 0

    # ----------------------------------------------------------- producer side
    async def push(self, rgb: np.ndarray, inference_ms: float = 0.0) -> None:
        """Publish a new frame. If the queue is full, drop the oldest."""

        qf = _QueuedFrame(rgb=rgb, inference_ms=inference_ms)
        if self._queue.full():
            # Drop the oldest — we prefer freshness over a backlog.
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        await self._queue.put(qf)

    def frames_published(self) -> int:
        return self._frame_count

    # ----------------------------------------------------------- aiortc side
    async def recv(self) -> av.VideoFrame:
        pts, time_base = await self._next_timestamp()
        try:
            qf = await asyncio.wait_for(self._queue.get(), timeout=self._frame_period * 2)
            self._last_frame = qf
        except asyncio.TimeoutError:
            if self._last_frame is None:
                # Nothing to send yet — emit a black frame.
                qf = _QueuedFrame(
                    rgb=np.zeros((self._height, self._width, 3), dtype=np.uint8),
                    inference_ms=0.0,
                )
            else:
                qf = self._last_frame

        frame = av.VideoFrame.from_ndarray(qf.rgb, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base
        self._frame_count += 1
        return frame

    async def _next_timestamp(self) -> tuple[int, fractions.Fraction]:
        time_base = fractions.Fraction(1, VIDEO_CLOCK_RATE)
        if self._next_pts == 0:
            self._next_pts = 0
        else:
            self._next_pts += int(VIDEO_CLOCK_RATE / self._fps)
        return self._next_pts, time_base
