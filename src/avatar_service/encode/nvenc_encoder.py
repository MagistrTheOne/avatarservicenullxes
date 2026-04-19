"""H.264 encoder helpers.

Two implementations with the same tiny interface:

- `NvencEncoder`  — uses PyAV with the `h264_nvenc` encoder. Intended for the
                    RunPod H200 pod where the NVIDIA driver exposes NVENC. Low
                    latency presets (`p1`, `tune=ll`, `zerolatency`) match the
                    sub-300 ms mouth-to-display budget.
- `SoftwareH264Encoder` — PyAV + `libx264`. Used on dev boxes without a GPU.
                    Still uses `tune=zerolatency` + `preset=ultrafast` to keep
                    per-frame encode time bounded.

Both accept RGB uint8 numpy arrays and emit encoded H.264 byte strings (NAL
units). They are meant for off-path recording / debugging — the primary
RTP encode is done by aiortc itself (`AvatarVideoTrack` hands it
`av.VideoFrame` objects). When aiortc exposes a public API for custom RTP
senders, this encoder can be plugged into the RTP path directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import av
import numpy as np


class _H264EncoderBase(ABC):
    name: str = "h264"

    def __init__(self, width: int, height: int, fps: int, bitrate_kbps: int) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate_kbps = bitrate_kbps
        self._codec: Any | None = None

    @abstractmethod
    def _make_codec(self) -> Any: ...

    def open(self) -> None:
        if self._codec is None:
            self._codec = self._make_codec()

    def close(self) -> None:
        if self._codec is not None:
            try:
                # Flush.
                packets = list(self._codec.encode(None))
                for _ in packets:
                    pass
            except Exception:
                pass
            self._codec.close()
            self._codec = None

    def encode_rgb(self, rgb: np.ndarray) -> list[bytes]:
        """Encode one RGB frame. Returns a list of NAL units as bytes."""

        if self._codec is None:
            self.open()
        assert self._codec is not None

        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        frame = frame.reformat(format="yuv420p")
        frame.pts = None
        packets = self._codec.encode(frame)
        return [bytes(p) for p in packets if p is not None]


class NvencEncoder(_H264EncoderBase):
    """NVENC-backed encoder. Requires an NVIDIA GPU with NVENC and a PyAV build
    that was linked against an ffmpeg with `--enable-nvenc`.

    On the RunPod pod both are available; on a dev box this will raise when
    `open()` is called, which is fine — use `SoftwareH264Encoder` there.
    """

    name = "h264_nvenc"

    def _make_codec(self) -> Any:
        codec = av.CodecContext.create("h264_nvenc", "w")
        codec.width = self.width
        codec.height = self.height
        codec.pix_fmt = "yuv420p"
        codec.framerate = self.fps
        codec.time_base = av.time_base.Fraction(1, self.fps)
        codec.bit_rate = self.bitrate_kbps * 1000
        codec.options = {
            "preset": "p1",          # fastest NVENC preset
            "tune": "ll",            # low-latency
            "profile": "baseline",   # widest browser compat
            "rc": "cbr",
            "b_frames": "0",
            "g": str(self.fps),      # GOP == 1 s; keyframe each second
            "zerolatency": "1",
            "delay": "0",
        }
        codec.open()
        return codec


class SoftwareH264Encoder(_H264EncoderBase):
    """libx264 fallback. Not as fast but runs anywhere ffmpeg runs."""

    name = "libx264"

    def _make_codec(self) -> Any:
        codec = av.CodecContext.create("libx264", "w")
        codec.width = self.width
        codec.height = self.height
        codec.pix_fmt = "yuv420p"
        codec.framerate = self.fps
        codec.time_base = av.time_base.Fraction(1, self.fps)
        codec.bit_rate = self.bitrate_kbps * 1000
        codec.options = {
            "preset": "ultrafast",
            "tune": "zerolatency",
            "profile": "baseline",
            "g": str(self.fps),
            "b_frames": "0",
        }
        codec.open()
        return codec


def create_encoder(
    width: int,
    height: int,
    fps: int,
    bitrate_kbps: int = 2500,
    prefer_nvenc: bool = True,
) -> _H264EncoderBase:
    """Return the best available encoder for the current process.

    Tries NVENC first when `prefer_nvenc=True`, falls back to libx264 when
    NVENC is not available (e.g. dev machines without the NVIDIA driver).
    """

    if prefer_nvenc:
        try:
            enc = NvencEncoder(width, height, fps, bitrate_kbps)
            enc.open()
            return enc
        except Exception:
            pass
    enc = SoftwareH264Encoder(width, height, fps, bitrate_kbps)
    enc.open()
    return enc
