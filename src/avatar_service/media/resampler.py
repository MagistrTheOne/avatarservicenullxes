"""Streaming PCM resampler.

Wraps `soxr` in a stateful object that preserves the filter tail across calls,
so we can resample audio chunk by chunk (as it arrives from OpenAI / SFU)
without boundary clicks.

16-bit mono PCM in, 16-bit mono PCM out. Float conversions happen internally.
"""

from __future__ import annotations

import numpy as np

try:
    import soxr

    _HAS_SOXR = True
except ImportError:  # pragma: no cover — soxr is a hard dep, but degrade gracefully
    _HAS_SOXR = False


class PcmResampler:
    """Stateful int16 mono resampler backed by soxr's streaming API.

    Parameters
    ----------
    in_rate
        Input sample rate (Hz). Common: 48000 (Stream SFU / Opus) or 24000 (OpenAI PCM16).
    out_rate
        Output sample rate (Hz). Common: the opposite of the above.
    quality
        soxr quality preset. "QQ" is fastest (~0.5 ms for 20 ms chunks);
        "HQ" is audiophile. For voice we use "LQ" which is ~2 ms for 20 ms
        chunks and imperceptible degradation.
    """

    def __init__(self, in_rate: int, out_rate: int, quality: str = "LQ") -> None:
        if in_rate <= 0 or out_rate <= 0:
            raise ValueError("sample rates must be positive")
        self.in_rate = in_rate
        self.out_rate = out_rate
        self.quality = quality
        if in_rate == out_rate:
            self._stream = None
            return
        if not _HAS_SOXR:
            raise RuntimeError(
                "soxr is required for non-identity resampling; install with `pip install soxr`"
            )
        self._stream = soxr.ResampleStream(in_rate, out_rate, 1, dtype="int16", quality=quality)

    def process(self, pcm_int16: np.ndarray) -> np.ndarray:
        """Resample one chunk and return the output samples.

        Returns an empty array if not enough samples have accumulated yet.
        """

        if pcm_int16.dtype != np.int16:
            pcm_int16 = pcm_int16.astype(np.int16)
        if pcm_int16.ndim != 1:
            pcm_int16 = pcm_int16.reshape(-1)

        if self._stream is None:
            return pcm_int16.copy()
        out = self._stream.resample_chunk(pcm_int16, last=False)
        if out.ndim > 1:
            out = out.reshape(-1)
        if out.dtype != np.int16:
            out = out.astype(np.int16)
        return out

    def flush(self) -> np.ndarray:
        """Flush the filter tail — call once when the input stream ends."""

        if self._stream is None:
            return np.zeros(0, dtype=np.int16)
        out = self._stream.resample_chunk(np.zeros(0, dtype=np.int16), last=True)
        if out.dtype != np.int16:
            out = out.astype(np.int16)
        return out
