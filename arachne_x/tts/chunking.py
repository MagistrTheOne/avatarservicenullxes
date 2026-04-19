"""
Fixed-duration audio chunks for streaming avatar inference (audio-driven micro-turns).

Avatar pipeline consumes a generator of float32 mono chunks at ``sample_rate`` Hz, each exactly
``chunk_samples`` long (last chunk zero-padded), matching ``generate_streaming_ai2v`` usage.
"""

from __future__ import annotations

from typing import Iterator

import librosa
import numpy as np


def iter_audio_micro_turns_from_file(
    audio_path: str,
    *,
    chunk_duration_sec: float,
    sample_rate: int = 16000,
) -> Iterator[np.ndarray]:
    """
    Yield mono ``float32`` chunks of length ``chunk_duration_sec * sample_rate``.

    The final chunk is zero-padded to full length so every yield has identical shape.
    """
    if chunk_duration_sec <= 0:
        raise ValueError("chunk_duration_sec must be positive")
    audio, _ = librosa.load(audio_path, sr=sample_rate)
    chunk_samples = int(round(chunk_duration_sec * sample_rate))
    if chunk_samples < 1:
        raise ValueError("chunk_duration_sec too small for sample_rate")
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i : i + chunk_samples]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        yield chunk.astype(np.float32, copy=False)
