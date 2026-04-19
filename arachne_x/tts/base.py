from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SpeechSynthesizer(Protocol):
    """Turns text into a WAV file on disk. Implementations must not import heavy deps at module import time."""

    def synthesize_to_path(self, text: str, wav_path: str) -> None:
        """
        Write a mono or stereo wave file to ``wav_path`` (parent directory must exist).

        Args:
            text: UTF-8 speech content (non-empty after strip).
            wav_path: Absolute or relative path ending in ``.wav``.
        """
        ...
