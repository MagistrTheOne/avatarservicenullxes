"""Abstract speech synthesis for avatar conditioning (wav conditioning, not DiT targets)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class SpeechSynthesizer(Protocol):
    """Turns text into a PCM wav file at a sample rate compatible with the avatar stack (default 16 kHz)."""

    def synthesize_to_wav(self, text: str, out_path: Path, *, sample_rate: int = 16000) -> Path:
        """
        Write ``text`` as speech to ``out_path`` (wav).

        Returns:
            Path to the written file (usually ``out_path``).
        """
        ...
