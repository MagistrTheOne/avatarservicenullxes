from __future__ import annotations

from pathlib import Path

from arachne_x.speech.protocol import SpeechSynthesizer


class StubSpeechSynthesizer:
    """Placeholder; fails on use. Useful to catch missing provider in tests."""

    def synthesize_to_wav(self, text: str, out_path: Path, *, sample_rate: int = 16000) -> Path:
        raise NotImplementedError(
            "StubSpeechSynthesizer: set --tts_provider to a real backend "
            "(edge_tts, espeak, external_argv, external_shell)."
        )


class UnsupportedSpeechSynthesizer:
    """Raises with a clear message (e.g. optional dependency missing)."""

    def __init__(self, message: str) -> None:
        self._message = message

    def synthesize_to_wav(self, text: str, out_path: Path, *, sample_rate: int = 16000) -> Path:
        raise RuntimeError(self._message)
