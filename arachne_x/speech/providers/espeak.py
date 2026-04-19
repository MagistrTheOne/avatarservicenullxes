from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import librosa
import soundfile as sf


class EspeakSpeechSynthesizer:
    """Local espeak-ng / espeak if available on PATH (no Python package)."""

    def __init__(self, binary: str | None = None) -> None:
        self._bin: str | None = None
        for candidate in (binary, "espeak-ng", "espeak"):
            if candidate and shutil.which(candidate):
                self._bin = candidate
                break
        if self._bin is None:
            raise RuntimeError(
                "espeak-ng (or espeak) not found on PATH. Install OS package or use --tts_provider edge_tts."
            )

    def synthesize_to_wav(self, text: str, out_path: Path, *, sample_rate: int = 16000) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            raw = Path(tmp.name)
        try:
            subprocess.run(
                [self._bin, "-w", str(raw), text.strip() or " "],
                check=True,
                capture_output=True,
                text=True,
            )
            audio, sr = librosa.load(str(raw), sr=None, mono=True)
            if sr != sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
            sf.write(str(out_path), audio, sample_rate, subtype="PCM_16")
        finally:
            if raw.is_file():
                raw.unlink()
        return out_path


def try_espeak(binary: str | None = None):
    try:
        return EspeakSpeechSynthesizer(binary=binary)
    except RuntimeError as e:
        from arachne_x.speech.providers.stub import UnsupportedSpeechSynthesizer

        return UnsupportedSpeechSynthesizer(str(e))
