from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import librosa
import soundfile as sf


class ExternalArgvSpeechSynthesizer:
    """
    Run an external program to synthesize speech.

    ``argv`` is a list of arguments; placeholders ``{out}`` and ``{sample_rate}`` are substituted;
    the utterance is passed as a single trailing argument (not shell-expanded).
    """

    def __init__(self, argv: list[str]) -> None:
        if not argv:
            raise ValueError("external argv must be non-empty")
        self._template = [str(x) for x in argv]

    def synthesize_to_wav(self, text: str, out_path: Path, *, sample_rate: int = 16000) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rendered: list[str] = []
        for part in self._template:
            rendered.append(
                part.replace("{out}", str(out_path)).replace("{sample_rate}", str(sample_rate))
            )
        cmd = rendered + [text.strip() or " "]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        if not out_path.is_file():
            raise FileNotFoundError(f"external TTS did not create {out_path}")
        audio, sr = librosa.load(str(out_path), sr=None, mono=True)
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
            sf.write(str(out_path), audio, sample_rate, subtype="PCM_16")
        return out_path


class ExternalShellSpeechSynthesizer:
    """
    ``command`` is executed with ``shell=True`` after formatting with
    ``text``, ``out`` (quoted path), ``sample_rate``.

    Use only with trusted commands.
    """

    def __init__(self, command_template: str) -> None:
        self._tpl = command_template

    def synthesize_to_wav(self, text: str, out_path: Path, *, sample_rate: int = 16000) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = self._tpl.format(
            text=shlex.quote(text.strip() or " "),
            out=shlex.quote(str(out_path)),
            sample_rate=sample_rate,
        )
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        return out_path
