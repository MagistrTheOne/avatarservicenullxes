from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import librosa
import soundfile as sf

try:
    import edge_tts
except ImportError:
    edge_tts = None  # type: ignore

from arachne_x.speech.providers.stub import UnsupportedSpeechSynthesizer


class EdgeTTSSpeechSynthesizer:
    """Microsoft Edge online TTS (optional ``pip install edge-tts``)."""

    def __init__(self, voice: str = "en-US-AriaNeural", rate: str | None = None) -> None:
        if edge_tts is None:
            raise ImportError("edge-tts is not installed. Install with: pip install edge-tts")
        self.voice = voice
        self.rate = rate

    async def _async_save(self, text: str, media_path: Path) -> None:
        if self.rate is not None:
            comm = edge_tts.Communicate(text, voice=self.voice, rate=self.rate)
        else:
            comm = edge_tts.Communicate(text, voice=self.voice)
        await comm.save(str(media_path))

    def synthesize_to_wav(self, text: str, out_path: Path, *, sample_rate: int = 16000) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_media = Path(tmp.name)
        try:
            asyncio.run(self._async_save(text.strip() or " ", tmp_media))
            audio, _ = librosa.load(str(tmp_media), sr=sample_rate, mono=True)
            sf.write(str(out_path), audio, sample_rate, subtype="PCM_16")
        finally:
            try:
                tmp_media.unlink(missing_ok=True)
            except TypeError:
                if tmp_media.is_file():
                    tmp_media.unlink()
        return out_path


def try_edge_tts(voice: str = "en-US-AriaNeural", rate: str | None = None):
    if edge_tts is None:
        return UnsupportedSpeechSynthesizer(
            "edge-tts is not installed. Install with: pip install edge-tts"
        )
    return EdgeTTSSpeechSynthesizer(voice=voice, rate=rate)
