from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


def _pick_attn_implementation(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    if not torch.cuda.is_available():
        return "sdpa"
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        return "sdpa"


@dataclass
class QwenCustomVoiceSettings:
    model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    device_map: str = "cuda:0"
    dtype: str | torch.dtype = torch.bfloat16
    language: str = "English"
    speaker: str = "Ryan"
    instruct: Optional[str] = None
    attn_implementation: Optional[str] = None


class Qwen3CustomVoiceSynthesizer:
    """
    Qwen3-TTS CustomVoice (HF id or local folder). Requires ``pip install qwen-tts soundfile``.

    Reference: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
    """

    _model_cache: Dict[str, Any] = {}

    def __init__(self, settings: QwenCustomVoiceSettings):
        self._s = settings
        attn = _pick_attn_implementation(settings.attn_implementation)
        dtype = settings.dtype
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype) if hasattr(torch, dtype) else torch.bfloat16
        if not torch.cuda.is_available():
            dtype = torch.float32
            device_map = "cpu" if settings.device_map.startswith("cuda") else settings.device_map
        else:
            device_map = settings.device_map
        self._dtype = dtype
        self._device_map = device_map
        self._attn = attn
        self._cache_key = f"{settings.model_id}|{device_map}|{dtype}|{attn}"

    def _model(self):
        if self._cache_key in self.__class__._model_cache:
            return self.__class__._model_cache[self._cache_key]
        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError as e:
            raise ImportError(
                "Qwen TTS provider requires optional packages. Install with:\n"
                "  pip install -r requirements-tts.txt\n"
                "(see https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)"
            ) from e
        model = Qwen3TTSModel.from_pretrained(
            self._s.model_id,
            device_map=self._device_map,
            dtype=self._dtype,
            attn_implementation=self._attn,
        )
        self.__class__._model_cache[self._cache_key] = model
        return model

    def synthesize_to_path(self, text: str, wav_path: str) -> None:
        t = (text or "").strip()
        if not t:
            raise ValueError("SpeechSynthesizer received empty text")
        try:
            import soundfile as sf
        except ImportError as e:
            raise ImportError("WAV I/O requires soundfile (pip install soundfile)") from e

        model = self._model()
        kwargs = {
            "text": t,
            "language": self._s.language,
            "speaker": self._s.speaker,
        }
        if self._s.instruct:
            kwargs["instruct"] = self._s.instruct
        wavs, sr = model.generate_custom_voice(**kwargs)
        wav0 = wavs[0]
        if hasattr(wav0, "cpu"):
            wav0 = wav0.cpu().numpy()
        sf.write(wav_path, wav0, int(sr))
