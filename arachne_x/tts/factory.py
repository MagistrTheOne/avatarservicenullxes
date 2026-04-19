from __future__ import annotations

from typing import Optional

from .base import SpeechSynthesizer


def create_speech_synthesizer(
    provider: str,
    *,
    model_id: Optional[str] = None,
    device_map: Optional[str] = None,
    language: str = "English",
    speaker: str = "Ryan",
    instruct: Optional[str] = None,
    attn_implementation: Optional[str] = None,
    audiodit_nfe: Optional[int] = None,
    audiodit_guidance_strength: Optional[float] = None,
    audiodit_guidance_method: Optional[str] = None,
    audiodit_prompt_audio: Optional[str] = None,
    audiodit_prompt_text: Optional[str] = None,
    audiodit_seed: Optional[int] = None,
) -> SpeechSynthesizer:
    """
    Factory for TTS backends. Core ``requirements.txt`` does not pin provider packages.

    Supported ``provider`` values:
    - ``qwen``: Qwen3 CustomVoice via ``qwen-tts`` (install ``requirements-tts.txt``).
    - ``longcat_audiodit`` / ``audiodit``: LongCat-AudioDiT (install ``requirements-audiodit.txt``).
    """
    p = (provider or "").strip().lower()
    if p == "qwen":
        import torch

        from .qwen import Qwen3CustomVoiceSynthesizer, QwenCustomVoiceSettings

        dm = device_map or ("cuda:0" if torch.cuda.is_available() else "cpu")
        mid = model_id or "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        settings = QwenCustomVoiceSettings(
            model_id=mid,
            device_map=dm,
            language=language,
            speaker=speaker,
            instruct=instruct,
            attn_implementation=attn_implementation,
        )
        return Qwen3CustomVoiceSynthesizer(settings)
    if p in ("longcat_audiodit", "audiodit"):
        import torch

        from .audiodit import LongCatAudioDiTSynthesizer, LongCatAudioDiTSettings

        dm = device_map or ("cuda:0" if torch.cuda.is_available() else "cpu")
        mid = model_id or "meituan-longcat/LongCat-AudioDiT-1B"
        settings = LongCatAudioDiTSettings(
            model_id=mid,
            device=dm,
            nfe=16 if audiodit_nfe is None else audiodit_nfe,
            cfg_strength=4.0 if audiodit_guidance_strength is None else audiodit_guidance_strength,
            guidance_method=(audiodit_guidance_method or "cfg"),
            prompt_audio_path=audiodit_prompt_audio,
            prompt_text=audiodit_prompt_text,
            seed=1024 if audiodit_seed is None else audiodit_seed,
        )
        return LongCatAudioDiTSynthesizer(settings)
    raise ValueError(f"Unknown --tts_provider {provider!r}. Supported: qwen, longcat_audiodit")
