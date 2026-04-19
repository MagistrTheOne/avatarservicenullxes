"""
LongCat-AudioDiT TTS: text (optional voice prompt) -> WAV at 16 kHz for avatar ``get_audio_embedding``.

Code and weights: https://huggingface.co/meituan-longcat/LongCat-AudioDiT-1B
Upstream repo is vendored at repo root ``LongCat-AudioDiT/``.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

# Avatar / Wav2Vec path expects 16 kHz (see ``pipeline_arachne_x_video_avatar.get_audio_embedding``).
OUTPUT_SAMPLE_RATE = 16000


def _repo_root() -> Path:
    # arachne_x/tts/audiodit.py -> parents[2] == repository root
    return Path(__file__).resolve().parents[2]


def _longcat_audiodit_root() -> Path:
    return _repo_root() / "LongCat-AudioDiT"


def _ensure_longcat_imports() -> None:
    root = _longcat_audiodit_root()
    if not root.is_dir():
        raise FileNotFoundError(
            f"LongCat-AudioDiT not found at {root}. Clone or copy the upstream repo into the ARACHNE-X root."
        )
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)


def _load_longcat_utils():
    _ensure_longcat_imports()
    path = _longcat_audiodit_root() / "utils.py"
    spec = importlib.util.spec_from_file_location("arachne_x_longcat_audiodit_utils", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load LongCat-AudioDiT utils from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass
class LongCatAudioDiTSettings:
    model_id: str = "meituan-longcat/LongCat-AudioDiT-1B"
    device: str = "cuda:0"
    nfe: int = 16
    cfg_strength: float = 4.0
    guidance_method: str = "cfg"  # "cfg" | "apg"
    prompt_audio_path: Optional[str] = None
    prompt_text: Optional[str] = None
    seed: int = 1024


class LongCatAudioDiTSynthesizer:
    """
    Hugging Face LongCat-AudioDiT checkpoint. Requires packages from
    ``requirements-audiodit.txt`` (or LongCat-AudioDiT ``requirements.txt``).
    """

    _model_cache: Dict[str, Any] = {}
    _tokenizer_cache: Dict[str, Any] = {}

    def __init__(self, settings: LongCatAudioDiTSettings):
        self._s = settings
        dev = settings.device
        if not torch.cuda.is_available() and dev.startswith("cuda"):
            dev = "cpu"
        self._device = torch.device(dev)
        self._cache_key = f"{settings.model_id}|{dev}"

    def _model_and_tokenizer(self):
        if self._cache_key in self.__class__._model_cache:
            return self.__class__._model_cache[self._cache_key], self.__class__._tokenizer_cache[
                self._cache_key
            ]
        try:
            _ensure_longcat_imports()
            import audiodit  # noqa: F401 — registers AudioDiTConfig
            from audiodit import AudioDiTModel
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "LongCat-AudioDiT provider requires extra dependencies. Install with:\n"
                "  pip install -r requirements-audiodit.txt\n"
                "See LongCat-AudioDiT/README.md"
            ) from e

        model = AudioDiTModel.from_pretrained(self._s.model_id).to(self._device)
        model.eval()
        if self._device.type == "cuda":
            model.vae.to_half()
        tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder_model)
        self.__class__._model_cache[self._cache_key] = model
        self.__class__._tokenizer_cache[self._cache_key] = tokenizer
        return model, tokenizer

    def synthesize_to_path(self, text: str, wav_path: str) -> None:
        t = (text or "").strip()
        if not t:
            raise ValueError("SpeechSynthesizer received empty text")

        try:
            import librosa
            import soundfile as sf
        except ImportError as e:
            raise ImportError("WAV I/O requires soundfile and librosa.") from e

        torch.backends.cudnn.benchmark = False
        torch.manual_seed(self._s.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self._s.seed)

        lc_utils = _load_longcat_utils()
        normalize_text = lc_utils.normalize_text
        load_audio = lc_utils.load_audio
        approx_duration_from_text = lc_utils.approx_duration_from_text

        model, tokenizer = self._model_and_tokenizer()
        device = self._device
        sr = model.config.sampling_rate
        full_hop = model.config.latent_hop
        max_duration = model.config.max_wav_duration

        gen_text = normalize_text(t)
        prompt_path = self._s.prompt_audio_path
        no_prompt = not (prompt_path and str(prompt_path).strip())

        if not no_prompt:
            if not self._s.prompt_text or not str(self._s.prompt_text).strip():
                raise ValueError(
                    "longcat_audiodit voice prompt requires --audiodit_prompt_text matching the reference audio."
                )
            prompt_text = normalize_text(self._s.prompt_text.strip())
            full_text = f"{prompt_text} {gen_text}"
        else:
            prompt_text = None
            full_text = gen_text

        inputs = tokenizer([full_text], padding="longest", return_tensors="pt")

        if not no_prompt:
            p = Path(prompt_path).expanduser()
            if not p.is_file():
                raise FileNotFoundError(f"audiodit prompt_audio not found: {p}")
            prompt_wav = load_audio(str(p), sr).unsqueeze(0)

            off = 3
            pw = load_audio(str(p), sr)
            if pw.shape[-1] % full_hop != 0:
                pw = F.pad(pw, (0, full_hop - pw.shape[-1] % full_hop))
            pw = F.pad(pw, (0, full_hop * off))
            with torch.no_grad():
                plt = model.vae.encode(pw.unsqueeze(0).to(device))
            if off:
                plt = plt[..., :-off]
            prompt_dur = plt.shape[-1]
        else:
            prompt_wav = None
            prompt_dur = 0

        prompt_time = prompt_dur * full_hop / sr
        dur_sec = approx_duration_from_text(gen_text, max_duration=max_duration - prompt_time)
        if not no_prompt:
            approx_pd = approx_duration_from_text(prompt_text, max_duration=max_duration)
            ratio = np.clip(prompt_time / approx_pd, 1.0, 1.5)
            dur_sec = dur_sec * ratio

        duration = int(dur_sec * sr // full_hop)
        duration = min(duration + prompt_dur, int(max_duration * sr // full_hop))

        gm = self._s.guidance_method.lower().strip()
        if gm not in ("cfg", "apg"):
            raise ValueError(f"guidance_method must be 'cfg' or 'apg', got {self._s.guidance_method!r}")

        with torch.no_grad():
            output = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                prompt_audio=prompt_wav,
                duration=duration,
                steps=self._s.nfe,
                cfg_strength=self._s.cfg_strength,
                guidance_method=gm,
            )

        wav = output.waveform.squeeze().detach().cpu().numpy().astype(np.float32)
        if sr != OUTPUT_SAMPLE_RATE:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=OUTPUT_SAMPLE_RATE)
        sf.write(wav_path, wav, OUTPUT_SAMPLE_RATE)
