from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Tuple

from arachne_x.speech.protocol import SpeechSynthesizer
from arachne_x.speech.providers.espeak import EspeakSpeechSynthesizer, try_espeak
from arachne_x.speech.providers.external import ExternalArgvSpeechSynthesizer, ExternalShellSpeechSynthesizer
from arachne_x.speech.providers.stub import StubSpeechSynthesizer
from arachne_x.speech.providers.edge_tts import try_edge_tts


def create_synthesizer(name: str, options: Mapping[str, Any] | None = None) -> SpeechSynthesizer:
    """
    Factory for TTS providers. No hard global dependency beyond what each provider needs.

    Supported ``name`` values:
        - ``stub``: always fails when synthesize is called
        - ``edge_tts``: optional ``edge-tts`` package
        - ``espeak``: OS ``espeak-ng`` / ``espeak`` binary
        - ``external_argv``: options must include ``argv`` (list of str with ``{out}``, ``{sample_rate}``)
        - ``external_shell``: options must include ``command`` (str template; trusted input only)
    """
    opts: dict[str, Any] = dict(options or {})
    key = (name or "").strip().lower()

    if key in ("stub", "none", ""):
        return StubSpeechSynthesizer()
    if key in ("edge", "edge_tts", "edgetts"):
        return try_edge_tts(
            voice=str(opts.get("voice", "en-US-AriaNeural")),
            rate=opts.get("rate"),
        )
    if key in ("espeak", "espeak-ng", "espeak_ng"):
        try:
            return EspeakSpeechSynthesizer(binary=opts.get("binary"))
        except RuntimeError:
            return try_espeak(binary=opts.get("binary"))
    if key in ("external", "external_argv"):
        argv = opts.get("argv")
        if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
            raise ValueError("external_argv provider requires options['argv']: list[str]")
        return ExternalArgvSpeechSynthesizer(argv)
    if key == "external_shell":
        cmd = opts.get("command")
        if not isinstance(cmd, str) or not cmd.strip():
            raise ValueError("external_shell provider requires options['command']: str")
        return ExternalShellSpeechSynthesizer(cmd)

    raise ValueError(
        f"Unknown tts_provider {name!r}. "
        f"Choose: stub, edge_tts, espeak, external_argv, external_shell."
    )


def synthesize_text_to_temp_wav(
    text: str,
    *,
    provider: str,
    options: Mapping[str, Any] | None = None,
    sample_rate: int = 16000,
    suffix: str = "_arachne_tts.wav",
) -> Path:
    """Synthesize to a tempfile; caller should delete when done."""
    import os
    import tempfile

    fd, path = tempfile.mkstemp(suffix=suffix, prefix="arachne_tts_")
    os.close(fd)
    out = Path(path)
    syn = create_synthesizer(provider, options)
    syn.synthesize_to_wav(text, out, sample_rate=sample_rate)
    return out


def resolve_avatar_audio(
    *,
    audio_path: str | None,
    speak_text: str | None,
    tts_provider: str | None,
    tts_options: dict[str, Any] | None,
    tts_sample_rate: int,
) -> Tuple[str, str | None]:
    """
    Returns ``(wav_path, temp_path_or_none)``. If ``temp_path_or_none`` is set, caller must os.unlink it.
    """
    import os

    if audio_path and os.path.isfile(audio_path):
        if speak_text and speak_text.strip():
            print("[tts] --audio is set; ignoring --speak_text")
        return audio_path, None
    text = (speak_text or "").strip()
    if not text:
        raise ValueError("Provide --audio or non-empty --speak_text for this avatar mode.")
    if not tts_provider:
        raise ValueError("--speak_text requires --tts_provider (e.g. edge_tts, espeak).")
    tmp = synthesize_text_to_temp_wav(
        text,
        provider=tts_provider,
        options=tts_options,
        sample_rate=tts_sample_rate,
    )
    print(f"[tts] synthesized to {tmp} (provider={tts_provider})")
    return str(tmp), str(tmp)
