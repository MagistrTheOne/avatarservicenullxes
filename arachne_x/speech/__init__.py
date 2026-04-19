from arachne_x.speech.factory import (
    create_synthesizer,
    resolve_avatar_audio,
    synthesize_text_to_temp_wav,
)
from arachne_x.speech.micro_turn import DEFAULT_STREAM_CHUNK_SECONDS
from arachne_x.speech.protocol import SpeechSynthesizer

__all__ = [
    "SpeechSynthesizer",
    "create_synthesizer",
    "synthesize_text_to_temp_wav",
    "resolve_avatar_audio",
    "DEFAULT_STREAM_CHUNK_SECONDS",
]
