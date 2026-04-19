"""Text-to-speech bridge for avatar inference (optional dependency per provider)."""

from .base import SpeechSynthesizer
from .factory import create_speech_synthesizer

__all__ = ["SpeechSynthesizer", "create_speech_synthesizer"]
