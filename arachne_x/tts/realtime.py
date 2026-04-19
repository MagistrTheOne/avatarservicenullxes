"""
Realtime orchestration contract (WebRTC / digital-employee stack).

The DiT is unchanged: **audio is the master clock**. The server should:
1. Cut each TTS (or microphone) utterance into fixed-duration chunks.
2. For each chunk, run one ``generate_streaming_ai2v`` step (one micro-turn).
3. Mux / stream video frames aligned to the same timeline as audio.

This module documents defaults only; ``src.server.webrtc_server`` may import helpers from
``arachne_x.tts.chunking`` when that package lands in-tree.
"""

# Aligns with Documentation/DIGITAL_EMPLOYEE_SYSTEM_ARCHITECTURE.md micro-turn guidance.
DEFAULT_MICRO_TURN_SECONDS = 0.5
