"""
Realtime / orchestration hints: avatar streaming consumes fixed-duration PCM chunks.

``generate_streaming_ai2v`` yields one frame per consumed chunk; keep chunk duration stable
so video PTS can follow audio (audio master clock). Do not change DiT — only chunking policy.

Use the same default in ``scripts/infer.py`` streaming and in any WebRTC worker that feeds
``audio_stream`` with the same cadence.
"""

# Seconds per chunk for file-backed streaming in infer.py (and recommended for micro-turn workers).
DEFAULT_STREAM_CHUNK_SECONDS = 0.5
