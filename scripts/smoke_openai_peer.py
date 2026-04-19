"""Smoke-test the OpenAI Realtime peer in isolation.

Opens a peer against the real ``/v1/realtime/calls`` endpoint, sends one
``response.create``, prints the first ~20 events that arrive on the
oai-events DataChannel, and exits.

Requires ``OPENAI_API_KEY`` to be set.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from avatar_service.api.schemas import OpenAIInitConfig
from avatar_service.media.audio_ring import AudioRing
from avatar_service.media.openai_peer import OpenAIRealtimePeer


async def amain() -> int:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("OPENAI_API_KEY is not set", file=sys.stderr)
        return 2

    tts_ring = AudioRing(sample_rate=16_000, capacity_seconds=2.0)
    mic_ring = AudioRing(sample_rate=16_000, capacity_seconds=1.0)
    received: list[dict] = []

    async def on_event(event: dict) -> None:
        received.append(event)
        print(json.dumps({"type": event.get("type"), "keys": list(event.keys())}))
        if len(received) >= 20:
            stop_event.set()

    stop_event = asyncio.Event()
    peer = OpenAIRealtimePeer(
        api_key=api_key,
        model=os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime"),
        base_url=os.environ.get("OPENAI_REALTIME_BASE_URL", "https://api.openai.com/v1"),
        tts_audio_ring=tts_ring,
        mic_audio_ring=mic_ring,
        on_event=on_event,
    )

    init = OpenAIInitConfig(
        instructions="You are a cheerful voice. Say hello in one sentence.",
        voice="alloy",
    )
    try:
        await peer.connect(init)
        await peer.apply_session_update(init)
        await peer.request_response("Say a short hello.")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=20.0)
        except asyncio.TimeoutError:
            print("timed out waiting for events", file=sys.stderr)
    finally:
        await peer.close()
        await tts_ring.close()
        await mic_ring.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(amain()))
