"""Stream Video SFU peer using the official getstream Python SDK.

Replaces the hand-rolled REST/Twirp peer (:mod:`.sfu_peer`) with the SDK's
WebRTC stack (:func:`getstream.video.rtc.join`), which encapsulates the full
SFU signalling: location discovery, coordinator JoinCall, WebSocket+protobuf
SFU JoinFlow, then Twirp HTTP for SetPublisher and ICE trickle.

Why this exists
---------------
Stream's SFU is **not** stateless Twirp. Their JoinFlow requires a WebSocket
session to be established (with a protobuf JoinRequest carrying the SFU token,
client_details, and reconnect strategy) before any Twirp endpoint will
recognise our ``session_id``. The legacy peer in :mod:`.sfu_peer` skipped that
step and therefore the SFU returned ``404 Not Found`` on SetPublisher with no
body. The protocol is closed-source and the protobuf schema changes silently
between releases, so we use the SDK rather than reimplement it.

Public surface
--------------
The class is a drop-in replacement for :class:`.sfu_peer.StreamSfuPeer`:

- ``__init__(config, stream_base_url, tts_audio_ring, mic_audio_ring,
  video_track, *, stream_api_key, stream_api_secret,
  stream_default_location, on_subscribed)``
- ``async connect() -> None``
- ``async close() -> None``

Limitations of this iteration
-----------------------------
Candidate-microphone forwarding (subscribing to the candidate's published
audio and feeding it to ``mic_audio_ring`` so OpenAI Realtime can hear them)
is **not yet wired**. The SDK exposes incoming tracks through
``ConnectionManager`` events plus the ``SubscriptionManager`` in
``getstream.video.rtc.tracks``, which require additional plumbing. The first
milestone is publishing the avatar's audio+video into the call so it shows up
in the candidate's grid. Mic forwarding will land in a follow-up.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import numpy as np
from aiortc import MediaStreamTrack

from getstream import Stream
import getstream.video.rtc as rtc
from getstream.video.rtc.audio_track import AudioStreamTrack
from getstream.video.rtc.track_util import PcmData

from ..api.schemas import SfuJoinConfig
from ..logging import get_logger
from .audio_ring import AudioRing
from .resampler import PcmResampler

_log = get_logger(__name__)

# 20 ms frames at 48 kHz mono — Opus default packetisation.
_FRAME_MS = 20
_OUT_RATE = 48_000


class VisionAgentsPeer:
    """Server-side participant joining a Stream Video call via the SDK."""

    def __init__(
        self,
        config: SfuJoinConfig,
        stream_base_url: str,
        tts_audio_ring: AudioRing,
        mic_audio_ring: AudioRing,
        video_track: MediaStreamTrack,
        *,
        stream_api_key: str,
        stream_api_secret: str,
        stream_default_location: str = "amsterdam",
        on_subscribed: Callable[[], None] | None = None,
    ) -> None:
        if not stream_api_key:
            raise RuntimeError(
                "stream_api_key is empty — set STREAM_API_KEY in the avatar pod env"
            )
        if not stream_api_secret:
            raise RuntimeError(
                "stream_api_secret is empty — set STREAM_API_SECRET in the avatar "
                "pod env (the SDK signs its own user JWT from the secret instead "
                "of relying on the gateway-issued one)"
            )
        self._config = config
        self._tts_ring = tts_audio_ring
        # mic_audio_ring is the destination ring for candidate microphone
        # audio — kept as a member so the future subscriber-side wiring has
        # somewhere to write to.
        self._mic_ring = mic_audio_ring
        self._video_track = video_track
        self._on_subscribed = on_subscribed
        # Coordinator URL is discovered by the SDK from the api_key, so we
        # don't pass `stream_base_url` here. The legacy code needed it
        # because it built REST URLs by hand.
        del stream_base_url
        del stream_default_location

        # `Stream` is the high-level client. It can mint user JWTs from the
        # API secret which the SDK uses internally during join.
        #
        # rtc.join requires an *async* video client because the connection
        # manager awaits client.post(...) results. Stream() returns a sync
        # client whose .post returns a StreamResponse (not a coroutine), so
        # we must call .as_async() to get the AsyncStream variant.
        self._stream = Stream(api_key=stream_api_key, api_secret=stream_api_secret)
        self._async_stream = self._stream.as_async()
        self._call = self._async_stream.video.call(config.call_type, config.call_id)

        # SDK-managed publisher track. We pump PcmData into it from the
        # tts_audio_ring; the SDK encodes Opus and sends it to the SFU.
        self._audio_track = AudioStreamTrack(
            sample_rate=_OUT_RATE, channels=1, format="s16"
        )

        self._conn: rtc.ConnectionManager | None = None
        self._tts_pump_task: asyncio.Task[None] | None = None
        self._closed = False

    # ----------------------------------------------------------- lifecycle
    async def connect(self) -> None:
        """Establish the SFU connection and start publishing tracks."""

        # `rtc.join` returns a ConnectionManager that is itself an async
        # context manager. We open it manually here so `connect()` returns
        # as soon as the SFU handshake completes; `close()` exits the
        # context.
        cm = await rtc.join(
            self._call,
            user_id=self._config.agent_user_id,
            # The gateway StreamProvisioner already created the call via the
            # admin token — passing create=True is harmless (idempotent
            # get_or_create) but we set False to make the dependency
            # explicit.
            create=False,
        )
        await cm.__aenter__()
        self._conn = cm

        # Subscribe to publisher-side state so the rest of the system can
        # observe the connection lifecycle through structured logs.
        @cm.on("connection.state_changed")
        def _on_state(payload: Any) -> None:  # pragma: no cover - telemetry only
            _log.info("sfu.pc.state", state=str(payload))

        @cm.on("track_published")
        def _on_published(payload: Any) -> None:  # pragma: no cover - telemetry only
            _log.info("sfu.track.published", payload=str(payload)[:200])

        # Publish both tracks in a single negotiation.
        await cm.add_tracks(audio=self._audio_track, video=self._video_track)

        # Start the pump that converts ring-buffer PCM into PcmData chunks
        # and writes them into the SDK audio track.
        self._tts_pump_task = asyncio.create_task(
            self._pump_tts_audio(), name="vap-tts-pump"
        )

        _log.info(
            "sfu.joined",
            call_id=self._config.call_id,
            agent_user_id=self._config.agent_user_id,
            backend="vision_agents",
        )
        if self._on_subscribed is not None:
            self._on_subscribed()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._tts_pump_task is not None:
            self._tts_pump_task.cancel()
            try:
                await self._tts_pump_task
            except (asyncio.CancelledError, Exception):
                pass
            self._tts_pump_task = None

        if self._conn is not None:
            try:
                await self._conn.__aexit__(None, None, None)
            except Exception as exc:  # pragma: no cover - best-effort cleanup
                _log.info("sfu.close.error", error=str(exc))
            self._conn = None

        try:
            close = getattr(self._async_stream, "close", None)
            if callable(close):
                result = close()
                if asyncio.iscoroutine(result):
                    await result
        except Exception:
            pass
        try:
            self._stream.close()
        except Exception:
            pass

    # ----------------------------------------------------------- pumps
    async def _pump_tts_audio(self) -> None:
        """Pull PCM from ``tts_audio_ring``, push as PcmData into SDK track.

        The TTS ring is mono int16 at the OpenAI Realtime sample rate
        (typically 24 kHz). Stream's AudioStreamTrack auto-resamples to its
        configured rate, so we don't resample here — we just convert to
        PcmData and write.
        """

        cursor = self._tts_ring.new_reader(start_at_latest=True)
        in_rate = self._tts_ring.sample_rate
        in_chunk = max(1, int(in_rate * _FRAME_MS / 1000.0))
        # Local resampler is unnecessary because PcmData carries sample_rate
        # and the SDK normalises on write(). Keep one around in case the
        # SDK's resampler quality is insufficient for voice — currently
        # disabled.
        _ = PcmResampler  # silence unused import in case we revive it
        try:
            while not self._closed:
                pcm = await cursor.read_exactly(in_chunk, timeout=0.1)
                if pcm is None or pcm.size < in_chunk:
                    pcm = np.zeros(in_chunk, dtype=np.int16)
                if pcm.dtype != np.int16:
                    pcm = pcm.astype(np.int16)
                await self._audio_track.write(
                    PcmData(
                        sample_rate=in_rate,
                        format="s16",
                        samples=pcm,
                        channels=1,
                    )
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _log.info("sfu.tts.pump.stopped", error=str(exc))
