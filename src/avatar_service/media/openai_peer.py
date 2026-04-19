"""OpenAI Realtime WebRTC peer, server-side.

The avatar pod opens one `RTCPeerConnection` to OpenAI's `/v1/realtime/calls`
endpoint. The peer:

- sends the candidate's microphone audio on an `m=audio` sendrecv transceiver
  (we wire in a custom `MediaStreamTrack` that reads from the SFU subscription);
- receives the agent's TTS audio on that same transceiver (ontrack -> audio ring);
- opens a DataChannel named `oai-events` and forwards every inbound server
  event to an async callback (for transcripts, tool calls, response lifecycle).

The handshake uses `multipart/form-data` per the current OpenAI Realtime GA
spec — `sdp` field carries the offer, `session` field carries the initial
JSON session config (type/model/audio/instructions/turn_detection). Sending
only `application/sdp` accepts the offer but the server responds with default
audio configuration and often emits no audio at all.
"""

from __future__ import annotations

import asyncio
import fractions
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

import av
import httpx
import numpy as np
from aiortc import (
    MediaStreamTrack,
    RTCConfiguration,
    RTCPeerConnection,
    RTCSessionDescription,
)

from ..api.schemas import OpenAIInitConfig
from ..logging import get_logger
from .audio_ring import AudioRing

_log = get_logger(__name__)

OAI_EVENTS_CHANNEL = "oai-events"
OAI_AUDIO_SAMPLE_RATE = 24_000


OaiEventCallback = Callable[[dict[str, Any]], Awaitable[None]]


def _normalize_sdp(sdp: str) -> str:
    normalized = sdp.replace("\r\n", "\n").replace("\r", "\n").strip()
    return normalized.replace("\n", "\r\n") + "\r\n"


def _build_session_payload(model: str, init: OpenAIInitConfig) -> dict[str, Any]:
    """Compose the JSON the server expects in the `session` form field.

    Shape is the GA WebRTC spec:
      { type: "realtime", model, audio: { input, output: {voice} }, instructions, ... }
    """

    audio: dict[str, Any] = {
        "input": {"format": {"type": "audio/pcm", "rate": OAI_AUDIO_SAMPLE_RATE}},
        "output": {
            "format": {"type": "audio/pcm", "rate": OAI_AUDIO_SAMPLE_RATE},
            "voice": init.voice,
        },
    }
    if init.input_audio_transcription_model:
        audio["input"]["transcription"] = {"model": init.input_audio_transcription_model}

    payload: dict[str, Any] = {
        "type": "realtime",
        "model": model,
        "audio": audio,
        "instructions": init.instructions,
    }
    if init.temperature is not None:
        payload["temperature"] = init.temperature
    return payload


class _MicForwardTrack(MediaStreamTrack):
    """Audio track sourced from an `AudioRing`. Emits 20 ms mono frames at the
    ring's native rate (16 kHz). aiortc/Opus will resample to 48 kHz internally
    if needed.

    PyAV's ``AudioFrame.from_ndarray`` for packed ``s16`` requires
    ``shape=(1, channels*nb_samples)``. The previous ``(2, samples)`` layout
    matched the planar ``s16p`` format and crashed every encode pass with
    ``ValueError: Expected packed array.shape[0] to equal 1 but got 2``.
    """

    kind = "audio"

    def __init__(self, ring: AudioRing) -> None:
        super().__init__()
        self._ring = ring
        self._cursor = ring.new_reader(start_at_latest=True)
        self._samples_per_chunk = int(ring.sample_rate * 0.02)  # 20ms
        self._pts = 0
        self._time_base = fractions.Fraction(1, ring.sample_rate)

    async def recv(self) -> av.AudioFrame:
        target = self._samples_per_chunk
        pcm = await self._cursor.read_exactly(target, timeout=0.1)
        if pcm is None or pcm.size < target:
            pcm = np.zeros(target, dtype=np.int16)
        mono = pcm.astype(np.int16, copy=False).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(mono, format="s16", layout="mono")
        frame.sample_rate = self._ring.sample_rate
        frame.pts = self._pts
        frame.time_base = self._time_base
        self._pts += target
        return frame


class OpenAIRealtimePeer:
    """One-shot peer to OpenAI Realtime. Create, `connect()`, later `close()`."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        tts_audio_ring: AudioRing,
        mic_audio_ring: AudioRing,
        on_event: OaiEventCallback,
    ) -> None:
        self._api_key = api_key
        self._model = model
        # Accept either form for OPENAI_REALTIME_BASE_URL:
        #   https://api.openai.com/v1            (will append /realtime/calls)
        #   https://api.openai.com/v1/realtime   (will append /calls)
        # We normalise to ".../v1/realtime" so the call to /calls is always
        # constructed the same way below.
        normalised = base_url.rstrip("/")
        if not normalised.endswith("/realtime"):
            normalised = normalised + "/realtime"
        self._base_url = normalised
        self._tts_ring = tts_audio_ring
        self._mic_ring = mic_audio_ring
        self._on_event = on_event

        self._pc: RTCPeerConnection | None = None
        self._datachannel: Any = None
        self._mic_track: _MicForwardTrack | None = None
        self._closed = False
        self._remote_call_id: str | None = None
        self._last_event_at: float = 0.0

        self._event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._event_pump_task: asyncio.Task[None] | None = None
        # The `oai-events` DataChannel opens asynchronously after ICE finishes;
        # callers (avatar_session.start) must await `wait_until_open` before
        # the first `session.update`, otherwise sends race the open handshake
        # and raise "datachannel is not open".
        self._dc_open_event: asyncio.Event = asyncio.Event()

    # ------------------------------------------------------------ properties
    @property
    def remote_call_id(self) -> str | None:
        return self._remote_call_id

    @property
    def last_event_at(self) -> float:
        return self._last_event_at

    # ------------------------------------------------------------ lifecycle
    async def connect(self, init: OpenAIInitConfig) -> None:
        """Open the peer. Blocks until the first handshake round-trip completes."""

        self._pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))

        # 1) mic forwarder — sendrecv transceiver, direction sendrecv so we also receive the agent TTS.
        self._mic_track = _MicForwardTrack(self._mic_ring)
        self._pc.addTransceiver("audio", direction="sendrecv")
        self._pc.addTrack(self._mic_track)

        # 2) data channel for server events.
        self._datachannel = self._pc.createDataChannel(OAI_EVENTS_CHANNEL, ordered=True)

        @self._datachannel.on("open")
        def _on_dc_open() -> None:
            _log.info("openai.dc.open")
            self._dc_open_event.set()

        @self._datachannel.on("close")
        def _on_dc_close() -> None:
            _log.info("openai.dc.close")
            self._dc_open_event.clear()

        @self._datachannel.on("message")
        def _on_dc_message(raw: Any) -> None:
            self._on_dc_message(raw)

        @self._pc.on("track")
        def _on_track(track: MediaStreamTrack) -> None:
            if track.kind != "audio":
                return
            _log.info("openai.track.received", kind=track.kind, id=getattr(track, "id", None))
            asyncio.create_task(self._drain_remote_audio(track))

        @self._pc.on("connectionstatechange")
        async def _on_state_change() -> None:
            if self._pc is None:
                return
            state = self._pc.connectionState
            _log.info("openai.pc.state", state=state)
            if state in {"failed", "closed"}:
                # Surface as a pseudo-event so the session manager can react.
                await self._emit_event({"type": "__transport.closed", "state": state})

        # 3) create offer, do HTTP handshake.
        offer = await self._pc.createOffer()
        await self._pc.setLocalDescription(offer)
        answer_sdp = await self._http_handshake(self._pc.localDescription.sdp, init)
        await self._pc.setRemoteDescription(RTCSessionDescription(sdp=answer_sdp, type="answer"))

        # 4) start event pump.
        self._event_pump_task = asyncio.create_task(self._event_pump(), name="oai-event-pump")

    async def _http_handshake(self, offer_sdp: str, init: OpenAIInitConfig) -> str:
        url = f"{self._base_url}/calls"
        session_json = json.dumps(_build_session_payload(self._model, init))
        files = {
            "sdp": (None, _normalize_sdp(offer_sdp), "application/sdp"),
            "session": (None, session_json, "application/json"),
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/sdp",
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, headers=headers, files=files)
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"OpenAI realtime handshake failed: {resp.status_code} {resp.text[:400]}"
                )
            ct = (resp.headers.get("content-type") or "").lower()
            body = resp.text
            if "application/sdp" in ct or "text/plain" in ct or body.strip().startswith("v=0"):
                if not body.strip().startswith("v=0"):
                    raise RuntimeError("OpenAI realtime returned a non-SDP body")
                return _normalize_sdp(body)
            data = resp.json()
            self._remote_call_id = data.get("id") if isinstance(data, dict) else None
            sdp = _extract_answer_sdp(data)
            if not sdp:
                raise RuntimeError("OpenAI realtime response missing SDP answer")
            return _normalize_sdp(sdp)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._event_pump_task is not None:
            self._event_pump_task.cancel()
            try:
                await self._event_pump_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._pc is not None:
            try:
                await self._pc.close()
            except Exception:
                pass
            self._pc = None

    # ------------------------------------------------------------ send
    async def wait_until_open(self, timeout: float = 10.0) -> None:
        """Block until the `oai-events` DataChannel transitions to ``open``.

        OpenAI's WebRTC server only opens the DC after ICE is connected and the
        SDP answer's data section has been finalised. Calling `send_event`
        before that races the open handshake and crashes the session start.
        """

        if self._datachannel is None:
            raise RuntimeError("oai-events datachannel was never created")
        if self._datachannel.readyState == "open":
            return
        try:
            await asyncio.wait_for(self._dc_open_event.wait(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            state = self._datachannel.readyState if self._datachannel else "<no-dc>"
            raise RuntimeError(
                f"oai-events datachannel did not open within {timeout:.1f}s (state={state})"
            ) from exc

    async def send_event(self, event: dict[str, Any]) -> None:
        """Send a client event (e.g. `session.update`, `response.create`) to OpenAI."""

        if self._datachannel is None or self._datachannel.readyState != "open":
            raise RuntimeError("oai-events datachannel is not open")
        self._datachannel.send(json.dumps(event))

    async def request_response(self, instructions: str | None = None) -> None:
        event: dict[str, Any] = {"type": "response.create", "response": {}}
        if instructions:
            event["response"]["instructions"] = instructions
        await self.send_event(event)

    async def cancel_response(self) -> None:
        await self.send_event({"type": "response.cancel"})

    async def apply_session_update(self, init: OpenAIInitConfig) -> None:
        """Send the initial `session.update` to fix voice / format / transcription / instructions."""

        payload = _build_session_payload(self._model, init)
        await self.send_event({"type": "session.update", "session": payload})

    # ------------------------------------------------------------ receive
    def _on_dc_message(self, raw: Any) -> None:
        try:
            text = raw if isinstance(raw, str) else raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
            event = json.loads(text)
        except Exception as exc:
            _log.warning("openai.dc.parse_failed", error=str(exc))
            return
        self._last_event_at = time.monotonic()
        self._event_queue.put_nowait(event)

    async def _event_pump(self) -> None:
        while not self._closed:
            try:
                event = await self._event_queue.get()
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover — defensive
                continue
            await self._emit_event(event)

    async def _emit_event(self, event: dict[str, Any]) -> None:
        try:
            await self._on_event(event)
        except Exception as exc:
            _log.exception("openai.event_callback.failed", error=str(exc), type=event.get("type"))

    async def _drain_remote_audio(self, track: MediaStreamTrack) -> None:
        """Read TTS audio from the remote track and push into the ring at native rate.

        OpenAI sends 24 kHz mono PCM over Opus (renegotiated to match the
        session.audio.output.format). aiortc gives us decoded `av.AudioFrame`s
        already, so we just pull samples, downmix to mono and write.
        """

        try:
            while not self._closed:
                frame: av.AudioFrame = await track.recv()
                arr = frame.to_ndarray()
                if arr.ndim == 2 and arr.shape[0] > 1:
                    # (channels, samples) -> average to mono.
                    arr = arr.mean(axis=0).astype(np.int16)
                elif arr.ndim == 2:
                    arr = arr[0]
                if arr.dtype != np.int16:
                    arr = arr.astype(np.int16)
                # Resample to our ring rate if needed.
                if frame.sample_rate != self._tts_ring.sample_rate:
                    # Fallback resample via soxr; kept off-path.
                    from .resampler import PcmResampler

                    resampler = getattr(self, "_tts_resampler", None)
                    if resampler is None or resampler.in_rate != frame.sample_rate:
                        resampler = PcmResampler(frame.sample_rate, self._tts_ring.sample_rate)
                        self._tts_resampler = resampler
                    arr = resampler.process(arr)
                await self._tts_ring.write(arr)
        except Exception as exc:
            _log.info("openai.track.drain.stopped", error=str(exc))


# --------------------------------------------------------------- helpers


def _extract_answer_sdp(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    answer = payload.get("answer")
    if isinstance(answer, str) and answer.strip():
        return answer
    if isinstance(answer, dict) and isinstance(answer.get("sdp"), str) and answer["sdp"].strip():
        return answer["sdp"]
    if isinstance(payload.get("sdp"), str) and payload["sdp"].strip():
        return payload["sdp"]
    return None
