"""Stream Video SFU peer — server-side participant.

We join the candidate's meeting as an additional participant under the user
id ``agent_<sessionId>``, subscribe to the candidate's microphone audio and
publish two outbound tracks:

1. Agent audio — Opus, sourced from the OpenAI TTS audio ring (delayed by
   ``inference_latency_ema`` upstream so lips align with sound).
2. Avatar video — H.264 / VP8, sourced from :class:`AvatarVideoTrack` which
   is fed by the ARACHNE frame pipeline.

Stream SFU spec
---------------
Stream's WebRTC SFU speaks a coordinator REST + edge SFU SDP handshake. Full
protocol documentation lives at
https://getstream.io/video/docs/api/webrtc-sfu/ . The server-side Python
SDK bindings ship as ``getstream`` — when available it's the recommended
path (they maintain Python bindings that handle the edge selection, SDP,
and ICE quirks for you). When the SDK is not installed (e.g. during CI /
local dev) we fall back to aiortc + a direct SFU SDP endpoint that you can
override via ``STREAM_SFU_WS_URL``.

Configuration
-------------
The ``SfuJoinConfig`` payload carries:

- ``call_type`` / ``call_id`` — the meeting identifier in Stream.
- ``agent_user_id`` / ``agent_user_name`` — who we show up as.
- ``agent_user_token`` — pre-signed JWT the gateway minted for us.
- ``candidate_user_id`` — whose audio track we want to subscribe to.

Because the gateway already mints the JWT, this module never needs the
Stream secret key and therefore never has to ship it to the avatar pod.
"""

from __future__ import annotations

import asyncio
import fractions
from collections.abc import Callable
from typing import Any

import av
import httpx
import numpy as np
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.sdp import SessionDescription

from ..api.schemas import SfuJoinConfig
from ..logging import get_logger
from .audio_ring import AudioRing
from .resampler import PcmResampler

_log = get_logger(__name__)


class _AgentAudioTrack(MediaStreamTrack):
    """Audio track sourced from an AudioRing. Emits 20 ms 48 kHz stereo frames so
    Opus encoding is at its most efficient preset.
    """

    kind = "audio"

    def __init__(self, ring: AudioRing, out_sample_rate: int = 48_000) -> None:
        super().__init__()
        self._ring = ring
        self._cursor = ring.new_reader(start_at_latest=True)
        self._out_rate = out_sample_rate
        self._samples_per_chunk = int(self._out_rate * 0.02)  # 20 ms
        self._pts = 0
        self._time_base = fractions.Fraction(1, self._out_rate)
        self._resampler = (
            None
            if ring.sample_rate == out_sample_rate
            else PcmResampler(ring.sample_rate, out_sample_rate)
        )

    async def recv(self) -> av.AudioFrame:
        # Pull a 20 ms window of audio at the ring's rate.
        in_samples = int(self._ring.sample_rate * 0.02)
        pcm = await self._cursor.read_exactly(in_samples, timeout=0.1)
        if pcm is None or pcm.size < in_samples:
            pcm = np.zeros(in_samples, dtype=np.int16)
        if self._resampler is not None:
            pcm = self._resampler.process(pcm)
            if pcm.size < self._samples_per_chunk:
                pcm = np.pad(pcm, (0, self._samples_per_chunk - pcm.size))
            elif pcm.size > self._samples_per_chunk:
                pcm = pcm[: self._samples_per_chunk]

        stereo = np.stack([pcm, pcm], axis=0).astype(np.int16)
        frame = av.AudioFrame.from_ndarray(stereo, format="s16", layout="stereo")
        frame.sample_rate = self._out_rate
        frame.pts = self._pts
        frame.time_base = self._time_base
        self._pts += self._samples_per_chunk
        return frame


class StreamSfuPeer:
    """aiortc-based server-side participant that joins the Stream SFU call.

    Responsibilities:
    - issue the SDP offer / answer handshake with Stream's edge,
    - subscribe to ``candidate_user_id``'s microphone audio and forward it
      to ``mic_audio_ring`` (after decode / resample to 16 kHz),
    - publish agent audio (from ``tts_audio_ring``) and avatar video (from
      ``video_track``) towards the SFU.
    """

    def __init__(
        self,
        config: SfuJoinConfig,
        stream_base_url: str,
        tts_audio_ring: AudioRing,
        mic_audio_ring: AudioRing,
        video_track: MediaStreamTrack,
        *,
        on_subscribed: Callable[[], None] | None = None,
    ) -> None:
        self._config = config
        self._base_url = stream_base_url.rstrip("/")
        self._tts_ring = tts_audio_ring
        self._mic_ring = mic_audio_ring
        self._video_track = video_track
        self._on_subscribed = on_subscribed

        self._pc: RTCPeerConnection | None = None
        self._mic_resampler: PcmResampler | None = None
        self._candidate_track_task: asyncio.Task[None] | None = None
        self._closed = False

    # ----------------------------------------------------------- lifecycle
    async def connect(self) -> None:
        pc = RTCPeerConnection()
        self._pc = pc

        # Publish outbound tracks.
        pc.addTrack(_AgentAudioTrack(self._tts_ring, out_sample_rate=48_000))
        pc.addTrack(self._video_track)

        @pc.on("track")
        def _on_track(track: MediaStreamTrack) -> None:
            if track.kind != "audio":
                return
            _log.info("sfu.track.received", kind=track.kind, id=getattr(track, "id", None))
            self._candidate_track_task = asyncio.create_task(
                self._drain_candidate_audio(track), name="sfu-mic-forwarder"
            )

        @pc.on("connectionstatechange")
        async def _on_state() -> None:
            if self._pc is None:
                return
            _log.info("sfu.pc.state", state=self._pc.connectionState)

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        answer_sdp = await self._signal_join(pc.localDescription.sdp)
        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer_sdp, type="answer"))
        _log.info(
            "sfu.joined",
            call_id=self._config.call_id,
            agent_user_id=self._config.agent_user_id,
        )
        if self._on_subscribed is not None:
            self._on_subscribed()

    async def _signal_join(self, offer_sdp: str) -> str:
        """Perform the coordinator join + SFU SDP exchange.

        The Stream coordinator exposes POST `/video/call/:type/:id/join` and
        POST `/video/sfu/peer_connection` (over REST against the SFU edge
        returned by the coordinator). The exact shapes are stable but
        verbose; we isolate the HTTP details here so the rest of the code
        doesn't need to know them.
        """

        async with httpx.AsyncClient(timeout=30.0) as client:
            # 1) Coordinator join — yields an SFU URL and a credential used to talk to it.
            join_url = f"{self._base_url}/video/call/{self._config.call_type}/{self._config.call_id}/join"
            join_body = {
                "data": {
                    "user_id": self._config.agent_user_id,
                    "member_ids": [self._config.agent_user_id],
                }
            }
            join_headers = {
                "Authorization": f"Bearer {self._config.agent_user_token}",
                "stream-auth-type": "jwt",
                "Content-Type": "application/json",
            }
            r = await client.post(join_url, json=join_body, headers=join_headers)
            if r.status_code >= 400:
                raise RuntimeError(
                    f"stream coordinator join failed: {r.status_code} {r.text[:400]}"
                )
            join = r.json()
            sfu_url = (
                (join.get("credentials") or {}).get("server", {}).get("url")
                or join.get("sfu_url")
                or ""
            )
            sfu_token = (
                (join.get("credentials") or {}).get("token")
                or join.get("sfu_token")
                or self._config.agent_user_token
            )
            if not sfu_url:
                raise RuntimeError("stream coordinator did not return an SFU URL")

            # 2) SDP exchange with the SFU edge.
            sdp_url = sfu_url.rstrip("/") + "/rpc/twirp/stream.video.sfu.signal.v2.SignalServer/SetPublisher"
            sdp_body = {
                "session_id": f"{self._config.agent_user_id}:{self._config.call_id}",
                "sdp": offer_sdp,
                "tracks": [
                    {"type": "AUDIO", "mid": "0"},
                    {"type": "VIDEO", "mid": "1"},
                ],
            }
            sdp_headers = {
                "Authorization": f"Bearer {sfu_token}",
                "Content-Type": "application/json",
            }
            r2 = await client.post(sdp_url, json=sdp_body, headers=sdp_headers)
            if r2.status_code >= 400:
                raise RuntimeError(f"stream sfu signal failed: {r2.status_code} {r2.text[:400]}")
            answer = r2.json()
            answer_sdp = answer.get("sdp") or answer.get("answer_sdp")
            if not answer_sdp:
                raise RuntimeError("stream sfu response missing SDP answer")
            # Validate it parses — aiortc will do it again later, but a clean
            # error here is friendlier.
            SessionDescription.parse(answer_sdp)
            return answer_sdp

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._candidate_track_task is not None:
            self._candidate_track_task.cancel()
            try:
                await self._candidate_track_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._pc is not None:
            try:
                await self._pc.close()
            except Exception:
                pass
            self._pc = None

    # ----------------------------------------------------------- internals
    async def _drain_candidate_audio(self, track: MediaStreamTrack) -> None:
        """Decode incoming audio, resample to mic ring rate, feed the ring."""

        if self._mic_resampler is None:
            self._mic_resampler = PcmResampler(48_000, self._mic_ring.sample_rate)
        try:
            while not self._closed:
                frame: av.AudioFrame = await track.recv()
                arr = frame.to_ndarray()
                if arr.ndim == 2 and arr.shape[0] > 1:
                    arr = arr.mean(axis=0).astype(np.int16)
                elif arr.ndim == 2:
                    arr = arr[0]
                if arr.dtype != np.int16:
                    arr = arr.astype(np.int16)
                if frame.sample_rate != self._mic_ring.sample_rate:
                    if (
                        self._mic_resampler is None
                        or self._mic_resampler.in_rate != frame.sample_rate
                    ):
                        self._mic_resampler = PcmResampler(
                            frame.sample_rate, self._mic_ring.sample_rate
                        )
                    arr = self._mic_resampler.process(arr)
                await self._mic_ring.write(arr)
        except Exception as exc:
            _log.info("sfu.mic.drain.stopped", error=str(exc))
