"""One interview session — wires OpenAI peer + SFU peer + inference pipeline.

Lifecycle
---------

``initializing`` → ``openai_connecting`` → ``sfu_connecting`` →
``model_loading`` → ``warming_up`` → ``ready`` → ``stopping`` → ``stopped``.

Any failure path short-circuits to ``failed`` with ``last_error`` populated.

The session owns two audio rings:

- ``mic_ring`` (16 kHz): candidate microphone samples, produced by the SFU
  peer, consumed by the OpenAI peer.
- ``tts_ring`` (16 kHz): TTS audio from OpenAI, produced by the OpenAI peer
  (resampled from 24 kHz), consumed by both the SFU agent audio track
  (after another resample to 48 kHz inside the track) and the ARACHNE
  frame pipeline (needs 16 kHz float32 on the way to Wav2Vec2).

Keeping both rings at 16 kHz mono is the minimum common denominator: it's
the native input for Wav2Vec2, and resampling for Opus publishing is cheap.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from ..api.schemas import (
    CreateSessionRequest,
    SessionPhase,
    SessionSnapshot,
)
from ..bus.gateway_client import GatewayClient
from ..config import Settings
from ..encode.video_track import AvatarVideoTrack
from ..inference.frame_pipeline import FramePipeline
from ..inference.identity_bank import IdentityBank, IdentityTokens
from ..inference.image_loader import ReferenceImageError, load_reference_image
from ..inference.runtime_base import AvatarRuntime
from ..logging import get_logger
from ..media.audio_ring import AudioRing
from ..media.openai_peer import OpenAIRealtimePeer
from ..media.sfu_peer import StreamSfuPeer

_log = get_logger(__name__)

# Runtime audio rate is fixed by ARACHNE's Wav2Vec2 input (16 kHz).
RING_SAMPLE_RATE = 16_000


@dataclass
class _Metrics:
    frames_published: int = 0
    audio_chunks_published: int = 0
    inference_p50_ms: float | None = None
    inference_p95_ms: float | None = None
    audio_underruns: int = 0


class AvatarSession:
    def __init__(
        self,
        request: CreateSessionRequest,
        settings: Settings,
        runtime: AvatarRuntime,
        identity_bank: IdentityBank,
        gateway: GatewayClient,
    ) -> None:
        self._request = request
        self._settings = settings
        self._runtime = runtime
        self._identity_bank = identity_bank
        self._gateway = gateway

        self._phase: SessionPhase = "initializing"
        self._created_at = time.time()
        self._ready_at: float | None = None
        self._stopped_at: float | None = None
        self._last_error: str | None = None

        # Rings: both at 16 kHz mono.
        self._mic_ring = AudioRing(sample_rate=RING_SAMPLE_RATE, capacity_seconds=1.0)
        self._tts_ring = AudioRing(sample_rate=RING_SAMPLE_RATE, capacity_seconds=2.0)

        self._video_track: AvatarVideoTrack | None = None
        self._openai_peer: OpenAIRealtimePeer | None = None
        self._sfu_peer: StreamSfuPeer | None = None
        self._pipeline: FramePipeline | None = None
        self._identity: IdentityTokens | None = None

        self._ready_event = asyncio.Event()

    # ----------------------------------------------------------- props
    @property
    def request(self) -> CreateSessionRequest:
        return self._request

    @property
    def phase(self) -> SessionPhase:
        return self._phase

    def snapshot(self) -> SessionSnapshot:
        return SessionSnapshot(
            session_id=self._request.session_id,
            meeting_id=self._request.meeting_id,
            phase=self._phase,
            agent_user_id=self._request.sfu.agent_user_id,
            created_at=self._created_at,
            ready_at=self._ready_at,
            stopped_at=self._stopped_at,
            last_error=self._last_error,
            frames_published=(self._video_track.frames_published() if self._video_track else 0),
            audio_chunks_published=0,  # can be surfaced from _AgentAudioTrack later
            inference_latency_ms_p50=(
                self._pipeline.latency_ewma.p50() if self._pipeline else None
            ),
            inference_latency_ms_p95=(
                self._pipeline.latency_ewma.p95() if self._pipeline else None
            ),
            audio_underruns=(self._pipeline.audio_underruns if self._pipeline else 0),
        )

    # ----------------------------------------------------------- lifecycle
    async def start(self) -> None:
        try:
            # 1) OpenAI peer.
            self._phase = "openai_connecting"
            self._openai_peer = OpenAIRealtimePeer(
                api_key=self._settings.openai_api_key,
                model=self._settings.openai_realtime_model,
                base_url=self._settings.openai_realtime_base_url,
                tts_audio_ring=self._tts_ring,
                mic_audio_ring=self._mic_ring,
                on_event=self._on_openai_event,
            )
            await self._openai_peer.connect(self._request.openai)
            # OpenAI's `oai-events` DataChannel opens asynchronously after ICE
            # finishes; without this wait `apply_session_update` races the open
            # handshake and the session bails out with
            # "RuntimeError: oai-events datachannel is not open".
            await self._openai_peer.wait_until_open(timeout=15.0)
            await self._openai_peer.apply_session_update(self._request.openai)

            # 2) Identity. AI2V mode: we need a reference portrait for ARACHNE.
            #    First check the identity bank — same avatar_key across sessions
            #    means we never re-encode the same face. On a miss we fetch the
            #    portrait (URL or base64), validate, and call prepare_identity.
            #    Only in dev / when the gateway omits the reference do we fall
            #    back to a neutral grey placeholder (the result will look like
            #    a generic averaged face — never ship that to a real interview).
            self._phase = "model_loading"
            cached = self._identity_bank.get(self._request.avatar_key)
            if cached is None:
                if self._request.reference_image is not None:
                    try:
                        portrait = await load_reference_image(self._request.reference_image)
                    except ReferenceImageError as exc:
                        raise RuntimeError(f"reference_image load failed: {exc}") from exc
                else:
                    _log.warning(
                        "avatar_session.no_reference_image",
                        session_id=self._request.session_id,
                        avatar_key=self._request.avatar_key,
                    )
                    portrait = _neutral_portrait()
                cached = await self._runtime.prepare_identity(
                    self._request.avatar_key, portrait
                )
                self._identity_bank.put(cached)
            self._identity = cached

            # 3) Video track + frame pipeline + SFU peer. ARACHNE knobs come
            #    from the per-request `arachne` block; settings provide the
            #    pod-wide defaults (`arachne_resolution`, `arachne_block_num_frames`,
            #    etc.) that the gateway can override per session.
            arachne_cfg = self._request.arachne
            resolution = arachne_cfg.resolution
            width, height = _resolution_wh(resolution)
            self._video_track = AvatarVideoTrack(
                width=width, height=height, fps=self._runtime.output_fps
            )
            self._pipeline = FramePipeline(
                runtime=self._runtime,
                tts_audio_ring=self._tts_ring,
                video_track=self._video_track,
                identity_tokens=self._identity,
                prompt=arachne_cfg.prompt,
                resolution=resolution,
                num_frames_per_block=arachne_cfg.num_frames,
                num_inference_steps=arachne_cfg.num_inference_steps,
                text_guidance_scale=arachne_cfg.text_guidance_scale,
                audio_guidance_scale=arachne_cfg.audio_guidance_scale,
                emotion=self._request.emotion,
            )

            self._phase = "sfu_connecting"
            self._sfu_peer = StreamSfuPeer(
                config=self._request.sfu,
                stream_base_url=self._settings.stream_base_url,
                stream_api_key=self._settings.stream_api_key,
                tts_audio_ring=self._tts_ring,
                mic_audio_ring=self._mic_ring,
                video_track=self._video_track,
                on_subscribed=None,
            )
            await self._sfu_peer.connect()

            # 4) Start the frame pipeline and wait for the first frame before we
            #    tell the gateway we're ready (so the HR sees video the moment the
            #    SSE event lands).
            self._phase = "warming_up"
            self._pipeline.start()
            await self._wait_first_frame(timeout=20.0)

            # 5) Kick the agent off: `response.create` so OpenAI starts speaking.
            await self._openai_peer.request_response()

            self._phase = "ready"
            self._ready_at = time.time()
            self._ready_event.set()
            self._gateway.emit(
                "avatar_ready",
                session_id=self._request.session_id,
                meeting_id=self._request.meeting_id,
                data={
                    "agent_user_id": self._request.sfu.agent_user_id,
                    "resolution": self._settings.arachne_resolution,
                    "fps": self._runtime.output_fps,
                },
            )
        except Exception as exc:
            self._phase = "failed"
            self._last_error = str(exc)
            self._gateway.emit(
                "error",
                session_id=self._request.session_id,
                meeting_id=self._request.meeting_id,
                data={"where": "start", "error": str(exc)[:1024]},
            )
            _log.exception("avatar_session.start.failed", error=str(exc))
            await self.stop()
            raise

    async def stop(self) -> None:
        if self._phase in {"stopping", "stopped"}:
            return
        self._phase = "stopping"
        # Shut down in reverse dependency order.
        if self._pipeline is not None:
            await self._pipeline.stop()
        if self._sfu_peer is not None:
            await self._sfu_peer.close()
        if self._openai_peer is not None:
            try:
                await self._openai_peer.cancel_response()
            except Exception:
                pass
            await self._openai_peer.close()
        await self._tts_ring.close()
        await self._mic_ring.close()
        self._phase = "stopped"
        self._stopped_at = time.time()
        self._gateway.emit(
            "stopped",
            session_id=self._request.session_id,
            meeting_id=self._request.meeting_id,
            data={"reason": "requested"},
        )

    # ----------------------------------------------------------- helpers
    async def _wait_first_frame(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        assert self._pipeline is not None
        while self._pipeline.first_frame_at is None:
            if time.monotonic() > deadline:
                raise TimeoutError("timed out waiting for the first avatar frame")
            await asyncio.sleep(0.05)

    async def _on_openai_event(self, event: dict[str, Any]) -> None:
        """Forward select OpenAI events to the gateway + react internally."""

        etype = event.get("type")
        if etype in {
            "response.output_audio_transcript.delta",
            "response.output_audio_transcript.completed",
        }:
            self._gateway.emit(
                "transcript_delta" if etype.endswith(".delta") else "transcript_completed",
                session_id=self._request.session_id,
                meeting_id=self._request.meeting_id,
                data={"delta": event.get("delta"), "role": "assistant"},
            )
        elif etype == "conversation.item.input_audio_transcription.completed":
            self._gateway.emit(
                "transcript_completed",
                session_id=self._request.session_id,
                meeting_id=self._request.meeting_id,
                data={"delta": event.get("transcript"), "role": "candidate"},
            )
        elif etype == "response.done":
            self._gateway.emit(
                "response_done",
                session_id=self._request.session_id,
                meeting_id=self._request.meeting_id,
                data={"response_id": (event.get("response") or {}).get("id")},
            )
        elif etype == "__transport.closed":
            self._gateway.emit(
                "error",
                session_id=self._request.session_id,
                meeting_id=self._request.meeting_id,
                data={"where": "openai_transport", "state": event.get("state")},
            )


# --------------------------------------------------------------------- helpers


def _resolution_wh(res: str) -> tuple[int, int]:
    return (832, 480) if res == "480p" else (1280, 720)


def _neutral_portrait() -> "object":  # numpy ndarray, but imported lazily
    import numpy as np

    arr = np.full((512, 512, 3), 128, dtype=np.uint8)
    return arr
