"""Pydantic DTOs for the control-plane HTTP API.

These shapes mirror the "Frontend-Backend Contract" published in the
ARACHNE-X-NULLXES- repository README, extended with a `transport: "webrtc-sfu"`
mode that carries Stream SFU join credentials.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# --- session request ----------------------------------------------------------

Transport = Literal["webrtc-sfu", "hls"]


class OpenAIInitConfig(BaseModel):
    """Minimal subset of the OpenAI Realtime session payload.

    The gateway builds the full instructions + tool configuration. We forward
    this verbatim as the body of the first `session.update` event we send on
    the oai-events DataChannel after handshake completes.
    """

    instructions: str
    voice: str = "alloy"
    input_audio_format: Literal["pcm16"] = "pcm16"
    output_audio_format: Literal["pcm16"] = "pcm16"
    input_audio_transcription_model: str | None = Field(
        default=None,
        description="e.g. 'gpt-4o-mini-transcribe' or 'whisper-1'. None disables input transcription.",
    )
    temperature: float | None = None


class SfuJoinConfig(BaseModel):
    """Credentials the avatar pod uses to join the Stream SFU call as a server-side participant."""

    call_type: str = Field(default="default", description="Stream call type ('default', 'livestream', ...)")
    call_id: str = Field(..., description="The Stream call id (matches meetingId in the gateway).")
    agent_user_id: str = Field(..., description="User id under which the avatar joins, e.g. 'agent_<sessionId>'.")
    agent_user_name: str = Field(default="HR ассистент")
    agent_user_token: str = Field(..., description="Pre-signed Stream user token for agent_user_id.")
    candidate_user_id: str = Field(
        ...,
        description="User id of the candidate participant; we subscribe to their audio.",
    )


class CreateSessionRequest(BaseModel):
    """POST /sessions — request body."""

    meeting_id: str = Field(..., description="Monorepo/gateway meeting identifier.")
    session_id: str = Field(..., description="Stable session id, used for logging and agent_<sessionId>.")
    avatar_key: str = Field(..., description="Identity key passed to ARACHNE (selects the reference portrait).")
    transport: Transport = "webrtc-sfu"
    openai: OpenAIInitConfig
    sfu: SfuJoinConfig
    emotion: str | None = Field(default=None, description="Initial emotion tag (e.g. 'neutral', 'warm').")


class CreateSessionResponse(BaseModel):
    """POST /sessions — response body."""

    provider: Literal["runpod", "local"] = "runpod"
    session_id: str
    status: Literal["starting", "ready"]
    agent_user_id: str


# --- session inspection -------------------------------------------------------

SessionPhase = Literal[
    "initializing",
    "openai_connecting",
    "sfu_connecting",
    "model_loading",
    "warming_up",
    "ready",
    "stopping",
    "stopped",
    "failed",
]


class SessionSnapshot(BaseModel):
    """GET /sessions/{sid} — current session state."""

    session_id: str
    meeting_id: str
    phase: SessionPhase
    agent_user_id: str
    created_at: float
    ready_at: float | None = None
    stopped_at: float | None = None
    last_error: str | None = None

    frames_published: int = 0
    audio_chunks_published: int = 0
    inference_latency_ms_p50: float | None = None
    inference_latency_ms_p95: float | None = None
    audio_underruns: int = 0


# --- events emitted back to the gateway ---------------------------------------

EventType = Literal[
    "avatar_ready",
    "transcript_delta",
    "transcript_completed",
    "response_done",
    "error",
    "stopped",
]


class AvatarEvent(BaseModel):
    """Envelope for anything the pod reports back to the gateway."""

    type: EventType
    session_id: str
    meeting_id: str
    ts: float
    data: dict[str, object] = Field(default_factory=dict)


# --- health -------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str


class ReadyResponse(BaseModel):
    ready: bool
    arachne_mode: str
    model_loaded: bool
    gpu_visible: bool
    nvenc_available: bool
    active_sessions: int
    details: dict[str, object] = Field(default_factory=dict)
