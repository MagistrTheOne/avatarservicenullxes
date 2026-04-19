"""HTTP control-plane routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Gauge, generate_latest

from .. import __version__
from ..config import Settings, get_settings
from ..inference.runtime_base import AvatarRuntime
from ..sessions.session_manager import SessionManager
from .errors import ApiError
from .schemas import (
    CreateSessionRequest,
    CreateSessionResponse,
    HealthResponse,
    ReadyResponse,
    SessionSnapshot,
)


def _nvenc_available() -> bool:
    try:
        import av

        av.CodecContext.create("h264_nvenc", "w")
        return True
    except Exception:
        return False


def _gpu_visible() -> bool:
    try:
        import torch  # type: ignore[import-not-found]

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def create_api_router(
    manager: SessionManager,
    runtime: AvatarRuntime,
) -> APIRouter:
    router = APIRouter()

    # Prometheus registry: lightweight — we don't run a separate port.
    registry = CollectorRegistry()
    g_active = Gauge(
        "avatar_active_sessions",
        "Number of active avatar sessions on this pod.",
        registry=registry,
    )
    g_model_loaded = Gauge(
        "avatar_model_loaded",
        "1 if the ARACHNE model is fully loaded, 0 otherwise.",
        registry=registry,
    )
    g_frames = Gauge(
        "avatar_frames_published_total",
        "Total avatar frames pushed into the SFU since pod start.",
        registry=registry,
    )

    # ----------------------------------------------------------- health
    @router.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(version=__version__)

    @router.get("/health/ready", response_model=ReadyResponse)
    async def health_ready() -> ReadyResponse:
        active = manager.active
        details: dict[str, Any] = {
            "phase": active.phase if active is not None else "idle",
            "mode": runtime.mode,
        }
        model_loaded = runtime.is_loaded
        gpu_visible = _gpu_visible() if runtime.mode == "real" else True
        nvenc = _nvenc_available() if runtime.mode == "real" else True
        ready = model_loaded and gpu_visible and nvenc
        return ReadyResponse(
            ready=ready,
            arachne_mode=runtime.mode,
            model_loaded=model_loaded,
            gpu_visible=gpu_visible,
            nvenc_available=nvenc,
            active_sessions=manager.active_count,
            details=details,
        )

    # ----------------------------------------------------------- sessions
    @router.post("/sessions", response_model=CreateSessionResponse, status_code=202)
    async def create_session(
        body: CreateSessionRequest,
        settings: Settings = Depends(get_settings),
    ) -> CreateSessionResponse:
        session = await manager.create(body)
        return CreateSessionResponse(
            provider="runpod" if settings.arachne_mode == "real" else "local",
            session_id=session.request.session_id,
            status="ready" if session.phase == "ready" else "starting",
            agent_user_id=session.request.sfu.agent_user_id,
        )

    @router.delete("/sessions/{session_id}", response_model=SessionSnapshot)
    async def stop_session(session_id: str) -> SessionSnapshot:
        session = await manager.stop(session_id)
        return session.snapshot()

    @router.get("/sessions/{session_id}", response_model=SessionSnapshot)
    async def get_session(session_id: str) -> SessionSnapshot:
        session = manager.get(session_id)
        if session is None:
            raise ApiError(
                status_code=404,
                code="session_not_found",
                message=f"no active session with id {session_id}",
            )
        return session.snapshot()

    # ----------------------------------------------------------- metrics
    @router.get("/metrics", response_class=PlainTextResponse)
    async def metrics(_request: Request) -> PlainTextResponse:
        g_active.set(manager.active_count)
        g_model_loaded.set(1 if runtime.is_loaded else 0)
        active = manager.active
        if active is not None:
            g_frames.set(active.snapshot().frames_published)
        body = generate_latest(registry)
        return PlainTextResponse(body, media_type=CONTENT_TYPE_LATEST)

    return router
