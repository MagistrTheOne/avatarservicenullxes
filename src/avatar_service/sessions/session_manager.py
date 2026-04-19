"""One active session per pod.

A single H200 fits exactly one 13.6B avatar inference at a time
(VRAM ~110-120 GB according to the ARACHNE-X spec), so the manager rejects
new ``POST /sessions`` calls while an existing session is in any phase
other than ``stopped`` or ``failed``.
"""

from __future__ import annotations

import asyncio

from ..api.errors import ApiError
from ..api.schemas import CreateSessionRequest
from ..bus.gateway_client import GatewayClient
from ..config import Settings
from ..inference.identity_bank import IdentityBank
from ..inference.runtime_base import AvatarRuntime
from ..logging import get_logger
from .avatar_session import AvatarSession

_log = get_logger(__name__)


class SessionManager:
    def __init__(
        self,
        settings: Settings,
        runtime: AvatarRuntime,
        identity_bank: IdentityBank,
        gateway: GatewayClient,
    ) -> None:
        self._settings = settings
        self._runtime = runtime
        self._identity_bank = identity_bank
        self._gateway = gateway
        self._active: AvatarSession | None = None
        self._lock = asyncio.Lock()

    @property
    def active(self) -> AvatarSession | None:
        return self._active

    @property
    def active_count(self) -> int:
        if self._active is None:
            return 0
        return 0 if self._active.phase in {"stopped", "failed"} else 1

    def get(self, session_id: str) -> AvatarSession | None:
        if self._active is not None and self._active.request.session_id == session_id:
            return self._active
        return None

    async def create(self, request: CreateSessionRequest) -> AvatarSession:
        async with self._lock:
            if self._active is not None and self._active.phase not in {"stopped", "failed"}:
                raise ApiError(
                    status_code=409,
                    code="pod_busy",
                    message="this pod is already hosting an avatar session",
                    details={
                        "active_session_id": self._active.request.session_id,
                        "active_phase": self._active.phase,
                    },
                )
            if not self._runtime.is_loaded:
                raise ApiError(
                    status_code=503,
                    code="runtime_not_loaded",
                    message="avatar runtime is still loading",
                )
            session = AvatarSession(
                request=request,
                settings=self._settings,
                runtime=self._runtime,
                identity_bank=self._identity_bank,
                gateway=self._gateway,
            )
            self._active = session

        # Kick off the heavy start sequence (OpenAI handshake + SFU join +
        # ARACHNE warmup of the first block) as a background task. The HTTP
        # response is `202 Accepted` and the gateway is notified about the
        # actual readiness through the `avatar_ready` webhook event. This
        # keeps the gateway's create-session HTTP call short (it would
        # otherwise time out after 15 s while the first block compiles).
        asyncio.create_task(
            self._run_start(session),
            name=f"avatar-session-start:{request.session_id}",
        )
        return session

    async def _run_start(self, session: AvatarSession) -> None:
        try:
            await session.start()
        except Exception as exc:
            # start() already called .stop() internally and set phase=failed,
            # plus emitted an `error` event to the gateway. Just log here.
            _log.warning(
                "session_manager.start.failed",
                session_id=session.request.session_id,
                error=str(exc),
            )

    async def stop(self, session_id: str) -> AvatarSession:
        session = self.get(session_id)
        if session is None:
            raise ApiError(
                status_code=404,
                code="session_not_found",
                message=f"no active session with id {session_id}",
            )
        await session.stop()
        return session

    async def shutdown(self) -> None:
        if self._active is not None and self._active.phase not in {"stopped", "failed"}:
            try:
                await self._active.stop()
            except Exception as exc:
                _log.warning("session_manager.shutdown.stop_failed", error=str(exc))
