"""FastAPI application factory + uvicorn entrypoint.

Boot sequence:

1. Load settings (``.env`` + environment).
2. Configure structlog.
3. Create the configured :class:`AvatarRuntime` and call ``load()`` on it —
   on RunPod this takes ~30-60 s as 13.6B DiT weights are paged from the
   Network Volume. We do it inside the app startup hook so ``/health``
   responds immediately with ``status: "ok"`` but ``/health/ready``
   reports ``model_loaded: false`` until the load finishes.
4. Wire the :class:`SessionManager` + :class:`GatewayClient` and mount the
   API router.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .api.errors import ApiError, problem_response
from .api.routes import create_api_router
from .bus.gateway_client import GatewayClient
from .config import Settings, get_settings
from .inference.identity_bank import IdentityBank
from .inference.runtime_factory import create_runtime
from .logging import configure_logging, get_logger
from .sessions.session_manager import SessionManager


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = get_settings()
    configure_logging(settings)
    logger = get_logger(__name__)

    runtime = create_runtime(settings)
    identity_bank = IdentityBank(capacity=32)
    gateway = GatewayClient(
        base_url=settings.gateway_base_url,
        shared_token=settings.gateway_shared_token,
    )
    manager = SessionManager(
        settings=settings,
        runtime=runtime,
        identity_bank=identity_bank,
        gateway=gateway,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await gateway.start()
        # Load model in the background so /health is immediately live.
        load_task = asyncio.create_task(runtime.load(), name="arachne-load")
        logger.info("startup.begin", mode=runtime.mode)
        try:
            yield
        finally:
            logger.info("shutdown.begin")
            load_task.cancel()
            try:
                await load_task
            except (asyncio.CancelledError, Exception):
                pass
            await manager.shutdown()
            await runtime.aclose()
            await gateway.close()
            logger.info("shutdown.done")

    app = FastAPI(
        title="avatarservicenullxes",
        description="NULLXES real-time AI avatar service (ARACHNE-X + OpenAI Realtime + Stream SFU).",
        version="0.1.0",
        lifespan=lifespan,
    )

    if settings.cors_allowed_origins_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allowed_origins_list,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.exception_handler(ApiError)
    async def _api_error_handler(request: Request, exc: ApiError):  # type: ignore[no-untyped-def]
        return problem_response(exc, request)

    app.include_router(create_api_router(manager=manager, runtime=runtime))
    return app


def run_uvicorn() -> None:
    """Convenience: `avatar-service serve` -> uvicorn --host ... --port ..."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "avatar_service.main:create_app",
        factory=True,
        host=settings.http_host,
        port=settings.http_port,
        log_config=None,
        access_log=False,
    )
