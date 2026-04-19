from __future__ import annotations

import asyncio

import httpx
import pytest

from avatar_service.main import create_app


@pytest.mark.asyncio
async def test_health_and_ready_in_stub_mode() -> None:
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async with app.router.lifespan_context(app):
            # Loading the stub runtime is instant, but give the startup task one tick.
            await asyncio.sleep(0.05)
            r = await client.get("/health")
            assert r.status_code == 200
            assert r.json() == {"status": "ok", "version": "0.1.0"}

            r2 = await client.get("/health/ready")
            assert r2.status_code == 200
            body = r2.json()
            assert body["arachne_mode"] == "stub"
            assert body["model_loaded"] is True
            assert body["active_sessions"] == 0
