"""Outbound HTTP client for ``POST /avatar/events`` on the gateway.

The avatar pod reports lifecycle and streaming signals back to the gateway
over a simple signed HTTP call. The gateway in turn mirrors them to the
frontend (SSE ``/interviews/:id/events``).

Delivery semantics are *at-most-once, best-effort*: events are fire-and-
forget from the media pipeline's point of view. A bounded retry (2 attempts
with jittered backoff) smooths over transient blips without blocking the
inference loop, but a hard gateway outage never stalls the avatar — it
just stops seeing avatar_ready / transcripts.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any

import httpx

from ..api.schemas import AvatarEvent, EventType
from ..logging import get_logger

_log = get_logger(__name__)


class GatewayClient:
    def __init__(
        self,
        base_url: str,
        shared_token: str,
        *,
        events_path: str = "/avatar/events",
        timeout_seconds: float = 5.0,
        retry_attempts: int = 2,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._token = shared_token
        self._events_path = events_path if events_path.startswith("/") else f"/{events_path}"
        self._timeout = timeout_seconds
        self._retries = max(1, int(retry_attempts))
        self._client: httpx.AsyncClient | None = None
        self._queue: asyncio.Queue[AvatarEvent] = asyncio.Queue(maxsize=1024)
        self._pump: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def enabled(self) -> bool:
        return bool(self._base and self._token)

    # ----------------------------------------------------------- lifecycle
    async def start(self) -> None:
        if not self.enabled:
            _log.info("gateway.disabled", reason="missing base_url or shared_token")
            return
        self._client = httpx.AsyncClient(timeout=self._timeout)
        self._pump = asyncio.create_task(self._run_pump(), name="gateway-event-pump")

    async def close(self) -> None:
        self._closed = True
        if self._pump is not None:
            self._pump.cancel()
            try:
                await self._pump
            except (asyncio.CancelledError, Exception):
                pass
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None

    # ----------------------------------------------------------- API
    def emit(
        self,
        type_: EventType,
        session_id: str,
        meeting_id: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Non-blocking enqueue. Drops the event if the queue is full and
        logs a warning (bounded memory + never blocks the GPU thread)."""

        if not self.enabled:
            return
        event = AvatarEvent(
            type=type_,
            session_id=session_id,
            meeting_id=meeting_id,
            ts=time.time(),
            data=data or {},
        )
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            _log.warning("gateway.queue_full", dropped_type=type_)

    # ----------------------------------------------------------- pump
    async def _run_pump(self) -> None:
        assert self._client is not None
        while not self._closed:
            try:
                event = await self._queue.get()
            except asyncio.CancelledError:
                return
            await self._post(event)

    async def _post(self, event: AvatarEvent) -> None:
        assert self._client is not None
        url = self._base + self._events_path
        payload = event.model_dump(mode="json")
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        for attempt in range(1, self._retries + 1):
            try:
                resp = await self._client.post(url, headers=headers, json=payload)
                if 200 <= resp.status_code < 300:
                    return
                _log.warning(
                    "gateway.event.http_error",
                    status=resp.status_code,
                    type=event.type,
                    attempt=attempt,
                    body=resp.text[:200],
                )
            except Exception as exc:  # network/TLS/etc.
                _log.warning(
                    "gateway.event.exception",
                    error=str(exc),
                    type=event.type,
                    attempt=attempt,
                )
            if attempt < self._retries:
                # Jittered exponential backoff: 150ms, 300ms, 600ms ... max 2s.
                delay = min(2.0, 0.15 * (2 ** (attempt - 1))) * (0.75 + 0.5 * random.random())
                await asyncio.sleep(delay)
