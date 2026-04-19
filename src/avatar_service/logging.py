"""structlog wiring.

The service emits one structured log record per event. In `pretty` mode we get
coloured console output for local development; in `json` mode we get a single
line of JSON per event so Grafana Loki / CloudWatch / stdout on RunPod can
aggregate them trivially.

A stable set of top-level keys is guaranteed across every emitted event:

    ts         ISO-8601 UTC timestamp
    level      debug|info|warning|error|critical
    logger     structlog logger name (module path, usually)
    event      short human-readable event name
    ...        additional contextual keys (session_id, meeting_id, ...)
"""

from __future__ import annotations

import logging
import sys
from typing import Any, cast

import structlog
from structlog.typing import EventDict, Processor

from .config import Settings

_SECRET_KEYS = {
    "openai_api_key",
    "stream_api_secret",
    "gateway_shared_token",
    "authorization",
    "cookie",
    "set-cookie",
    "x-api-key",
}


def _redact_secrets(_logger: object, _name: str, event_dict: EventDict) -> EventDict:
    """Mask known-sensitive keys before they hit the sink."""

    for key in list(event_dict.keys()):
        if key.lower() in _SECRET_KEYS and event_dict[key]:
            event_dict[key] = "***"
    return event_dict


def configure_logging(settings: Settings) -> None:
    """Bootstrap stdlib logging + structlog.

    Call once at process startup. Safe to call multiple times (idempotent).
    """

    level = logging.getLevelNamesMapping().get(settings.log_level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
        force=True,
    )

    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True, key="ts"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _redact_secrets,
    ]

    if settings.log_format == "json":
        renderer: Processor = structlog.processors.JSONRenderer(serializer=_json_dumps)
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True, sort_keys=False)

    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _json_dumps(obj: Any, **kwargs: Any) -> str:
    import orjson

    return orjson.dumps(obj).decode("utf-8")


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound to `name` (defaults to caller module)."""

    logger = structlog.get_logger(name) if name else structlog.get_logger()
    return cast(structlog.stdlib.BoundLogger, logger)
