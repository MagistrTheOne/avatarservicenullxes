"""RFC 7807-style problem responses."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse


class ApiError(Exception):
    """Business-layer error that the route handlers raise to short-circuit a request."""

    def __init__(self, status_code: int, code: str, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}


def problem_response(err: ApiError, request: Request | None = None) -> JSONResponse:
    path = str(request.url.path) if request else ""
    body = {
        "type": f"about:blank#{err.code}",
        "title": err.code.replace("_", " ").title(),
        "status": err.status_code,
        "detail": err.message,
        "instance": path,
    }
    if err.details:
        body["meta"] = err.details
    return JSONResponse(status_code=err.status_code, content=body)
