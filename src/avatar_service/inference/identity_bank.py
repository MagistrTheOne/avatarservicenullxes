"""Identity token cache.

ARACHNE-X-ULTRA-AVATAR keeps a per-avatar "identity token bank" (reference
features) that conditions Audio+Image -> Video generation. Computing these
tokens from a reference portrait costs a few hundred milliseconds on the H200;
doing it on every new session is wasteful, so we keep them in a process-local
LRU.

Tokens for a given `avatar_key` are immutable (deterministic from the portrait
bytes), so we can safely share them across sessions. Real implementations are
free to additionally persist them to Redis; the in-memory LRU below is enough
for a single pod.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IdentityTokens:
    """Opaque token blob passed back to the runtime on every inference step.

    The exact shape is runtime-specific (PyTorch tensor for the real ARACHNE
    runtime, numpy array for the stub). The frame pipeline only needs to hold
    and forward it, never to introspect it.
    """

    avatar_key: str
    payload: Any  # tensor or ndarray, owned by this object


class IdentityBank:
    """Thread-safe LRU cache of identity tokens."""

    def __init__(self, capacity: int = 32) -> None:
        self._capacity = capacity
        self._store: OrderedDict[str, IdentityTokens] = OrderedDict()

    def get(self, avatar_key: str) -> IdentityTokens | None:
        if avatar_key not in self._store:
            return None
        self._store.move_to_end(avatar_key)
        return self._store[avatar_key]

    def put(self, tokens: IdentityTokens) -> None:
        self._store[tokens.avatar_key] = tokens
        self._store.move_to_end(tokens.avatar_key)
        while len(self._store) > self._capacity:
            self._store.popitem(last=False)

    def __contains__(self, avatar_key: str) -> bool:
        return avatar_key in self._store

    def __len__(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()
