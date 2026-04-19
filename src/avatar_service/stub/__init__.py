"""Dev-mode alternative backends (CPU-only, no GPU required)."""

from .fake_arachne import FakeArachneRuntime

__all__ = ["FakeArachneRuntime"]
