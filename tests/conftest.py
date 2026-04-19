from __future__ import annotations

import os

# Ensure stub mode for all tests — no GPU, no weights, no torch.
os.environ.setdefault("ARACHNE_MODE", "stub")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder")
os.environ.setdefault("STREAM_API_KEY", "test-stream-key")
os.environ.setdefault("STREAM_API_SECRET", "test-stream-secret")
os.environ.setdefault("GATEWAY_BASE_URL", "")
os.environ.setdefault("GATEWAY_SHARED_TOKEN", "")
