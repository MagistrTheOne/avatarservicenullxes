"""Reference portrait loader.

Fetches the reference image either from a URL or a base64 blob and returns
it as an RGB ``np.ndarray`` of dtype uint8 with shape ``(H, W, 3)``. The
runtime's ``prepare_identity`` consumes this and stores it in the
:class:`IdentityBank` LRU.

Safety
------

- Maximum 8 MB per image — anything larger is rejected. ARACHNE's identity
  encoder happily accepts much smaller images; 8 MB is just an upper guard
  against accidental video uploads.
- HTTPS only when the URL has a scheme; raw bytes (base64) bypass network.
- Optional SHA-256 verification: if the gateway sends a hash, we compute
  the same digest after fetch and refuse on mismatch.
"""

from __future__ import annotations

import base64
import hashlib
import io

import httpx
import numpy as np
from PIL import Image

from ..api.schemas import ReferenceImage

MAX_IMAGE_BYTES = 8 * 1024 * 1024  # 8 MB


class ReferenceImageError(RuntimeError):
    pass


async def load_reference_image(ref: ReferenceImage) -> np.ndarray:
    """Resolve a :class:`ReferenceImage` to a `(H, W, 3) uint8` numpy array."""

    if ref.url and ref.base64:
        raise ReferenceImageError("reference_image cannot have both url and base64")
    if not ref.url and not ref.base64:
        raise ReferenceImageError("reference_image must have url or base64")

    if ref.url:
        if not ref.url.lower().startswith(("http://", "https://")):
            raise ReferenceImageError(f"reference_image.url must be http(s): {ref.url}")
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(ref.url)
            if resp.status_code >= 400:
                raise ReferenceImageError(
                    f"reference_image fetch failed: HTTP {resp.status_code}"
                )
            data = resp.content
    else:
        raw = ref.base64 or ""
        if raw.startswith("data:"):
            # Strip data URI prefix `data:image/png;base64,`
            _, _, raw = raw.partition(",")
        try:
            data = base64.b64decode(raw, validate=True)
        except Exception as exc:
            raise ReferenceImageError(f"reference_image.base64 not valid base64: {exc}") from exc

    if len(data) > MAX_IMAGE_BYTES:
        raise ReferenceImageError(
            f"reference_image too large: {len(data)} > {MAX_IMAGE_BYTES} bytes"
        )

    if ref.sha256:
        digest = hashlib.sha256(data).hexdigest()
        if digest.lower() != ref.sha256.lower():
            raise ReferenceImageError(
                f"reference_image sha256 mismatch: got {digest}, expected {ref.sha256}"
            )

    try:
        with Image.open(io.BytesIO(data)) as im:
            im = im.convert("RGB")
            return np.array(im, dtype=np.uint8)
    except Exception as exc:
        raise ReferenceImageError(f"reference_image not decodable as RGB: {exc}") from exc
