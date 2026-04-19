from __future__ import annotations

import base64
import hashlib
import io

import numpy as np
import pytest
from PIL import Image

from avatar_service.api.schemas import ReferenceImage
from avatar_service.inference.image_loader import (
    ReferenceImageError,
    load_reference_image,
)


def _png_bytes(color: tuple[int, int, int] = (200, 50, 50), size: int = 64) -> bytes:
    im = Image.new("RGB", (size, size), color)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_base64_decode_roundtrip() -> None:
    raw = _png_bytes()
    ref = ReferenceImage(base64=base64.b64encode(raw).decode("ascii"))
    arr = await load_reference_image(ref)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.uint8
    assert arr.shape == (64, 64, 3)
    # Center pixel should be the fill color.
    assert tuple(int(c) for c in arr[32, 32]) == (200, 50, 50)


@pytest.mark.asyncio
async def test_base64_data_uri_prefix_is_stripped() -> None:
    raw = _png_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    ref = ReferenceImage(base64=f"data:image/png;base64,{encoded}")
    arr = await load_reference_image(ref)
    assert arr.shape == (64, 64, 3)


@pytest.mark.asyncio
async def test_sha256_mismatch_is_rejected() -> None:
    raw = _png_bytes()
    ref = ReferenceImage(
        base64=base64.b64encode(raw).decode("ascii"),
        sha256="0" * 64,
    )
    with pytest.raises(ReferenceImageError):
        await load_reference_image(ref)


@pytest.mark.asyncio
async def test_sha256_match_is_accepted() -> None:
    raw = _png_bytes()
    ref = ReferenceImage(
        base64=base64.b64encode(raw).decode("ascii"),
        sha256=hashlib.sha256(raw).hexdigest(),
    )
    arr = await load_reference_image(ref)
    assert arr.shape == (64, 64, 3)


@pytest.mark.asyncio
async def test_url_and_base64_mutually_exclusive() -> None:
    ref = ReferenceImage(url="https://example.com/x.png", base64="abc")
    with pytest.raises(ReferenceImageError):
        await load_reference_image(ref)


@pytest.mark.asyncio
async def test_either_required() -> None:
    ref = ReferenceImage()
    with pytest.raises(ReferenceImageError):
        await load_reference_image(ref)
