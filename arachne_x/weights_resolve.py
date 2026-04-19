"""
Resolve a Hugging Face Hub repo id or local directory to a local weights root for loader.py.

Loader keeps ``local_files_only=True`` on all ``from_pretrained`` calls; this module optionally
downloads a snapshot first so the root is always a real directory on disk.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+$")


def _has_expected_layout(root: Path) -> bool:
    """Heuristic: ARACHNE / LongCat style bundle has tokenizer + vae."""
    return (root / "tokenizer").is_dir() and (root / "vae").is_dir()


def resolve_weights_root(
    path_or_repo: str,
    *,
    allow_hub: bool = False,
    cache_dir: Optional[str] = None,
) -> str:
    """
    Return a local directory path suitable for ``--checkpoint_dir``.

    - If ``path_or_repo`` is an existing directory with ``tokenizer/`` and ``vae/``, returns it.
    - If ``allow_hub`` and the string looks like ``org/name``, downloads via ``snapshot_download``
      into ``cache_dir`` (or HF default cache) and returns that directory.
    - Otherwise raises ``ValueError`` with a short hint.
    """
    raw = path_or_repo.strip().strip('"').strip("'")
    if not raw:
        raise ValueError("checkpoint path or repo id is empty")

    p = Path(raw).expanduser()
    if p.is_dir() and _has_expected_layout(p):
        return str(p.resolve())

    looks_like_repo = _REPO_ID_RE.match(raw) is not None
    if not looks_like_repo:
        if p.is_dir():
            raise ValueError(
                f"Directory exists but does not look like a weights bundle (missing tokenizer/ or vae/): {raw}"
            )
        raise FileNotFoundError(
            f"Not a local weights directory: {raw}. "
            f"Use an existing path or enable Hub download with allow_hub=True for org/model."
        )

    if not allow_hub:
        raise ValueError(
            f"Path not found locally and looks like a HF repo id ({raw}). "
            f"Pass allow_hub_download=True (or set HF snapshot locally first)."
        )

    from huggingface_hub import snapshot_download

    local = snapshot_download(
        repo_id=raw,
        cache_dir=cache_dir,
        local_files_only=False,
    )
    resolved = Path(local)
    if not _has_expected_layout(resolved):
        raise ValueError(
            f"Downloaded repo {raw} at {resolved} does not contain tokenizer/ + vae/. "
            f"Check repo layout vs WeightsLayout."
        )
    return str(resolved.resolve())


def add_resolve_args(parser) -> None:
    """Attach standard CLI flags for checkpoint resolution (optional)."""
    parser.add_argument(
        "--allow_hub_download",
        action="store_true",
        help="If --checkpoint_dir looks like org/model on Hugging Face, download snapshot first.",
    )
    parser.add_argument(
        "--weights_cache_dir",
        type=str,
        default=None,
        help="Optional HF hub cache / local_dir parent for snapshot_download.",
    )
