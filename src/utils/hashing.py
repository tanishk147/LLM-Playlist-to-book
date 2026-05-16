"""Hashing helpers. All artifacts get a content hash for reproducibility."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path | str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_dict(d: dict[str, Any]) -> str:
    """Stable hash of a dict (sorted keys, compact JSON)."""
    return sha256_text(json.dumps(d, sort_keys=True, separators=(",", ":"), default=str))


def short(h: str, n: int = 12) -> str:
    return h[:n]
