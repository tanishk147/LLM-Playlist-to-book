"""Filesystem-backed cache, keyed by content hash."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from .hashing import sha256_dict


class DiskCache:
    """Simple keyed cache for expensive deterministic computations.

    Used inside stages to skip re-running work when the input identity hasn't
    changed. Distinct from Anthropic prompt caching (that lives in llm/client.py).
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.pkl"

    def key(self, **kwargs: Any) -> str:
        return sha256_dict(kwargs)

    def get(self, key: str) -> Any | None:
        p = self._path(key)
        if not p.exists():
            return None
        with p.open("rb") as f:
            return pickle.load(f)

    def set(self, key: str, value: Any) -> None:
        with self._path(key).open("wb") as f:
            pickle.dump(value, f)

    def has(self, key: str) -> bool:
        return self._path(key).exists()


def write_json(path: Path | str, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(data, f, indent=2, default=str)


def read_json(path: Path | str) -> Any:
    with open(path) as f:
        return json.load(f)
