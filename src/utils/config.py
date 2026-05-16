"""Config loading. Reads pipeline.yaml, playlist.yaml, and environment."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

@dataclass
class Config:
    """Frozen view of all pipeline configuration."""

    pipeline: dict[str, Any]
    playlist: dict[str, Any]
    env: dict[str, str]
    root: Path

    # Convenience accessors
    def path(self, key: str) -> Path:
        """Resolve a path from `paths` section, rooted at repo root."""
        rel = self.pipeline["paths"][key]
        return (self.root / rel).resolve()

    def model(self, key: str) -> str:
        return self.pipeline["models"][key]

    def threshold(self, key: str) -> float:
        return float(self.pipeline["thresholds"][key])

    def price(self, model_id: str, kind: str) -> float:
        """USD per 1M tokens for a given (model, kind). kind ∈ input|output|cache_write|cache_read."""
        return float(self.pipeline["pricing"][model_id][kind])

    def budget(self, key: str) -> float:
        return float(self.pipeline["budget_usd"][key])

    def gen(self, key: str) -> Any:
        return self.pipeline["generation"][key]

    @property
    def pilot(self) -> bool:
        return self.env.get("PILOT", "0") == "1"


def _find_root(start: Path | None = None) -> Path:
    """Walk up from start (or CWD) until we find pyproject.toml."""
    p = (start or Path.cwd()).resolve()
    for candidate in [p, *p.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Could not find repo root (no pyproject.toml in any parent)")


def load_config(root: Path | None = None) -> Config:
    """Load the full config bundle."""
    root = root or _find_root()
    pipeline_path = root / "config" / "pipeline.yaml"
    playlist_path = root / "config" / "playlist.yaml"

    with pipeline_path.open() as f:
        pipeline = yaml.safe_load(f)
    with playlist_path.open() as f:
        playlist = yaml.safe_load(f)

    # Load .env file automatically
    load_dotenv(root / ".env")

    # Pull a few known env vars; others remain in os.environ
    env = {
        "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", ""),
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
        "LOG_LEVEL": os.environ.get("LOG_LEVEL", "INFO"),
        "PILOT": os.environ.get("PILOT", "0"),
        "BUDGET_USD_TOTAL": os.environ.get(
            "BUDGET_USD_TOTAL", str(pipeline["budget_usd"]["total_max"])
        ),
        "BUDGET_USD_PER_CHAPTER": os.environ.get(
            "BUDGET_USD_PER_CHAPTER", str(pipeline["budget_usd"]["per_chapter_max"])
        ),
    }

    # Override budget from env if set
    pipeline["budget_usd"]["total_max"] = float(env["BUDGET_USD_TOTAL"])
    pipeline["budget_usd"]["per_chapter_max"] = float(env["BUDGET_USD_PER_CHAPTER"])

    return Config(pipeline=pipeline, playlist=playlist, env=env, root=root)


def ensure_dirs(cfg: Config) -> None:
    """Create all output directories declared in `paths`."""
    for key in cfg.pipeline["paths"]:
        path = cfg.path(key)
        # Only mkdir for directory-like paths (no file extension or known dirs)
        if "." not in path.name or path.name in {"figures_generated", "figures_embedded"}:
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
    (cfg.path("data") / ".stage").mkdir(parents=True, exist_ok=True)
