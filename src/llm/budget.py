"""USD budget tracking. Aborts pipeline before exceeding configured ceiling."""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

from ..utils.logging import get_logger

log = get_logger(__name__)


class BudgetExceeded(RuntimeError):
    """Raised when a planned LLM call would push spend past the configured ceiling."""


@dataclass
class _Entry:
    stage: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    usd: float
    timestamp: float


@dataclass
class BudgetTracker:
    """Thread-safe USD tracker. Persists to a JSONL log."""

    log_path: Path
    total_max_usd: float
    per_chapter_max_usd: float
    abort_on_exceed: bool = True
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _entries: list[_Entry] = field(default_factory=list, repr=False)
    _per_chapter: dict[str, float] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.log_path.exists():
            self._load()

    def _load(self) -> None:
        with self.log_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                self._entries.append(_Entry(**d))
                if d["stage"].startswith("chapter:"):
                    chap = d["stage"].split(":", 1)[1]
                    self._per_chapter[chap] = self._per_chapter.get(chap, 0.0) + d["usd"]

    @property
    def total_usd(self) -> float:
        return sum(e.usd for e in self._entries)

    def chapter_usd(self, chapter_id: str) -> float:
        return self._per_chapter.get(chapter_id, 0.0)

    def check(self, predicted_usd: float, stage: str) -> None:
        """Raise BudgetExceeded if this call would push us over a limit."""
        proj_total = self.total_usd + predicted_usd
        if proj_total > self.total_max_usd:
            msg = (
                f"Projected total ${proj_total:.2f} exceeds ceiling "
                f"${self.total_max_usd:.2f} (stage={stage})"
            )
            if self.abort_on_exceed:
                raise BudgetExceeded(msg)
            log.warning(msg)
        if stage.startswith("chapter:"):
            chap = stage.split(":", 1)[1]
            proj_chap = self.chapter_usd(chap) + predicted_usd
            if proj_chap > self.per_chapter_max_usd:
                msg = (
                    f"Projected chapter '{chap}' cost ${proj_chap:.2f} exceeds "
                    f"per-chapter ceiling ${self.per_chapter_max_usd:.2f}"
                )
                if self.abort_on_exceed:
                    raise BudgetExceeded(msg)
                log.warning(msg)

    def record(
        self,
        stage: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_write_tokens: int,
        cache_read_tokens: int,
        usd: float,
        timestamp: float,
    ) -> None:
        with self._lock:
            entry = _Entry(
                stage=stage,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_write_tokens=cache_write_tokens,
                cache_read_tokens=cache_read_tokens,
                usd=usd,
                timestamp=timestamp,
            )
            self._entries.append(entry)
            if stage.startswith("chapter:"):
                chap = stage.split(":", 1)[1]
                self._per_chapter[chap] = self._per_chapter.get(chap, 0.0) + usd
            with self.log_path.open("a") as f:
                f.write(json.dumps(entry.__dict__) + "\n")

    def entries(self) -> list[_Entry]:
        return list(self._entries)
