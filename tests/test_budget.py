"""Tests for the budget tracker."""
import time

import pytest

from src.llm.budget import BudgetExceeded, BudgetTracker


def _mk(tmp_path, total=100.0, per_chap=10.0, abort=True):
    return BudgetTracker(
        log_path=tmp_path / "cost.jsonl",
        total_max_usd=total,
        per_chapter_max_usd=per_chap,
        abort_on_exceed=abort,
    )


def test_record_accumulates(tmp_path):
    b = _mk(tmp_path)
    b.record("ingest", "claude-opus-4-7", 100, 50, 0, 0, 1.50, time.time())
    b.record("ingest", "claude-opus-4-7", 100, 50, 0, 0, 0.75, time.time())
    assert b.total_usd == pytest.approx(2.25)


def test_check_raises_on_total_exceed(tmp_path):
    b = _mk(tmp_path, total=5.0)
    b.record("ingest", "m", 1, 1, 0, 0, 4.0, time.time())
    with pytest.raises(BudgetExceeded):
        b.check(predicted_usd=2.0, stage="ingest")


def test_check_per_chapter(tmp_path):
    b = _mk(tmp_path, total=100.0, per_chap=2.0)
    b.record("chapter:intro", "m", 1, 1, 0, 0, 1.5, time.time())
    with pytest.raises(BudgetExceeded):
        b.check(predicted_usd=1.0, stage="chapter:intro")


def test_check_passes_under_limit(tmp_path):
    b = _mk(tmp_path)
    b.record("ingest", "m", 1, 1, 0, 0, 5.0, time.time())
    b.check(predicted_usd=1.0, stage="ingest")  # should not raise


def test_persistence(tmp_path):
    b1 = _mk(tmp_path)
    b1.record("ingest", "m", 1, 1, 0, 0, 1.0, time.time())
    b2 = _mk(tmp_path)
    assert b2.total_usd == pytest.approx(1.0)
