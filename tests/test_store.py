"""Tests for the SQLite claim store."""
import tempfile
from pathlib import Path

import pytest

from src.grounding.store import Claim, ClaimStore


@pytest.fixture
def store(tmp_path):
    return ClaimStore(tmp_path / "claims.sqlite")


def _mk(content: str, video_id: str = "vid1", ts: float = 0.0, type_: str = "fact") -> Claim:
    return Claim(
        type=type_,
        content=content,
        video_id=video_id,
        ts_start=ts,
        ts_end=ts + 10.0,
        frame_ids=[],
        confidence=0.9,
        source_hash="abc",
    )


def test_add_and_get(store):
    ids = store.add_claims([_mk("foo"), _mk("bar")])
    assert len(ids) == 2
    c = store.get_claim(ids[0])
    assert c is not None
    assert c.content == "foo"


def test_get_claims_preserves_order(store):
    ids = store.add_claims([_mk("a"), _mk("b"), _mk("c")])
    result = store.get_claims([ids[2], ids[0]])
    assert [c.content for c in result] == ["c", "a"]


def test_claims_for_video(store):
    store.add_claims([
        _mk("v1-a", video_id="vid1"),
        _mk("v2-a", video_id="vid2"),
        _mk("v1-b", video_id="vid1"),
    ])
    v1 = store.claims_for_video("vid1")
    assert {c.content for c in v1} == {"v1-a", "v1-b"}


def test_chapter_assignment(store):
    ids = store.add_claims([_mk("a"), _mk("b"), _mk("c")])
    store.assign_claims("chap_intro", [ids[0], ids[2]])
    cs = store.claims_for_chapter("chap_intro")
    assert {c.content for c in cs} == {"a", "c"}


def test_record_citations_and_audit(store):
    ids = store.add_claims([_mk("a")])
    sentences = [
        {"index": 0, "text": "S0", "type": "factual",
         "cited_claim_ids": [f"c{ids[0]}"], "status": "grounded", "reason": ""},
        {"index": 1, "text": "S1", "type": "factual",
         "cited_claim_ids": [], "status": "unsupported", "reason": "no cite"},
        {"index": 2, "text": "S2", "type": "narrative",
         "cited_claim_ids": [], "status": "narrative_ok", "reason": ""},
    ]
    store.record_citations("chap_x", sentences)
    summary = store.audit_summary()
    assert summary["grounded"] == 1
    assert summary["unsupported"] == 1
    assert summary["narrative_ok"] == 1


def test_audit_for_chapter(store):
    ids = store.add_claims([_mk("a")])
    store.record_citations("chap_x", [
        {"index": 0, "text": "S0", "type": "factual",
         "cited_claim_ids": [f"c{ids[0]}"], "status": "grounded", "reason": ""},
    ])
    rows = store.audit_for_chapter("chap_x")
    assert len(rows) == 1
    assert rows[0]["status"] == "grounded"
