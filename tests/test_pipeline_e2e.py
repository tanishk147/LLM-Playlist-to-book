"""End-to-end mock test for the chapter sub-pipeline.

We don't call any external services. Instead we:
  - populate a ClaimStore with hand-crafted claims
  - assign them to a synthetic chapter
  - swap LLMClient.call with a fake that returns a canned draft, verify, polish
  - exercise the draft -> verify -> strip -> polish flow
  - assert the audit captures the right grounding states

This proves the integration points line up (config keys, prompt loading,
citation regex, claim-ID validation, sentence stripping) without needing
ffmpeg, whisper, or Anthropic credits.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.grounding.store import Claim, ClaimStore
from src.stages import chapters as chapters_stage


@pytest.fixture
def tiny_config(tmp_path: Path):
    data = tmp_path / "data"
    chapters_dir = data / "chapters"
    reports = tmp_path / "reports"
    for p in (data, chapters_dir, reports):
        p.mkdir(parents=True, exist_ok=True)

    outline = {
        "chapters": [
            {
                "id": "intro",
                "number": 1,
                "title": "Tokenization Basics",
                "summary": "BPE merges and vocab sizes",
                "estimated_pages": 2,
            }
        ],
        "appendices": [],
    }
    (data / "outline.json").write_text(json.dumps(outline))

    pipeline = {
        "models": {"draft": "fake-opus", "verify": "fake-opus", "polish": "fake-opus"},
        "generation": {
            "draft_temperature": 0.2,
            "verify_temperature": 0.0,
            "polish_temperature": 0.3,
            "max_output_tokens_chapter": 2000,
        },
        "thresholds": {
            "max_claims_per_chapter": 0,
            "dense_retrieval_weight": 0.0,
        },
        "caching": {"ttl": "1h", "beta_header": "x"},
        "budget_usd": {
            "per_chapter_max": 10.0,
            "total_max": 100.0,
            "abort_on_exceed": False,
        },
        "pricing": {
            "fake-opus": {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0},
        },
        "paths": {
            "data": str(data.relative_to(tmp_path)),
            "raw": "data/raw",
            "outline": "data/outline.json",
            "claims_db": "data/claims.sqlite",
            "chapters": "data/chapters",
            "reports": "reports",
        },
    }

    cfg = SimpleNamespace(
        pipeline=pipeline,
        playlist={"book": {"title": "x", "subtitle": "y", "author": "z"}},
        env={"ANTHROPIC_API_KEY": "fake"},
        root=tmp_path,
    )
    cfg.path = lambda k: (tmp_path / pipeline["paths"][k])
    cfg.model = lambda k: pipeline["models"][k]
    cfg.threshold = lambda k: pipeline["thresholds"][k]
    cfg.budget = lambda k: pipeline["budget_usd"][k]
    cfg.gen = lambda k: pipeline["generation"][k]
    return cfg


class _FakeResp:
    def __init__(self, text: str):
        self.text = text


class _FakeClient:
    """Drop-in for LLMClient.call. Returns canned outputs by stage prefix."""

    def __init__(self, draft_md: str, verify_json: dict, polish_md: str):
        self.draft_md = draft_md
        self.verify_json = verify_json
        self.polish_md = polish_md
        self.calls: list[str] = []

    def call(self, *, stage: str, **_) -> _FakeResp:
        self.calls.append(stage)
        # First call from chapter:* is draft, then verify, then polish
        # Distinguish by call ordinal within the chapter
        n = sum(1 for s in self.calls if s == stage)
        if n == 1:
            return _FakeResp(self.draft_md)
        if n == 2:
            return _FakeResp(json.dumps(self.verify_json))
        return _FakeResp(self.polish_md)


def test_chapter_pipeline_strips_unsupported(tiny_config, monkeypatch):
    cfg = tiny_config
    store = ClaimStore(cfg.path("claims_db"))
    cids = store.add_claims([
        Claim(type="fact", content="BPE merges most-frequent pairs.",
              video_id="v1", ts_start=0.0, ts_end=10.0,
              frame_ids=[], confidence=1.0, source_hash="h1"),
        Claim(type="fact", content="GPT-2 used a vocab of 50257 tokens.",
              video_id="v1", ts_start=20.0, ts_end=25.0,
              frame_ids=[], confidence=1.0, source_hash="h2"),
    ])
    store.assign_claims("intro", cids)

    draft_md = (
        "## Tokenization Basics\n\n"
        f"BPE merges most-frequent pairs. [c{cids[0]}]\n\n"
        f"GPT-2 used a vocab of 50257 tokens. [c{cids[1]}]\n\n"
        "This unsupported sentence has no citation.\n"
    )
    verify = {
        "sentences": [
            {"index": 0, "text": "Tokenization Basics",
             "type": "narrative", "cited_claim_ids": [], "status": "narrative_ok", "reason": ""},
            {"index": 1, "text": f"BPE merges most-frequent pairs. [c{cids[0]}]",
             "type": "factual", "cited_claim_ids": [f"c{cids[0]}"],
             "status": "grounded", "reason": ""},
            {"index": 2, "text": f"GPT-2 used a vocab of 50257 tokens. [c{cids[1]}]",
             "type": "factual", "cited_claim_ids": [f"c{cids[1]}"],
             "status": "grounded", "reason": ""},
            {"index": 3, "text": "This unsupported sentence has no citation.",
             "type": "factual", "cited_claim_ids": [],
             "status": "unsupported", "reason": "no citation"},
        ],
        "summary": {"grounded": 2, "unsupported": 1, "narrative_ok": 1},
    }
    polish_md = draft_md.replace(
        "This unsupported sentence has no citation.\n", ""
    )
    fake = _FakeClient(draft_md, verify, polish_md)

    # Patch LLMClient and BudgetTracker in the chapters module
    monkeypatch.setattr(chapters_stage, "LLMClient", lambda *a, **kw: fake)
    monkeypatch.setattr(chapters_stage, "BudgetTracker", lambda **kw: SimpleNamespace(
        record=lambda **k: None, check=lambda *a, **kw: None,
    ))

    chapters_stage.run(cfg)

    out_dir = cfg.path("chapters") / "01_intro"
    assert (out_dir / "draft.md").exists()
    assert (out_dir / "verify.json").exists()
    assert (out_dir / "verified.md").exists()
    assert (out_dir / "polished.md").exists()
    audit = json.loads((out_dir / "audit.json").read_text())
    assert audit["chapter_id"] == "intro"
    assert audit["stripped_sentence_indices"] == [3]


def test_phantom_claim_id_demoted(tiny_config, monkeypatch):
    cfg = tiny_config
    store = ClaimStore(cfg.path("claims_db"))
    cids = store.add_claims([
        Claim(type="fact", content="Real claim",
              video_id="v1", ts_start=0.0, ts_end=10.0,
              frame_ids=[], confidence=1.0, source_hash="h"),
    ])
    store.assign_claims("intro", cids)

    draft_md = (
        "## Tokenization Basics\n\n"
        f"Real claim sentence. [c{cids[0]}]\n\n"
        "Fake reference here. [c99999]\n"
    )
    verify = {
        "sentences": [
            {"index": 0, "text": "Tokenization Basics",
             "type": "narrative", "cited_claim_ids": [], "status": "narrative_ok", "reason": ""},
            {"index": 1, "text": f"Real claim sentence. [c{cids[0]}]",
             "type": "factual", "cited_claim_ids": [f"c{cids[0]}"],
             "status": "grounded", "reason": ""},
            # Verifier mistakenly accepts a phantom ID:
            {"index": 2, "text": "Fake reference here. [c99999]",
             "type": "factual", "cited_claim_ids": ["c99999"],
             "status": "grounded", "reason": ""},
        ],
        "summary": {"grounded": 2, "narrative_ok": 1},
    }
    fake = _FakeClient(draft_md, verify, draft_md)
    monkeypatch.setattr(chapters_stage, "LLMClient", lambda *a, **kw: fake)
    monkeypatch.setattr(chapters_stage, "BudgetTracker", lambda **kw: SimpleNamespace(
        record=lambda **k: None, check=lambda *a, **kw: None,
    ))

    chapters_stage.run(cfg)

    out_dir = cfg.path("chapters") / "01_intro"
    verify_out = json.loads((out_dir / "verify.json").read_text())
    phantom = next(s for s in verify_out["sentences"] if s["index"] == 2)
    assert phantom["status"] == "unsupported"
    assert "not in chapter" in phantom["reason"]
