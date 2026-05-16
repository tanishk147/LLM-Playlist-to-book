"""Tests for hybrid retrieval (BM25 path only; dense optional)."""
from src.grounding.retrieval import select_claims
from src.grounding.store import Claim


def _mk(content: str, ctype: str = "fact", cid: int = 0) -> Claim:
    c = Claim(
        type=ctype,
        content=content,
        video_id="v1",
        ts_start=0.0,
        ts_end=1.0,
        frame_ids=[],
        confidence=1.0,
        source_hash="x",
    )
    c.id = cid
    return c


def test_returns_all_when_under_cap():
    claims = [_mk("anything", cid=i) for i in range(3)]
    out = select_claims(claims, "Title", "Summary", max_claims=10)
    assert len(out) == 3


def test_keeps_code_claims():
    claims = [
        _mk("transformer architecture explanation", cid=1),
        _mk("def attn(): pass", ctype="code", cid=2),
        _mk("random unrelated content here xyz", cid=3),
    ]
    out = select_claims(claims, "Attention mechanism", "How attention works", max_claims=2)
    ids = {c.id for c in out}
    assert 2 in ids  # code claim always retained


def test_bm25_picks_relevant():
    claims = [
        _mk("attention mechanism scaled dot product", cid=1),
        _mk("baking bread sourdough loaves", cid=2),
        _mk("query key value matrix multiplication", cid=3),
        _mk("garden gnome statues", cid=4),
    ]
    out = select_claims(
        claims,
        "Attention",
        "scaled dot product attention with query key value",
        max_claims=2,
    )
    selected_ids = {c.id for c in out}
    # The two relevant claims (1, 3) should outrank bread and gnomes
    assert selected_ids == {1, 3}
