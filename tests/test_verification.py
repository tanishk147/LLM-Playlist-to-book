"""Tests for the deterministic grounding fallback."""
from src.grounding.citation import Sentence
from src.verification.grounded_check import fallback_verify, sentence_grounded


def test_grounded_when_citation_matches():
    s = Sentence(index=0, text="X uses Y [c1].", type="factual", cited_claim_ids=[1])
    out = sentence_grounded(s, available_claim_ids=[1, 2])
    assert out["status"] == "grounded"


def test_unsupported_when_no_citation():
    s = Sentence(index=0, text="X uses Y.", type="factual", cited_claim_ids=[])
    out = sentence_grounded(s, available_claim_ids=[1])
    assert out["status"] == "unsupported"


def test_unsupported_when_citation_missing():
    s = Sentence(index=0, text="X uses Y [c99].", type="factual", cited_claim_ids=[99])
    out = sentence_grounded(s, available_claim_ids=[1, 2])
    assert out["status"] == "unsupported"


def test_narrative_ok():
    s = Sentence(index=0, text="We now discuss X.", type="narrative", cited_claim_ids=[])
    out = sentence_grounded(s, available_claim_ids=[])
    assert out["status"] == "narrative_ok"


def test_figure_directive_narrative_ok():
    s = Sentence(index=0, text="[FIGURE: attention]", type="figure_directive", cited_claim_ids=[])
    out = sentence_grounded(s, available_claim_ids=[])
    assert out["status"] == "narrative_ok"


def test_fallback_verify_summary():
    md = (
        "## Intro\n\n"
        "We now discuss self-attention. "
        "Self-attention uses Q projections [c1]. "
        "Some uncited factual claim with numbers like 768."
    )
    result = fallback_verify(md, available_claim_ids=[1])
    assert result.summary["grounded"] >= 1
    assert result.summary["unsupported"] >= 1
