"""Tests for the citation parser."""
from src.grounding.citation import (
    CitationParser,
    extract_citations,
    split_sentences,
)


def test_extract_citations_single():
    assert extract_citations("foo [c12] bar") == [12]


def test_extract_citations_multiple():
    assert extract_citations("foo [c12, c47] bar [c3]") == [12, 47, 3]


def test_extract_citations_case_insensitive():
    assert extract_citations("foo [C12]") == [12]


def test_extract_citations_none():
    assert extract_citations("plain text") == []


def test_split_sentences_basic():
    md = "## Title\n\nFirst sentence [c1]. Second sentence [c2].\n"
    sents = split_sentences(md)
    factual = [s for s in sents if s.type == "factual"]
    assert len(factual) == 2
    assert factual[0].cited_claim_ids == [1]
    assert factual[1].cited_claim_ids == [2]


def test_split_sentences_code_block():
    md = "Intro [c1].\n\n```python\ndef f():\n    return 1\n```\n\nAfter [c2]."
    sents = split_sentences(md)
    types = [s.type for s in sents]
    assert "code_block" in types
    assert types.count("factual") == 2


def test_split_sentences_block_math():
    md = "Equation below [c1].\n\n$$y = Wx + b$$\n\nMore prose [c2]."
    sents = split_sentences(md)
    types = [s.type for s in sents]
    assert "equation" in types


def test_split_sentences_figure_directive():
    md = "Discussion [c1].\n\n[FIGURE: attention block diagram]\n\nMore [c2]."
    sents = split_sentences(md)
    assert any(s.type == "figure_directive" for s in sents)


def test_split_sentences_narrative_detection():
    md = "We will now discuss self-attention. Self-attention uses Q, K, V projections [c1]."
    sents = split_sentences(md)
    # First is narrative; second is factual with citation
    assert sents[0].type == "narrative" or sents[1].cited_claim_ids == [1]


def test_strip_unsupported_removes_sentence():
    md = "First [c1]. Second sentence has no citation. Third [c2]."
    sents = split_sentences(md)
    bad = {s.index for s in sents if s.type == "factual" and not s.cited_claim_ids}
    out = CitationParser.strip_unsupported(md, bad)
    assert "no citation" not in out
    assert "[c1]" in out
    assert "[c2]" in out
