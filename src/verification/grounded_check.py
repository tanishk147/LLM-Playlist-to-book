"""Deterministic fallback grounding checks.

The primary verification is done by a fresh LLM call (see stages/10_verify.py).
This module provides a non-LLM fallback used when the verifier returns
malformed JSON or fails repeatedly. It's intentionally strict: anything that
looks factual but has no citation is flagged unsupported.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..grounding.citation import Sentence, split_sentences


@dataclass
class VerificationResult:
    sentences: list[dict]
    summary: dict[str, int]


def sentence_grounded(s: Sentence, available_claim_ids: Iterable[int]) -> dict:
    """Apply a deterministic grounding rule to a single sentence.

    Returns a dict matching the verify-prompt schema.
    """
    available = set(available_claim_ids)
    if s.type == "narrative":
        return {
            "index": s.index,
            "text": s.text,
            "type": s.type,
            "cited_claim_ids": [],
            "status": "narrative_ok",
            "reason": "no factual content",
        }
    if s.type == "figure_directive":
        return {
            "index": s.index,
            "text": s.text,
            "type": s.type,
            "cited_claim_ids": [],
            "status": "narrative_ok",
            "reason": "figure directive",
        }
    if not s.cited_claim_ids:
        return {
            "index": s.index,
            "text": s.text,
            "type": s.type,
            "cited_claim_ids": [],
            "status": "unsupported",
            "reason": "no citation",
        }
    bad = [c for c in s.cited_claim_ids if c not in available]
    if bad:
        return {
            "index": s.index,
            "text": s.text,
            "type": s.type,
            "cited_claim_ids": [f"c{c}" for c in s.cited_claim_ids],
            "status": "unsupported",
            "reason": f"cited claims not in chapter: {bad}",
        }
    return {
        "index": s.index,
        "text": s.text,
        "type": s.type,
        "cited_claim_ids": [f"c{c}" for c in s.cited_claim_ids],
        "status": "grounded",
        "reason": "citations match available claims",
    }


def fallback_verify(chapter_md: str, available_claim_ids: Iterable[int]) -> VerificationResult:
    """Run the deterministic grounding rule over a whole chapter."""
    sentences = split_sentences(chapter_md)
    available = list(available_claim_ids)
    per_sentence = [sentence_grounded(s, available) for s in sentences]
    summary = {
        "grounded": 0,
        "unsupported": 0,
        "contradicted": 0,
        "needs_external_check": 0,
        "narrative_ok": 0,
    }
    for ps in per_sentence:
        summary[ps["status"]] = summary.get(ps["status"], 0) + 1
    return VerificationResult(sentences=per_sentence, summary=summary)
