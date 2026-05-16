"""Hybrid retrieval for chapter drafting.

Today chapter drafting dumps every assigned claim into the prompt. For long
chapters this burns input tokens and risks hitting context limits. Hybrid
retrieval (BM25 token overlap + dense embedding cosine) scores each assigned
claim against the chapter title + summary, then returns the top-N. Always
keeps code-type and equation-type claims (they're load-bearing and short).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .store import Claim
from ..utils.logging import get_logger

log = get_logger(__name__)


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class _BM25:
    doc_freq: Counter
    doc_lens: list[int]
    avgdl: float
    k1: float = 1.5
    b: float = 0.75
    N: int = 0


def _build_bm25(docs: list[list[str]]) -> _BM25:
    doc_freq: Counter = Counter()
    for d in docs:
        for term in set(d):
            doc_freq[term] += 1
    doc_lens = [len(d) for d in docs]
    avgdl = (sum(doc_lens) / len(doc_lens)) if doc_lens else 0.0
    return _BM25(doc_freq=doc_freq, doc_lens=doc_lens, avgdl=avgdl, N=len(docs))


def _bm25_score(idx: int, doc_tokens: list[str], query: list[str], bm: _BM25) -> float:
    if bm.N == 0 or not query:
        return 0.0
    tf = Counter(doc_tokens)
    dl = bm.doc_lens[idx] or 1
    score = 0.0
    for q in query:
        f = tf.get(q, 0)
        if f == 0:
            continue
        df = bm.doc_freq.get(q, 0)
        idf = math.log((bm.N - df + 0.5) / (df + 0.5) + 1.0)
        num = f * (bm.k1 + 1)
        den = f + bm.k1 * (1 - bm.b + bm.b * dl / (bm.avgdl or 1))
        score += idf * (num / den)
    return score


def _dense_scores(query: str, docs: list[str]) -> list[float] | None:
    """Return cosine similarities or None if embeddings unavailable."""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        all_vecs = model.encode([query, *docs], normalize_embeddings=True)
        q_vec = all_vecs[0]
        return [float(sum(x * y for x, y in zip(q_vec, v))) for v in all_vecs[1:]]
    except Exception as e:  # pragma: no cover (depends on env)
        log.warning("Dense retrieval unavailable (%s); using BM25 only", e)
        return None


def select_claims(
    claims: list[Claim],
    chapter_title: str,
    chapter_summary: str,
    max_claims: int,
    keep_types: tuple[str, ...] = ("code", "equation", "definition"),
    dense_weight: float = 0.5,
) -> list[Claim]:
    """Score and return the top-N most relevant claims.

    Always keeps short load-bearing types (code, equations, definitions) even
    if their text-similarity is low — they're often what a chapter needs and
    are cheap in tokens.
    """
    if not claims:
        return []
    if len(claims) <= max_claims:
        return claims

    must_keep: list[Claim] = [c for c in claims if c.type in keep_types]
    candidates: list[Claim] = [c for c in claims if c.type not in keep_types]
    budget = max_claims - len(must_keep)
    if budget <= 0:
        # Trim must_keep itself by recency (newest first) if it's already too big
        must_keep.sort(key=lambda c: c.id or 0, reverse=True)
        return must_keep[:max_claims]

    query = f"{chapter_title}. {chapter_summary}"
    query_tokens = _tokenize(query)
    doc_token_lists = [_tokenize(c.content) for c in candidates]
    bm = _build_bm25(doc_token_lists)
    bm25_raw = [_bm25_score(i, d, query_tokens, bm) for i, d in enumerate(doc_token_lists)]

    # Normalize BM25 to [0,1] for blending
    bm25_max = max(bm25_raw) if bm25_raw else 0.0
    bm25_norm = [s / bm25_max if bm25_max > 0 else 0.0 for s in bm25_raw]

    dense = _dense_scores(query, [c.content for c in candidates])
    if dense is None:
        blended = bm25_norm
    else:
        blended = [
            (1 - dense_weight) * b + dense_weight * d
            for b, d in zip(bm25_norm, dense)
        ]

    ranked_idx = sorted(range(len(candidates)), key=lambda i: blended[i], reverse=True)
    selected = [candidates[i] for i in ranked_idx[:budget]]
    # Stable order in output: must-keep first (by id), then selected (by score)
    must_keep.sort(key=lambda c: c.id or 0)
    return must_keep + selected
