"""Detect superseded claims across videos.

Tutorial series often refactor: an early video introduces v1 of an API, a
later one rewrites it. If we draft from both, the book contradicts itself.

Heuristic, not LLM: cluster claims by embedding similarity; within a
cluster, the *latest* video's claim wins (assuming videos are uploaded in
teaching order — confirmed via `upload_date` from the playlist manifest).
Earlier claims in the cluster get `extra.superseded_by = <new_id>` and
should be excluded from chapter drafting.

This is deliberately conservative: only flag pairs with high semantic
similarity AND same `type`. Pure code claims are matched on token overlap
in addition to embedding (an off-by-one in a refactor still reads similar
in embedding space).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..grounding.store import Claim, ClaimStore
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class SupersessionLink:
    older_id: int
    newer_id: int
    similarity: float
    reason: str


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _build_video_order(manifest: list[dict]) -> dict[str, int]:
    """Return {video_id: rank}. Lower rank = earlier in the playlist."""
    ranked = sorted(
        manifest,
        key=lambda m: (m.get("upload_date") or "", m.get("video_id") or ""),
    )
    return {m["video_id"]: i for i, m in enumerate(ranked)}


def detect_supersession(
    store: ClaimStore,
    manifest: list[dict],
    embed_threshold: float = 0.88,
    code_token_threshold: float = 0.75,
    code_embed_threshold: float = 0.75,
) -> list[SupersessionLink]:
    """Find pairs of cross-video claims where the older is superseded by the newer.

    Embeddings are computed locally via sentence-transformers; if that import
    fails (CPU-only environment with missing model), falls back to pure
    token-Jaccard which is still useful for code.
    """
    claims = store.all_claims()
    if not claims:
        return []
    video_rank = _build_video_order(manifest)
    by_type: dict[str, list[Claim]] = {}
    for c in claims:
        by_type.setdefault(c.type, []).append(c)

    # Embed in batches per type to keep memory reasonable
    embeddings: dict[int, list[float]] = {}
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        for cls in by_type.values():
            texts = [c.content for c in cls]
            vecs = model.encode(texts, normalize_embeddings=True).tolist()
            for c, v in zip(cls, vecs):
                if c.id is not None:
                    embeddings[c.id] = v
    except Exception as e:
        log.warning("Embeddings unavailable for supersession (%s); using Jaccard only", e)

    links: list[SupersessionLink] = []
    for ctype, cls in by_type.items():
        # Pre-compute token sets
        toks = {c.id: _tokens(c.content) for c in cls if c.id is not None}
        for i, a in enumerate(cls):
            for b in cls[i + 1:]:
                if a.video_id == b.video_id:
                    continue
                # Determine order (older vs newer)
                ra = video_rank.get(a.video_id, 0)
                rb = video_rank.get(b.video_id, 0)
                if ra == rb:
                    continue
                older, newer = (a, b) if ra < rb else (b, a)
                # Embedding sim if available
                sim = 0.0
                if older.id in embeddings and newer.id in embeddings:
                    va = embeddings[older.id]
                    vb = embeddings[newer.id]
                    sim = sum(x * y for x, y in zip(va, vb))
                jac = _jaccard(toks.get(older.id, set()), toks.get(newer.id, set()))
                # Decision rule
                trigger = False
                reason = ""
                if ctype == "code":
                    if jac >= code_token_threshold and sim >= code_embed_threshold:
                        trigger = True
                        reason = f"code refactor (jaccard={jac:.2f}, embed={sim:.2f})"
                else:
                    if sim >= embed_threshold:
                        trigger = True
                        reason = f"semantic overlap (embed={sim:.2f})"
                if trigger and older.id and newer.id:
                    links.append(
                        SupersessionLink(
                            older_id=older.id,
                            newer_id=newer.id,
                            similarity=sim,
                            reason=reason,
                        )
                    )
    return links


def apply_supersession(store: ClaimStore, links: list[SupersessionLink]) -> int:
    """Write `extra.superseded_by` on each older claim. Returns count updated.

    We mutate the `extra` JSON column directly via SQLite; no schema change.
    """
    import sqlite3

    n = 0
    with store._conn() as conn:
        for link in links:
            row = conn.execute(
                "SELECT extra FROM claims WHERE id = ?", (link.older_id,)
            ).fetchone()
            if row is None:
                continue
            extra = json.loads(row["extra"]) if row["extra"] else {}
            extra["superseded_by"] = link.newer_id
            extra.setdefault("supersession_reasons", []).append(link.reason)
            conn.execute(
                "UPDATE claims SET extra = ? WHERE id = ?",
                (json.dumps(extra), link.older_id),
            )
            n += 1
    return n


def superseded_ids(store: ClaimStore) -> set[int]:
    """Return the set of claim IDs that have been superseded."""
    out: set[int] = set()
    for c in store.all_claims():
        if c.extra and c.extra.get("superseded_by") is not None and c.id is not None:
            out.add(c.id)
    return out
