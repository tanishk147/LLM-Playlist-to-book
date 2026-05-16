"""Stage 08 - outline.

Asks Opus to design the book's TOC from the full claim store. Writes
data/outline.json. Also persists chapter assignments back to the claim store.
"""
from __future__ import annotations

import json
import re

from ..grounding import ClaimStore
from ..grounding.supersession import (
    apply_supersession,
    detect_supersession,
    superseded_ids,
)
from ..llm import BudgetTracker, LLMClient
from ..llm.client import Message, Block, CacheableBlock
from ..llm.prompts import load_prompt
from ..utils.cache import read_json, write_json
from ..utils.config import Config
from ..utils.logging import get_logger

log = get_logger(__name__)


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)
    return text.strip()


def _video_metadata_lines(manifest: list[dict]) -> list[str]:
    lines = []
    for entry in manifest:
        d = entry.get("duration_sec") or 0
        lines.append(
            f"- {entry['video_id']}: \"{entry.get('title', '')}\" "
            f"({d/60:.0f} min)"
        )
    return lines


def _claim_summary_lines(store: ClaimStore, skip_ids: set[int] | None = None) -> list[str]:
    """One line per claim. ID first, then type, then short content. Skips
    any IDs in `skip_ids` (e.g. superseded claims)."""
    skip = skip_ids or set()
    out = []
    for c in store.all_claims():
        if c.id in skip:
            continue
        content = c.content.replace("\n", " ").strip()
        if len(content) > 200:
            content = content[:197] + "..."
        out.append(
            f"c{c.id} [{c.type}|{c.video_id}|{c.ts_start:.0f}s]: {content}"
        )
    return out


def run(cfg: Config) -> None:
    store = ClaimStore(cfg.path("claims_db"))
    manifest = read_json(cfg.path("raw") / "manifest.json")

    # Supersession: tag older claims that a later video has refactored.
    # We run this before TOC design so the LLM never sees stale code/APIs.
    log.info("Detecting cross-video supersession")
    links = detect_supersession(store, manifest)
    if links:
        applied = apply_supersession(store, links)
        log.warning(
            "Marked %d claims as superseded (e.g. c%d -> c%d)",
            applied, links[0].older_id, links[0].newer_id,
        )
    skip = superseded_ids(store)

    video_lines = _video_metadata_lines(manifest)
    claim_lines = _claim_summary_lines(store, skip_ids=skip)

    log.info(
        "Outline input: %d videos, %d claims (%d superseded skipped)",
        len(video_lines), len(claim_lines), len(skip),
    )

    budget = BudgetTracker(
        log_path=cfg.path("reports") / "cost_log.jsonl",
        total_max_usd=cfg.budget("total_max"),
        per_chapter_max_usd=cfg.budget("per_chapter_max"),
        abort_on_exceed=cfg.pipeline["budget_usd"]["abort_on_exceed"],
    )
    client = LLMClient(cfg, budget)

    prompt = load_prompt("outline")

    user_blocks = [
        Block(type="text", text="## Videos\n" + "\n".join(video_lines)),
        Block(type="text", text="\n## Claims\n" + "\n".join(claim_lines)),
        Block(type="text", text=(
            f"\n## Book metadata\nTitle: {cfg.playlist['book']['title']}\n"
            f"Subtitle: {cfg.playlist['book']['subtitle']}\n\n"
            "Produce the TOC JSON now."
        )),
    ]

    resp = client.call(
        stage="outline",
        model=cfg.model("outline"),
        system=prompt,
        messages=[Message(role="user", blocks=user_blocks)],
        max_tokens=64000,
        temperature=cfg.gen("outline_temperature"),
    )
    raw = _strip_json_fence(resp.text)
    try:
        outline = json.loads(raw)
    except json.JSONDecodeError as e:
        log.error("Outline JSON parse failed: %s", e)
        log.error("Raw output (first 2000 chars):\n%s", raw[:2000])
        raise

    # Persist chapter assignments, filtering any superseded IDs the LLM may have
    # cited despite the prompt telling it not to.
    for ch in (outline.get("chapters") or []) + (outline.get("appendices") or []):
        cleaned = [cid for cid in ch.get("claim_ids", []) if cid not in skip]
        if len(cleaned) != len(ch.get("claim_ids", [])):
            log.warning(
                "Chapter %s: dropped %d superseded claim refs from outline",
                ch["id"], len(ch.get("claim_ids", [])) - len(cleaned),
            )
        store.assign_claims(ch["id"], cleaned)

    write_json(cfg.path("outline"), outline)
    log.info("Outline saved with %d chapters + %d appendices",
             len(outline.get("chapters", [])),
             len(outline.get("appendices", [])))
