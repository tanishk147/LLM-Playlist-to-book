"""Stage 09 - chapters.

For each chapter in the outline, run:
  draft → verify → strip unsupported → polish

inside a single Anthropic prompt-cache session (so the source bundle is
cached once and reused across the three Opus calls).

Output per chapter (under data/chapters/<NN_slug>/):
  sources.jsonl     - frozen claim list used as input
  draft.md          - initial draft with inline citations
  verify.json       - per-sentence verifier output
  verified.md       - draft with unsupported sentences stripped
  polished.md       - final, copy-edited version
  audit.json        - canonical record of all verification decisions
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..grounding import ClaimStore
from ..grounding.citation import CitationParser, split_sentences
from ..grounding.retrieval import select_claims
from ..grounding.store import _parse_claim_ref
from ..llm import BudgetTracker, LLMClient
from ..llm.client import (
    Block,
    CacheableBlock,
    Message,
    cached_system,
)
from ..llm.prompts import load_prompt
from ..utils.cache import read_json, write_json
from ..utils.config import Config
from ..utils.logging import get_logger
from ..verification.grounded_check import fallback_verify

log = get_logger(__name__)


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)
    return text.strip()


def _strip_md_fence(text: str) -> str:
    """For drafts where the model wrapped the entire markdown in a fence."""
    text = text.strip()
    if text.startswith("```markdown") or text.startswith("```md"):
        text = re.sub(r"^```(?:markdown|md)\s*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)
    return text.strip()


def _claims_payload(
    store: ClaimStore,
    chapter: dict,
    max_claims: int = 0,
    dense_weight: float = 0.5,
) -> str:
    """One line per claim, in a format the draft/verify prompts understand.

    When `max_claims > 0`, hybrid retrieval picks the most relevant subset
    using BM25 + dense embeddings against the chapter title + summary.
    """
    chapter_id = chapter["id"]
    claims = store.claims_for_chapter(chapter_id)
    if not claims:
        return "## Claims\n(none assigned)"
    if max_claims and len(claims) > max_claims:
        before = len(claims)
        claims = select_claims(
            claims,
            chapter_title=chapter.get("title", ""),
            chapter_summary=chapter.get("summary", ""),
            max_claims=max_claims,
            dense_weight=dense_weight,
        )
        log.info(
            "Chapter %s: retrieval narrowed claims %d -> %d",
            chapter_id, before, len(claims),
        )
    lines = ["## Claims (cite these by ID inline as [cNNN])"]
    for c in claims:
        body = c.content.replace("\n", "\n  ")
        lines.append(
            f"- c{c.id} [{c.type} | {c.video_id} @ {c.ts_start:.0f}-{c.ts_end:.0f}s]\n  {body}"
        )
    return "\n".join(lines)


def _draft_chapter(
    client: LLMClient,
    cfg: Config,
    chapter: dict,
    store: ClaimStore,
) -> tuple[str, str]:
    """Returns (draft_md, claims_payload). Uses cached system."""
    sys_prompt = load_prompt("chapter_draft")
    claims_payload = _claims_payload(
        store,
        chapter,
        max_claims=int(cfg.threshold("max_claims_per_chapter") or 0),
        dense_weight=float(cfg.threshold("dense_retrieval_weight")),
    )

    system = cached_system(
        sys_prompt,                  # cached: prompt template
        (claims_payload, True),      # cached: claims for this chapter
    )

    user_text = (
        f"Chapter info:\n"
        f"  number: {chapter.get('number')}\n"
        f"  title: {chapter['title']}\n"
        f"  summary: {chapter.get('summary', '')}\n"
        f"  estimated_pages: {chapter.get('estimated_pages', 10)}\n\n"
        f"Now write the chapter as instructed."
    )

    resp = client.call(
        stage=f"chapter:{chapter['id']}",
        model=cfg.model("draft"),
        system=system,
        messages=[Message(role="user", blocks=[Block(type="text", text=user_text)])],
        max_tokens=cfg.gen("max_output_tokens_chapter"),
        temperature=cfg.gen("draft_temperature"),
        use_extended_cache=True,
    )
    return _strip_md_fence(resp.text), claims_payload


def _verify_chapter(
    client: LLMClient,
    cfg: Config,
    chapter: dict,
    draft_md: str,
    store: ClaimStore,
) -> dict:
    """Returns parsed verify-output dict. Fresh context: no draft history."""
    sys_prompt = load_prompt("chapter_verify")
    claims_payload = _claims_payload(
        store,
        chapter,
        max_claims=int(cfg.threshold("max_claims_per_chapter") or 0),
        dense_weight=float(cfg.threshold("dense_retrieval_weight")),
    )

    # Build inputs: claims payload + draft, in that order; cache the claims payload
    system = cached_system(
        sys_prompt,
        (claims_payload, True),
    )

    user_text = (
        f"## Chapter draft to verify\n\n{draft_md}\n\n"
        "Classify every sentence per the schema. JSON only."
    )

    resp = client.call(
        stage=f"chapter:{chapter['id']}",
        model=cfg.model("verify"),
        system=system,
        messages=[Message(role="user", blocks=[Block(type="text", text=user_text)])],
        max_tokens=cfg.gen("max_output_tokens_chapter"),
        temperature=cfg.gen("verify_temperature"),
        use_extended_cache=True,
    )
    raw = _strip_json_fence(resp.text)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        log.warning(
            "Verifier returned malformed JSON for %s; using deterministic fallback",
            chapter["id"],
        )
        available_ids = [c.id for c in store.claims_for_chapter(chapter["id"])]
        fallback = fallback_verify(draft_md, available_ids)
        parsed = {"sentences": fallback.sentences, "summary": fallback.summary}
    return parsed


def _validate_cited_ids(verify: dict, store: ClaimStore, chapter_id: str) -> int:
    """Post-hoc: any sentence whose cited_claim_ids reference unknown or
    out-of-chapter IDs gets demoted to 'unsupported'. Returns count demoted.

    The verifier prompt is *supposed* to catch this, but it's an LLM — we
    enforce in code so a hallucinated [c999] cannot pass through.
    """
    chap_claim_ids = {c.id for c in store.claims_for_chapter(chapter_id) if c.id is not None}
    demoted = 0
    for s in verify.get("sentences", []):
        cited_raw = s.get("cited_claim_ids") or []
        if not cited_raw:
            continue
        parsed = [_parse_claim_ref(c) for c in cited_raw]
        parsed_set = {p for p in parsed if p is not None}
        if not parsed_set:
            # all garbage refs
            if s.get("status") == "grounded":
                s["status"] = "unsupported"
                s["reason"] = (s.get("reason", "") + " | cited refs unparseable").strip(" |")
                demoted += 1
            continue
        unknown = parsed_set - chap_claim_ids
        if unknown and s.get("status") == "grounded":
            s["status"] = "unsupported"
            s["reason"] = (
                s.get("reason", "")
                + f" | cited claim ids not in chapter: {sorted(unknown)}"
            ).strip(" |")
            demoted += 1
    return demoted


def _strip_unsupported(draft_md: str, verify: dict) -> tuple[str, list[int]]:
    """Remove unsupported sentences from draft. Returns (verified_md, removed_idxs)."""
    bad_idxs = {
        s["index"]
        for s in verify.get("sentences", [])
        if s.get("status") == "unsupported"
    }
    if not bad_idxs:
        return draft_md, []
    verified = CitationParser.strip_unsupported(draft_md, bad_idxs)
    # Tidy up double spaces / blank lines
    verified = re.sub(r"\n{3,}", "\n\n", verified)
    verified = re.sub(r"[ \t]{2,}", " ", verified)
    return verified.strip() + "\n", sorted(bad_idxs)


def _polish_chapter(
    client: LLMClient,
    cfg: Config,
    chapter: dict,
    verified_md: str,
    store: ClaimStore,
) -> str:
    sys_prompt = load_prompt("chapter_polish")
    claims_payload = _claims_payload(
        store,
        chapter,
        max_claims=int(cfg.threshold("max_claims_per_chapter") or 0),
        dense_weight=float(cfg.threshold("dense_retrieval_weight")),
    )

    system = cached_system(
        sys_prompt,
        (claims_payload, True),
    )

    user_text = (
        f"## Chapter to polish\n\n{verified_md}\n\n"
        "Apply the polish pass per the rules. Output only the polished markdown."
    )

    resp = client.call(
        stage=f"chapter:{chapter['id']}",
        model=cfg.model("polish"),
        system=system,
        messages=[Message(role="user", blocks=[Block(type="text", text=user_text)])],
        max_tokens=cfg.gen("max_output_tokens_chapter"),
        temperature=cfg.gen("polish_temperature"),
        use_extended_cache=True,
    )
    return _strip_md_fence(resp.text)


def _process_chapter(
    client: LLMClient,
    cfg: Config,
    chapter: dict,
    store: ClaimStore,
) -> None:
    chap_id = chapter["id"]
    out_dir = cfg.path("chapters") / f"{chapter['number']:02d}_{chap_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    polished_path = out_dir / "polished.md"
    if polished_path.exists():
        log.info("Skip chapter %s: polished output already exists", chap_id)
        return

    # 1. Draft
    draft_path = out_dir / "draft.md"
    if draft_path.exists():
        log.info("Chapter %s: reusing existing draft", chap_id)
        draft_md = draft_path.read_text()
    else:
        log.info("Chapter %s: drafting", chap_id)
        draft_md, claims_payload = _draft_chapter(client, cfg, chapter, store)
        draft_path.write_text(draft_md)
        # snapshot the claims that went in (frozen evidence for audit)
        (out_dir / "sources.jsonl").write_text(claims_payload)

    # 2. Verify
    verify_path = out_dir / "verify.json"
    if verify_path.exists():
        log.info("Chapter %s: reusing existing verification", chap_id)
        verify = read_json(verify_path)
    else:
        log.info("Chapter %s: verifying", chap_id)
        verify = _verify_chapter(client, cfg, chapter, draft_md, store)
        # Post-hoc ID-existence check (defense in depth against verifier mistakes)
        demoted = _validate_cited_ids(verify, store, chap_id)
        if demoted:
            log.warning("Chapter %s: demoted %d sentences with phantom claim IDs",
                        chap_id, demoted)
        # Recompute summary after demotion
        summary = verify.get("summary") or {}
        if demoted:
            counts: dict[str, int] = {}
            for s in verify.get("sentences", []):
                counts[s.get("status", "unknown")] = counts.get(s.get("status", "unknown"), 0) + 1
            verify["summary"] = counts
        write_json(verify_path, verify)

    # 3. Strip unsupported
    verified_md, removed = _strip_unsupported(draft_md, verify)
    (out_dir / "verified.md").write_text(verified_md)
    if removed:
        log.warning("Chapter %s: stripped %d unsupported sentences", chap_id, len(removed))

    # 4. Polish
    log.info("Chapter %s: polishing", chap_id)
    polished_md = _polish_chapter(client, cfg, chapter, verified_md, store)
    polished_path.write_text(polished_md)

    # 5. Record citations to the audit store
    store.record_citations(chap_id, verify.get("sentences", []))

    # 6. Per-chapter audit summary
    audit = {
        "chapter_id": chap_id,
        "title": chapter["title"],
        "summary": verify.get("summary", {}),
        "stripped_sentence_indices": removed,
    }
    write_json(out_dir / "audit.json", audit)
    log.info("Chapter %s: done. Summary=%s", chap_id, audit["summary"])


def run(cfg: Config) -> None:
    outline = read_json(cfg.path("outline"))
    store = ClaimStore(cfg.path("claims_db"))

    budget = BudgetTracker(
        log_path=cfg.path("reports") / "cost_log.jsonl",
        total_max_usd=cfg.budget("total_max"),
        per_chapter_max_usd=cfg.budget("per_chapter_max"),
        abort_on_exceed=cfg.pipeline["budget_usd"]["abort_on_exceed"],
    )
    client = LLMClient(cfg, budget)

    all_chapters = (outline.get("chapters") or []) + (outline.get("appendices") or [])
    for ch in all_chapters:
        _process_chapter(client, cfg, ch, store)
