"""Stage 07 - claims.

Runs the claim extractor LLM over each segment, parses JSON output, and
inserts atomic claims into the SQLite grounding store.
"""
from __future__ import annotations

import json
import re

from ..grounding import Claim, ClaimStore
from ..llm import BudgetTracker, LLMClient
from ..llm.client import Message, Block
from ..llm.prompts import load_prompt
from ..utils.cache import read_json
from ..utils.config import Config
from ..utils.hashing import sha256_text
from ..utils.logging import get_logger
from ..verification.code_validator import check_code

log = get_logger(__name__)


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)
    return text.strip()


def _build_segment_payload(segment: dict, video_id: str) -> str:
    """Render a segment as compact text for the extractor prompt."""
    lines = [f"# Segment from video {video_id}",
             f"# ts: {segment['ts_start']:.1f} - {segment['ts_end']:.1f}",
             f"# title (if any): {segment.get('title', '')}",
             "",
             "## Aligned transcript + visuals"]
    for s in segment["segments"]:
        ts = f"[{s['ts_start']:.1f}-{s['ts_end']:.1f}]"
        lines.append(f"{ts} {s['text']}")
        for v in s.get("visuals", []):
            lines.append(f"  FRAME {v['frame_id']} ({v.get('category', '?')}):")
            if v.get("title"):
                lines.append(f"    title: {v['title']}")
            if v.get("extracted_text"):
                lines.append(f"    text: {v['extracted_text'][:500]}")
            if v.get("code"):
                code = v["code"]
                lines.append(f"    code ({code.get('language', '')}):")
                for cl in (code.get("content") or "").splitlines():
                    lines.append(f"      {cl}")
            if v.get("math"):
                for m in v["math"]:
                    lines.append(f"    math: {m}")
            if v.get("diagram_summary"):
                lines.append(f"    diagram: {v['diagram_summary']}")
                if v.get("diagram_elements"):
                    lines.append(f"    elements: {', '.join(v['diagram_elements'])}")
    return "\n".join(lines)


def _extract_from_segment(
    client: LLMClient,
    cfg: Config,
    segment_payload: str,
    min_conf: float,
) -> list[dict]:
    prompt = load_prompt("claim_extract")
    msg = Message(role="user", blocks=[Block(type="text", text=segment_payload)])
    resp = client.call(
        stage="claims",
        model=cfg.model("extract"),
        system=prompt,
        messages=[msg],
        max_tokens=cfg.gen("max_output_tokens_extract"),
        temperature=cfg.gen("extract_temperature"),
    )
    raw = _strip_json_fence(resp.text)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning("Bad JSON from claim extractor: %s. Raw[:300]: %s", e, raw[:300])
        return []
    if not isinstance(data, list):
        log.warning("Claim extractor returned non-list: %s", type(data).__name__)
        return []
    return [c for c in data if c.get("confidence", 1.0) >= min_conf]


def run(cfg: Config) -> None:
    segments_dir = cfg.path("data") / "segments"
    store = ClaimStore(cfg.path("claims_db"))
    budget = BudgetTracker(
        log_path=cfg.path("reports") / "cost_log.jsonl",
        total_max_usd=cfg.budget("total_max"),
        per_chapter_max_usd=cfg.budget("per_chapter_max"),
        abort_on_exceed=cfg.pipeline["budget_usd"]["abort_on_exceed"],
    )
    client = LLMClient(cfg, budget)
    min_conf = cfg.threshold("min_claim_confidence")

    total = 0
    for seg_file in sorted(segments_dir.glob("*.json")):
        data = read_json(seg_file)
        vid = data["video_id"]
        # Skip if we already have claims for this video
        if store.claims_for_video(vid):
            log.info("Skip %s: already has claims in store", vid)
            continue
        new_claims: list[Claim] = []
        bad_code = 0
        for segment in data["segments"]:
            payload = _build_segment_payload(segment, vid)
            extracted = _extract_from_segment(client, cfg, payload, min_conf)
            for c in extracted:
                content = c.get("content", "").strip()
                ctype = c.get("type", "fact")
                extra = c.get("extra") if isinstance(c.get("extra"), dict) else None
                # Validate code claims syntactically
                if ctype == "code":
                    check = check_code(content, c.get("language"))
                    extra = (extra or {}) | {
                        "code_check": {
                            "parseable": check.parseable,
                            "language": check.language,
                            "reason": check.reason,
                        }
                    }
                    if not check.parseable:
                        bad_code += 1
                        log.warning(
                            "Unparseable code claim in %s @ %.0fs (%s): %s",
                            vid, segment["ts_start"], check.language, check.reason,
                        )
                new_claims.append(
                    Claim(
                        type=ctype,
                        content=content,
                        video_id=vid,
                        ts_start=float(c.get("ts_start", segment["ts_start"])),
                        ts_end=float(c.get("ts_end", segment["ts_end"])),
                        frame_ids=c.get("frame_ids", []) or [],
                        confidence=float(c.get("confidence", 1.0)),
                        source_hash=sha256_text(payload),
                        extra=extra,
                    )
                )
        if bad_code:
            log.warning("%s: %d code claims failed syntactic validation", vid, bad_code)
        ids = store.add_claims(new_claims)
        log.info("%s: extracted %d claims", vid, len(ids))
        total += len(ids)

    log.info("Total claims in store: %d (this run added %d)",
             len(store.all_claims()), total)
