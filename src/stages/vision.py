"""Stage 04 - vision.

Sends each surviving keyframe to Haiku for classification + extraction in
one call (per the gating discussion). Saves one JSON per frame.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from ..llm import BudgetTracker, LLMClient
from ..llm.client import Block, Message
from ..llm.prompts import load_prompt
from ..utils.cache import read_json, write_json
from ..utils.config import Config
from ..utils.logging import get_logger

log = get_logger(__name__)


def _strip_json_fence(text: str) -> str:
    """LLMs sometimes wrap JSON in ```json ... ``` despite instructions."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)
    return text.strip()


def _process_frame(
    client: LLMClient,
    cfg: Config,
    frame_record: dict,
    out_path: Path,
) -> bool:
    if out_path.exists():
        return False
    image_path = cfg.root / frame_record["path"]
    if not image_path.exists():
        log.warning("Missing image: %s", image_path)
        return False

    prompt = load_prompt("vision_extract")
    msg = Message(
        role="user",
        blocks=[
            Block(type="image", image_path=image_path, image_media_type="image/jpeg"),
            Block(type="text", text="Classify and extract this frame per the schema."),
        ],
    )

    try:
        resp = client.call(
            stage="vision",
            model=cfg.model("vision"),
            system=prompt,
            messages=[msg],
            max_tokens=cfg.gen("max_output_tokens_extract"),
            temperature=cfg.gen("extract_temperature"),
        )
    except Exception as e:
        log.error("Vision call failed for %s: %s", frame_record["frame_id"], e)
        return False

    raw = _strip_json_fence(resp.text)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("Bad JSON from vision for %s; storing raw", frame_record["frame_id"])
        parsed = {"category": "other", "skip": True, "raw_text": resp.text, "_parse_error": True}

    parsed["frame_id"] = frame_record["frame_id"]
    parsed["video_id"] = frame_record["video_id"]
    parsed["timestamp_sec"] = frame_record["timestamp_sec"]
    parsed["source_image"] = frame_record["path"]

    write_json(out_path, parsed)
    return True


def run(cfg: Config) -> None:
    frames_root = cfg.path("frames")
    vision_root = cfg.path("vision")
    vision_root.mkdir(parents=True, exist_ok=True)

    budget = BudgetTracker(
        log_path=cfg.path("reports") / "cost_log.jsonl",
        total_max_usd=cfg.budget("total_max"),
        per_chapter_max_usd=cfg.budget("per_chapter_max"),
        abort_on_exceed=cfg.pipeline["budget_usd"]["abort_on_exceed"],
    )
    client = LLMClient(cfg, budget)

    video_dirs = sorted([d for d in frames_root.iterdir() if d.is_dir()])
    for vd in video_dirs:
        idx_file = vd / "index.json"
        if not idx_file.exists():
            log.warning("No index.json in %s; skipping", vd)
            continue
        records = read_json(idx_file)
        out_dir = vision_root / vd.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, rec in enumerate(records):
            did_work = _process_frame(client, cfg, rec, out_dir / f"{rec['frame_id']}.json")
            # Free-tier limit: 15 RPM → ~4s/call minimum. Paid tier ignores this.
            if did_work and i < len(records) - 1:
                time.sleep(4)
        log.info("Vision pass complete for %s: %d frames", vd.name, len(records))
