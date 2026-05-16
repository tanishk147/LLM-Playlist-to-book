"""Stage 05 - align.

Joins transcript segments with frame extractions by timestamp. Produces one
aligned JSON per video where each transcript segment carries the IDs of any
frames whose timestamps fall within its range.

This is the unit of grounding for the claims extractor.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..utils.cache import read_json, write_json
from ..utils.config import Config
from ..utils.logging import get_logger

log = get_logger(__name__)


def _frames_in_range(frame_index: list[dict], start: float, end: float) -> list[str]:
    return [f["frame_id"] for f in frame_index if start <= f["timestamp_sec"] <= end]


def _load_vision_extractions(vision_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not vision_dir.exists():
        return out
    for jf in vision_dir.glob("*.json"):
        data = json.loads(jf.read_text())
        if data.get("skip"):
            continue
        out[data["frame_id"]] = data
    return out


def _build_aligned(
    video_id: str,
    transcript: dict,
    frame_index: list[dict],
    vision: dict[str, dict],
) -> dict:
    aligned_segments = []
    for seg in transcript["segments"]:
        frame_ids = _frames_in_range(frame_index, seg["start"], seg["end"])
        visuals = [vision[fid] for fid in frame_ids if fid in vision]
        aligned_segments.append(
            {
                "ts_start": seg["start"],
                "ts_end": seg["end"],
                "text": seg["text"].strip(),
                "frame_ids": frame_ids,
                "visuals": visuals,
            }
        )
    return {
        "video_id": video_id,
        "duration_sec": transcript.get("duration_sec"),
        "segments": aligned_segments,
        "frames_total": len(frame_index),
        "frames_with_content": len(vision),
    }


def run(cfg: Config) -> None:
    transcripts_dir = cfg.path("transcripts")
    frames_dir = cfg.path("frames")
    vision_dir = cfg.path("vision")
    out_dir = cfg.path("data") / "aligned"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_json(cfg.path("raw") / "manifest.json")
    for entry in manifest:
        vid = entry["video_id"]
        tpath = transcripts_dir / f"{vid}.json"
        ipath = frames_dir / vid / "index.json"
        if not tpath.exists() or not ipath.exists():
            log.warning("Missing transcript or frame index for %s", vid)
            continue
        transcript = read_json(tpath)
        frame_index = read_json(ipath)
        vision = _load_vision_extractions(vision_dir / vid)
        aligned = _build_aligned(vid, transcript, frame_index, vision)
        write_json(out_dir / f"{vid}.json", aligned)
        log.info(
            "Aligned %s: %d segments, %d frames (%d with content)",
            vid,
            len(aligned["segments"]),
            aligned["frames_total"],
            aligned["frames_with_content"],
        )
