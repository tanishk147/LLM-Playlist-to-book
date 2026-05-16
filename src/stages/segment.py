"""Stage 06 - segment.

Produce topic boundaries for each video. Two strategies, in priority order:

1. If YouTube provided chapter markers in the metadata, use those.
2. Otherwise, embedding-based segmentation: split transcript into rolling
   windows, compute sentence-transformer embeddings, place a boundary
   wherever cosine similarity to the previous window drops below threshold.

The segmenter writes data/segments/<video_id>.json with a list of segments
that the claim extractor will iterate over.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from ..utils.cache import read_json, write_json
from ..utils.config import Config
from ..utils.logging import get_logger

log = get_logger(__name__)


def _segments_from_chapters(aligned: dict, chapters: list[dict]) -> list[dict]:
    """Group aligned transcript segments by YouTube-provided chapters."""
    out = []
    for ch in chapters:
        start = float(ch.get("start_time", 0.0))
        end = float(ch.get("end_time", aligned["duration_sec"] or 0.0))
        segs = [s for s in aligned["segments"] if s["ts_start"] >= start and s["ts_end"] <= end]
        if not segs:
            continue
        out.append(
            {
                "id": f"{aligned['video_id']}_{len(out):03d}",
                "title": ch.get("title") or "",
                "ts_start": start,
                "ts_end": end,
                "segments": segs,
            }
        )
    return out


def _flatten_words(segs: Iterable[dict]) -> int:
    n = 0
    for s in segs:
        n += len(s["text"].split())
    return n


def _segments_from_embeddings(aligned: dict, cfg: Config) -> list[dict]:
    from sentence_transformers import SentenceTransformer

    threshold = cfg.threshold("segment_similarity_threshold")
    min_words = int(cfg.threshold("segment_min_words"))

    segments = aligned["segments"]
    if not segments:
        return []

    # Group transcript into ~30s windows for embedding
    windows: list[dict] = []
    cur: list[dict] = []
    cur_dur = 0.0
    for s in segments:
        cur.append(s)
        cur_dur += s["ts_end"] - s["ts_start"]
        if cur_dur >= 30.0:
            windows.append(
                {
                    "ts_start": cur[0]["ts_start"],
                    "ts_end": cur[-1]["ts_end"],
                    "text": " ".join(s["text"] for s in cur),
                    "raw_segments": cur,
                }
            )
            cur = []
            cur_dur = 0.0
    if cur:
        windows.append(
            {
                "ts_start": cur[0]["ts_start"],
                "ts_end": cur[-1]["ts_end"],
                "text": " ".join(s["text"] for s in cur),
                "raw_segments": cur,
            }
        )

    if len(windows) <= 1:
        return [
            {
                "id": f"{aligned['video_id']}_000",
                "title": "",
                "ts_start": aligned["segments"][0]["ts_start"],
                "ts_end": aligned["segments"][-1]["ts_end"],
                "segments": aligned["segments"],
            }
        ]

    log.info("Loading sentence-transformer (this may take a moment)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embs = model.encode([w["text"] for w in windows], normalize_embeddings=True)

    # Boundary wherever consecutive cosine similarity drops below threshold
    boundaries = [0]
    for i in range(1, len(windows)):
        sim = float(np.dot(embs[i], embs[i - 1]))
        if sim < threshold:
            boundaries.append(i)
    boundaries.append(len(windows))

    out: list[dict] = []
    for k in range(len(boundaries) - 1):
        a, b = boundaries[k], boundaries[k + 1]
        win_slice = windows[a:b]
        all_segs = [s for w in win_slice for s in w["raw_segments"]]
        if _flatten_words(all_segs) < min_words and out:
            # Merge tiny tail into previous
            out[-1]["segments"].extend(all_segs)
            out[-1]["ts_end"] = all_segs[-1]["ts_end"]
            continue
        out.append(
            {
                "id": f"{aligned['video_id']}_{len(out):03d}",
                "title": "",
                "ts_start": all_segs[0]["ts_start"],
                "ts_end": all_segs[-1]["ts_end"],
                "segments": all_segs,
            }
        )
    return out


def run(cfg: Config) -> None:
    aligned_dir = cfg.path("data") / "aligned"
    out_dir = cfg.path("data") / "segments"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(cfg.path("raw") / "manifest.json")

    for entry in manifest:
        vid = entry["video_id"]
        aligned_path = aligned_dir / f"{vid}.json"
        if not aligned_path.exists():
            log.warning("No aligned data for %s", vid)
            continue
        aligned = read_json(aligned_path)
        chapters = entry.get("chapters") or []

        if chapters:
            log.info("%s: using YouTube chapter markers (%d)", vid, len(chapters))
            segments = _segments_from_chapters(aligned, chapters)
        else:
            log.info("%s: no chapters, running embedding-based segmentation", vid)
            segments = _segments_from_embeddings(aligned, cfg)

        write_json(out_dir / f"{vid}.json", {"video_id": vid, "segments": segments})
        log.info("%s: %d segments", vid, len(segments))
