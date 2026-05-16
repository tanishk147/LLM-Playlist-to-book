"""Stage 03 - keyframes.

For each video, we need the original video stream (not just audio) to extract
visual keyframes. We re-fetch video with yt-dlp on demand, extract scene-change
frames via ffmpeg, dedupe with perceptual hashing, and filter out low-information
frames (likely talking-head) using edge density.

Output: data/frames/<video_id>/<frame_id>.jpg + index.json per video.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import imagehash
from PIL import Image, ImageFilter

from ..utils.cache import read_json, write_json
from ..utils.config import Config
from ..utils.hashing import sha256_file
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class FrameRecord:
    frame_id: str
    video_id: str
    timestamp_sec: float
    path: str
    phash: str
    edge_density: float
    sha256: str


def _download_video(video_id: str, out_path: Path) -> None:
    if out_path.exists():
        return
    log.info("Downloading video stream for %s", video_id)
    cmd = [
        "yt-dlp",
        "--no-warnings",
        "-f", "best[height<=720][ext=mp4]/best[ext=mp4]/best",
        "-o", str(out_path),
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"Video download failed for {video_id}")


def _extract_scene_frames(
    video_path: Path,
    out_dir: Path,
    scene_threshold: float,
) -> list[tuple[float, Path]]:
    """Use ffmpeg to extract scene-change frames. Returns list of (ts, path).

    Approach: the `metadata=print:file=` filter writes per-frame metadata to
    a sidecar text file with a stable format. Each scene-change pass emits a
    block containing `pts_time=<seconds>`. This is far more robust than
    parsing stderr/showinfo, which changes between ffmpeg versions.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "frame_%05d.jpg")
    meta_path = out_dir / "_frames_meta.txt"
    if meta_path.exists():
        meta_path.unlink()

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(video_path),
        "-vf",
        (
            f"select='gt(scene,{scene_threshold})',"
            f"metadata=print:file={meta_path}"
        ),
        "-vsync", "vfr",
        "-q:v", "3",
        pattern,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        log.error("ffmpeg stderr (tail):\n%s", res.stderr[-2000:])
        raise RuntimeError("ffmpeg scene extraction failed")

    timestamps = _parse_metadata_timestamps(meta_path)

    files = sorted(out_dir.glob("frame_*.jpg"))
    pairs: list[tuple[float, Path]] = []
    if len(timestamps) != len(files):
        log.warning(
            "Timestamp/frame count mismatch (%d ts vs %d files). "
            "Falling back to ffprobe per-frame.",
            len(timestamps), len(files),
        )
        # Best-effort fallback: probe each emitted JPEG and use file mtime ordinal
        for i, f in enumerate(files):
            ts = timestamps[i] if i < len(timestamps) else float(i)
            pairs.append((ts, f))
    else:
        for ts, f in zip(timestamps, files):
            pairs.append((ts, f))
    return pairs


def _parse_metadata_timestamps(meta_path: Path) -> list[float]:
    """Parse `pts_time=<sec>` lines emitted by ffmpeg's metadata filter."""
    if not meta_path.exists():
        return []
    out: list[float] = []
    for line in meta_path.read_text().splitlines():
        line = line.strip()
        # The metadata filter writes lines like:
        #   frame:0    pts:12345     pts_time:0.5
        # We grab the pts_time= or pts_time: token.
        for sep in ("pts_time=", "pts_time:"):
            idx = line.find(sep)
            if idx >= 0:
                tail = line[idx + len(sep):].split()[0]
                try:
                    out.append(float(tail))
                except ValueError:
                    pass
                break
    return out


def _edge_density(img: Image.Image) -> float:
    """Cheap edge density: variance of a small grayscale Laplacian-ish pass."""
    g = img.convert("L").resize((256, 144))
    edges = g.filter(ImageFilter.FIND_EDGES)
    # Mean pixel intensity of the edge image, normalized to [0, 1].
    hist = edges.histogram()
    total = sum(hist)
    if total == 0:
        return 0.0
    s = sum(i * c for i, c in enumerate(hist))
    return (s / total) / 255.0


def _dedupe_and_filter(
    pairs: list[tuple[float, Path]],
    min_edge_density: float,
    phash_distance: int,
    min_interval: float,
    max_frames: int,
) -> list[tuple[float, Path]]:
    """Apply (1) edge-density filter (drops talking-head), (2) perceptual dedup,
    (3) minimum spacing, (4) global cap."""
    kept: list[tuple[float, Path, imagehash.ImageHash]] = []
    last_ts = -math.inf
    for ts, path in pairs:
        try:
            with Image.open(path) as im:
                im.load()
                ed = _edge_density(im)
                ph = imagehash.phash(im)
        except Exception as e:
            log.warning("Could not load %s: %s", path, e)
            continue

        if ed < min_edge_density:
            path.unlink(missing_ok=True)
            continue
        if ts - last_ts < min_interval:
            path.unlink(missing_ok=True)
            continue
        if any(ph - prev_ph <= phash_distance for _, _, prev_ph in kept):
            path.unlink(missing_ok=True)
            continue
        kept.append((ts, path, ph))
        last_ts = ts

    # Apply global cap by uniform downsampling
    if len(kept) > max_frames:
        step = len(kept) / max_frames
        idxs = {int(i * step) for i in range(max_frames)}
        new: list[tuple[float, Path, imagehash.ImageHash]] = []
        for i, item in enumerate(kept):
            if i in idxs:
                new.append(item)
            else:
                item[1].unlink(missing_ok=True)
        kept = new

    return [(ts, p) for ts, p, _ in kept]


def _process_video(video_id: str, work_dir: Path, frames_dir: Path, cfg: Config) -> None:
    out_dir = frames_dir / video_id
    if out_dir.exists() and (out_dir / "index.json").exists():
        log.info("Skip %s: frames already extracted", video_id)
        return

    raw_video = work_dir / f"{video_id}.mp4"
    _download_video(video_id, raw_video)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = _extract_scene_frames(
        raw_video,
        out_dir,
        scene_threshold=cfg.threshold("scene_change"),
    )
    log.info("%s: %d candidate keyframes after scene detect", video_id, len(pairs))

    pairs = _dedupe_and_filter(
        pairs,
        min_edge_density=cfg.threshold("min_edge_density"),
        phash_distance=int(cfg.threshold("phash_distance")),
        min_interval=cfg.threshold("min_frame_interval_sec"),
        max_frames=int(cfg.threshold("vision_max_frames_per_video")),
    )
    log.info("%s: %d frames kept after dedup+filter", video_id, len(pairs))

    # Rename to stable IDs and write index
    records: list[FrameRecord] = []
    for i, (ts, path) in enumerate(pairs):
        frame_id = f"{video_id}_f{i:04d}"
        new_path = out_dir / f"{frame_id}.jpg"
        path.rename(new_path)
        with Image.open(new_path) as im:
            ph = str(imagehash.phash(im))
            ed = _edge_density(im)
        records.append(
            FrameRecord(
                frame_id=frame_id,
                video_id=video_id,
                timestamp_sec=ts,
                path=os.path.relpath(new_path, cfg.root),
                phash=ph,
                edge_density=ed,
                sha256=sha256_file(new_path),
            )
        )

    # Remove leftover untouched files
    for stale in out_dir.glob("frame_*.jpg"):
        stale.unlink(missing_ok=True)

    write_json(out_dir / "index.json", [r.__dict__ for r in records])


def run(cfg: Config) -> None:
    raw = cfg.path("raw")
    frames = cfg.path("frames")
    work = cfg.path("data") / "_video_cache"
    work.mkdir(parents=True, exist_ok=True)

    manifest = read_json(raw / "manifest.json")
    for entry in manifest:
        _process_video(entry["video_id"], work, frames, cfg)
