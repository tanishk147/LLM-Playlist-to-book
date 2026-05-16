"""Stage 01 - ingest.

Uses yt-dlp to download audio + metadata for each video in the playlist.
Output: data/raw/<video_id>.m4a and data/raw/<video_id>.info.json.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from ..utils.cache import write_json
from ..utils.config import Config, ensure_dirs
from ..utils.hashing import sha256_file
from ..utils.logging import get_logger

log = get_logger(__name__)


def _ydl_cmd(
    playlist_url: str,
    out_dir: Path,
    audio_only: bool = True,
    pilot: bool = False,
) -> list[str]:
    """Construct the yt-dlp command. URL is always the final element."""
    cmd = [
        "yt-dlp",
        "--ignore-errors",
        "--no-warnings",
        "--write-info-json",
        "--write-auto-sub",
        "--sub-lang", "en",
        "--sub-format", "vtt",
        "--restrict-filenames",
        "--output", str(out_dir / "%(id)s.%(ext)s"),
        "--print-to-file", "id", str(out_dir / "_video_ids.txt"),
    ]
    if audio_only:
        cmd += ["-f", "bestaudio[ext=m4a]/bestaudio", "--extract-audio", "--audio-format", "m4a"]
    if pilot:
        cmd += ["--playlist-items", "1"]
    cookies = os.environ.get("YT_COOKIES_PATH", "")
    if cookies and Path(cookies).exists():
        cmd += ["--cookies", cookies]
    cmd.append(playlist_url)
    return cmd


def _read_video_ids(out_dir: Path) -> list[str]:
    p = out_dir / "_video_ids.txt"
    if not p.exists():
        return []
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


def run(cfg: Config) -> None:
    ensure_dirs(cfg)
    raw = cfg.path("raw")
    playlist_url = cfg.playlist["playlist_url"]
    excluded = set(cfg.playlist.get("exclude_video_ids") or [])

    manifest_path = raw / "manifest.json"
    if manifest_path.exists():
        log.info("Skip ingest: manifest.json already exists")
        return

    log.info("Downloading playlist (audio only): %s", playlist_url)
    if cfg.pilot:
        log.warning("PILOT mode: only first video will be ingested")
    # Truncate ID file before running: yt-dlp --print-to-file APPENDS, so
    # without this a re-run accumulates stale IDs from previous runs.
    ids_file = raw / "_video_ids.txt"
    ids_file.unlink(missing_ok=True)
    cmd = _ydl_cmd(playlist_url, raw, pilot=cfg.pilot)

    log.debug("yt-dlp cmd: %s", " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"yt-dlp failed with code {res.returncode}")

    video_ids = [v for v in _read_video_ids(raw) if v not in excluded]
    if not video_ids:
        raise RuntimeError("No videos ingested. Check playlist URL or yt-dlp output.")

    # In pilot mode, hard-cap to 1 video regardless of what yt-dlp reported.
    # yt-dlp with v=...&list=... URLs may enumerate all playlist IDs into the
    # print-to-file even when --playlist-items 1 limits actual downloads.
    if cfg.pilot and len(video_ids) > 1:
        log.warning("PILOT: capping manifest to first video (%s)", video_ids[0])
        video_ids = video_ids[:1]

    # Build a manifest of what we got
    manifest = []
    for vid in video_ids:
        audio = raw / f"{vid}.m4a"
        info = raw / f"{vid}.info.json"
        if not audio.exists() or not info.exists():
            log.warning("Missing files for %s; skipping in manifest", vid)
            continue
        info_data = json.loads(info.read_text())
        manifest.append(
            {
                "video_id": vid,
                "title": info_data.get("title"),
                "duration_sec": info_data.get("duration"),
                "uploader": info_data.get("uploader"),
                "upload_date": info_data.get("upload_date"),
                "chapters": info_data.get("chapters") or [],
                "audio_path": str(audio.relative_to(cfg.root)),
                "audio_sha256": sha256_file(audio),
            }
        )

    write_json(raw / "manifest.json", manifest)
    log.info("Ingested %d videos. Manifest: %s", len(manifest), raw / "manifest.json")
