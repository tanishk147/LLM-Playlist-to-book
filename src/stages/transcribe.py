"""Stage 02 - transcribe.

Runs faster-whisper on each ingested audio file. Outputs word-level JSON.
Uses the glossary file as the Whisper initial_prompt for biasing.
"""
from __future__ import annotations

from pathlib import Path

from ..utils.cache import read_json, write_json
from ..utils.config import Config
from ..utils.logging import get_logger

log = get_logger(__name__)


def _build_initial_prompt(glossary_path: Path) -> str:
    """Read glossary terms and turn them into a Whisper biasing prompt."""
    terms = []
    for line in glossary_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        terms.extend(line.split())
    seen = set()
    out: list[str] = []
    for t in terms:
        tl = t.lower()
        if tl in seen:
            continue
        seen.add(tl)
        out.append(t)
    return "Technical vocabulary: " + ", ".join(out[:200]) + "."


def _transcribe_one(audio_path: Path, out_path: Path, model: "WhisperModel", initial_prompt: str) -> None:
    """Transcribe a single audio file using a pre-loaded Whisper model."""

    log.info("Transcribing %s", audio_path.name)
    segments_iter, info = model.transcribe(
        str(audio_path),
        language="en",
        word_timestamps=True,
        initial_prompt=initial_prompt,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    segments = []
    for seg in segments_iter:
        words = [
            {"start": w.start, "end": w.end, "word": w.word, "probability": w.probability}
            for w in (seg.words or [])
        ]
        segments.append(
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "words": words,
            }
        )

    write_json(
        out_path,
        {
            "audio_path": str(audio_path.name),
            "language": info.language,
            "language_probability": info.language_probability,
            "duration_sec": info.duration,
            "model": model.model_size_or_path if hasattr(model, 'model_size_or_path') else "unknown",
            "segments": segments,
        },
    )
    log.info("Wrote %s (%d segments, %.1fs audio)", out_path.name, len(segments), info.duration)


def _get_num_gpus() -> int:
    """Detect how many CUDA GPUs are available."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except ImportError:
        pass
    return 0


def _worker(gpu_id: int, tasks: list[tuple[Path, Path]], model_name: str, initial_prompt: str) -> None:
    """Worker that loads a Whisper model on a specific GPU and processes its assigned videos."""
    from faster_whisper import WhisperModel  # imported lazily

    device = f"cuda" if gpu_id < 0 else f"cuda"
    log.info("GPU %d: Loading Whisper model: %s", gpu_id, model_name)
    model = WhisperModel(model_name, device="cuda", compute_type="float16", device_index=gpu_id)

    for audio_path, out_path in tasks:
        _transcribe_one(audio_path, out_path, model, initial_prompt)


def run(cfg: Config) -> None:
    raw = cfg.path("raw")
    out = cfg.path("transcripts")
    out.mkdir(parents=True, exist_ok=True)

    manifest = read_json(raw / "manifest.json")
    glossary = cfg.root / "config" / "glossary.txt"
    initial_prompt = _build_initial_prompt(glossary)

    model_name = cfg.model("whisper")

    # Collect pending (not yet transcribed) videos
    pending: list[tuple[Path, Path]] = []
    for entry in manifest:
        vid = entry["video_id"]
        audio_path = cfg.root / entry["audio_path"]
        out_path = out / f"{vid}.json"
        if out_path.exists():
            log.info("Skip %s: already transcribed", vid)
            continue
        pending.append((audio_path, out_path))

    if not pending:
        log.info("All videos already transcribed.")
        return

    num_gpus = _get_num_gpus()

    if num_gpus >= 2:
        # Dual-GPU mode: split videos evenly between GPUs and process in parallel
        from concurrent.futures import ThreadPoolExecutor, as_completed

        mid = len(pending) // 2
        gpu_tasks = [pending[:mid], pending[mid:]]
        log.info("Dual-GPU mode: GPU 0 gets %d videos, GPU 1 gets %d videos", len(gpu_tasks[0]), len(gpu_tasks[1]))

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = []
            for gpu_id, tasks in enumerate(gpu_tasks):
                if tasks:
                    futures.append(pool.submit(_worker, gpu_id, tasks, model_name, initial_prompt))
            for f in as_completed(futures):
                f.result()  # raises if a worker failed
    else:
        # Single-GPU or CPU fallback: sequential processing
        from faster_whisper import WhisperModel  # imported lazily

        log.info("Loading Whisper model: %s", model_name)
        model = WhisperModel(model_name, device="auto", compute_type="auto")

        for audio_path, out_path in pending:
            _transcribe_one(audio_path, out_path, model, initial_prompt)
