"""CLI entrypoint.

Usage:
    python -m src.cli <stage>
    python -m src.cli report cost
    python -m src.cli report grounding

Stages: ingest transcribe keyframes vision align segment claims outline
        chapters figures typeset
"""
from __future__ import annotations

import argparse
import sys
import traceback

from .utils.config import load_config, ensure_dirs
from .utils.logging import get_logger, setup_logging
from .utils.seeding import seed_everything


def _dispatch_stage(name: str) -> None:
    """Lazy-import the stage module so we don't load heavy deps unless needed."""
    name = name.replace("-", "_")
    mod_name = f"src.stages.{name}"
    try:
        import importlib

        mod = importlib.import_module(mod_name)
    except ImportError as e:
        raise SystemExit(f"Unknown stage: {name} ({e})")
    if not hasattr(mod, "run"):
        raise SystemExit(f"Stage {name} has no run() function")
    cfg = load_config()
    seed = int(cfg.pipeline.get("reproducibility", {}).get("seed", 1729))
    seed_everything(seed)
    ensure_dirs(cfg)
    mod.run(cfg)


def _dispatch_report(kind: str) -> None:
    from .reports import cost as cost_report, grounding as grounding_report

    cfg = load_config()
    if kind == "cost":
        cost_report.run(cfg)
    elif kind == "grounding":
        grounding_report.run(cfg)
    elif kind == "all":
        cost_report.run(cfg)
        grounding_report.run(cfg)
    else:
        raise SystemExit(f"Unknown report kind: {kind}")


STAGES = (
    "ingest",
    "transcribe",
    "keyframes",
    "vision",
    "align",
    "segment",
    "claims",
    "outline",
    "chapters",
    "figures",
    "typeset",
)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    log = get_logger("cli")

    parser = argparse.ArgumentParser(prog="playlist-book")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for s in STAGES:
        sub.add_parser(s, help=f"Run stage: {s}")
    p_report = sub.add_parser("report", help="Generate reports")
    p_report.add_argument("kind", choices=["cost", "grounding", "all"])

    args = parser.parse_args(argv)

    try:
        if args.cmd in STAGES:
            _dispatch_stage(args.cmd)
        elif args.cmd == "report":
            _dispatch_report(args.kind)
        else:
            raise SystemExit(f"Unknown command: {args.cmd}")
    except SystemExit:
        raise
    except Exception as e:  # pragma: no cover
        log.error("Stage %s failed: %s", args.cmd, e)
        log.debug(traceback.format_exc())
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
