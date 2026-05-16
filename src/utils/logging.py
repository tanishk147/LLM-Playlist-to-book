"""Structured logging with rich console output."""
from __future__ import annotations

import logging
import os
import sys

from rich.console import Console
from rich.logging import RichHandler

_CONSOLE = Console(stderr=True)
_CONFIGURED = False


def setup_logging(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = level or os.environ.get("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=_CONSOLE, rich_tracebacks=True, show_path=False)],
    )
    # Quiet down noisy libs
    for noisy in ("urllib3", "httpx", "httpcore", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


def console() -> Console:
    return _CONSOLE
