"""Set deterministic seeds for every library that might use a PRNG.

Called once from the CLI before any stage runs. The seed is sourced from
config/pipeline.yaml (`reproducibility.seed`) with a sane default so old
configs keep working.
"""
from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> None:
    """Best-effort seed of python, numpy, and torch (if installed).

    Sets PYTHONHASHSEED in the environment too — only takes effect for
    subprocesses since the current interpreter has already initialized
    its hash randomization, but useful for child workers.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms where possible (slower but reproducible)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    except Exception:
        pass
