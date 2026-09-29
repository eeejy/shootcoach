"""Pick the best available compute device on any machine (Mac / NVIDIA / CPU)."""
from __future__ import annotations

from shootcoach.config import REPO_ROOT

MODELS = REPO_ROOT / "models"


def best_device() -> str | int:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return 0
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def model_path(name: str) -> str:
    """Prefer the copy shipped in models/ (works offline, e.g. on an intranet); else let Ultralytics download."""
    p = MODELS / name
    return str(p) if p.exists() else name
