"""Reproducibility helpers for deterministic experiments."""

from __future__ import annotations

import json
import os
import platform
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def set_global_seed(seed: int, deterministic_algorithms: bool = True) -> None:
    """Set seeds for Python, NumPy, PyTorch CPU and CUDA.

    Also configures cuDNN and CUDA determinism for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def seed_worker(worker_id: int) -> None:
    """Seed DataLoader workers for reproducibility."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_torch_generator(seed: int = 42) -> torch.Generator:
    """Return a seeded Generator for DataLoader shuffling."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def resolve_project_root(marker: str = "CausalMask-XAI.md") -> Path:
    """Resolve the project root by walking up from cwd or checking env."""
    env_root = os.environ.get("CAUSALMASK_PROJECT_ROOT")
    if env_root:
        p = Path(env_root)
        if (p / marker).exists():
            return p.resolve()
    cwd = Path.cwd()
    for candidate in [cwd] + list(cwd.parents):
        if (candidate / marker).exists():
            return candidate.resolve()
    colab_fallback = Path("/content/CausalMask-XAI")
    if colab_fallback.exists() and (colab_fallback / marker).exists():
        return colab_fallback.resolve()
    raise RuntimeError(
        f"Cannot find {marker}. "
        "Set CAUSALMASK_PROJECT_ROOT or run from within the repo."
    )


def capture_environment(project_root: Path | None = None) -> dict:
    """Capture environment summary for run reproducibility."""
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "torch": torch.__version__,
        "torchvision": None,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "seed": None,
    }
    try:
        import torchvision
        info["torchvision"] = torchvision.__version__
    except (ImportError, AttributeError):
        pass
    if project_root is not None:
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5,
                cwd=str(project_root),
            )
            info["git_commit"] = result.stdout.strip() if result.returncode == 0 else None
            result2 = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=5,
                cwd=str(project_root),
            )
            info["git_dirty"] = bool(result2.stdout.strip())
        except Exception:
            info["git_commit"] = None
            info["git_dirty"] = None
    return info


def save_environment_json(info: dict, path: Path) -> None:
    """Save environment info to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(info, f, indent=2, default=str)


def configure_reproducibility(seed: int = 42) -> dict:
    """Full reproducibility setup. Returns a summary dict for logging."""
    set_global_seed(seed)
    info = {
        "seed": seed,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "torch_deterministic_algorithms": True,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_available": torch.cuda.is_available(),
    }
    return info
