"""Reproducibility helpers for deterministic experiments."""

import os
import random
import sys

import numpy as np
import torch


def set_global_seed(seed: int, deterministic_algorithms: bool = True) -> None:
    """Set seeds for Python random, NumPy, PyTorch CPU, and PyTorch CUDA.

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

    # cuDNN determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Ensure deterministic DataLoader workers
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def seed_worker(worker_id: int) -> None:
    """Seed DataLoader workers for reproducibility.

    Use with DataLoader(worker_init_fn=seed_worker).
    """
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_torch_generator(seed: int = 42) -> torch.Generator:
    """Return a seeded Generator for DataLoader shuffling."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


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
