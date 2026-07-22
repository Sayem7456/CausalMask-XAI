"""Checkpoint save and resume for training runs.

Preserves:
- model state dict
- optimizer state dict
- scheduler state dict
- gradient scaler state (AMP)
- epoch and global step
- best checkpoint reference
- RNG states
- run configuration
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    epoch: int
    global_step: int
    model_state: dict[str, torch.Tensor]
    optimizer_state: dict[str, Any] | None = None
    scheduler_state: dict[str, Any] | None = None
    scaler_state: dict[str, Any] | None = None
    best_metric: float | None = None
    best_epoch: int | None = None
    config: dict[str, Any] | None = None
    rng_states: dict[str, Any] = field(default_factory=dict)
    metrics_history: list[dict[str, float]] | None = None


def save_checkpoint(
    checkpoint: Checkpoint,
    path: Path,
    is_best: bool = False,
) -> Path:
    """Save checkpoint to disk.

    Args:
        checkpoint: Checkpoint data.
        path: Path to save to.
        is_best: If True, also saves 'best.pt' copy.

    Returns:
        Path to saved checkpoint.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {
        "epoch": checkpoint.epoch,
        "global_step": checkpoint.global_step,
        "model_state": checkpoint.model_state,
        "optimizer_state": checkpoint.optimizer_state,
        "scheduler_state": checkpoint.scheduler_state,
        "scaler_state": checkpoint.scaler_state,
        "best_metric": checkpoint.best_metric,
        "best_epoch": checkpoint.best_epoch,
        "config": checkpoint.config,
        "rng_states": checkpoint.rng_states,
        "metrics_history": checkpoint.metrics_history,
    }

    torch.save(serializable, path)
    logger.info(f"Saved checkpoint: {path} (epoch {checkpoint.epoch})")

    if is_best:
        best_path = path.parent / "best.pt"
        torch.save(serializable, best_path)
        logger.info(f"Saved best checkpoint: {best_path}")

    return path


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    device: torch.device | None = None,
    strict: bool = True,
) -> Checkpoint:
    """Load checkpoint and restore model (and optionally optimizer/scheduler) state.

    Args:
        path: Path to checkpoint file.
        model: Model to load state into.
        optimizer: Optional optimizer to restore.
        scheduler: Optional scheduler to restore.
        scaler: Optional gradient scaler to restore.
        device: Device to map tensors to.
        strict: Whether to load model state strictly.

    Returns:
        Loaded Checkpoint object.

    Raises:
        FileNotFoundError: If checkpoint not found.
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    device = device or torch.device("cpu")
    data = torch.load(path, map_location=device, weights_only=False)

    model_state = data.get("model_state", {})
    if any(k.startswith("module.") for k in model_state):
        model_state = {k.removeprefix("module."): v for k, v in model_state.items()}
    model.load_state_dict(model_state, strict=strict)

    if optimizer is not None and data.get("optimizer_state"):
        optimizer.load_state_dict(data["optimizer_state"])

    if scheduler is not None and data.get("scheduler_state"):
        scheduler.load_state_dict(data["scheduler_state"])

    if scaler is not None and data.get("scaler_state"):
        scaler.load_state_dict(data["scaler_state"])

    checkpoint = Checkpoint(
        epoch=data.get("epoch", 0),
        global_step=data.get("global_step", 0),
        model_state=model_state,
        optimizer_state=data.get("optimizer_state"),
        scheduler_state=data.get("scheduler_state"),
        scaler_state=data.get("scaler_state"),
        best_metric=data.get("best_metric"),
        best_epoch=data.get("best_epoch"),
        config=data.get("config"),
        rng_states=data.get("rng_states", {}),
        metrics_history=data.get("metrics_history"),
    )

    logger.info(
        f"Loaded checkpoint: {path} "
        f"(epoch {checkpoint.epoch}, step {checkpoint.global_step})"
    )
    return checkpoint


def _make_numpy_state_safe(state: Any) -> tuple:
    """Convert numpy random state to a pickle-safe format.

    ``np.random.get_state()`` returns a tuple whose second element is an
    ndarray.  PyTorch 2.6+ ``torch.load(weights_only=True)`` rejects pickle
    globals needed to reconstruct that array.  We convert the array to bytes
    so the state survives ``weights_only=True`` serialization.
    """
    return (state[0], state[1].tobytes(), state[2], state[3], state[4])


def _restore_numpy_state(state: tuple) -> tuple:
    """Restore a numpy random state that may be in safe format.

    Handles both the original tuple-with-ndarray (for backward compatibility
    with checkpoints saved before the safe-format conversion) and the safe
    tuple-with-bytes format.
    """
    if isinstance(state[1], bytes):
        arr = np.frombuffer(state[1], dtype=np.uint32)
        return (state[0], arr, state[2], state[3], state[4])
    return state


def capture_rng_states(seed: int | None = None) -> dict[str, Any]:
    """Capture current RNG states for all relevant generators."""
    states = {
        "python_random": random.getstate(),
        "numpy_random": _make_numpy_state_safe(np.random.get_state()),
        "torch_cpu_rng": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["torch_cuda_rng"] = torch.cuda.get_rng_state_all()
    return states


def restore_rng_states(states: dict[str, Any]) -> None:
    """Restore previously captured RNG states."""
    if "python_random" in states:
        random.setstate(states["python_random"])
    if "numpy_random" in states:
        np.random.set_state(_restore_numpy_state(states["numpy_random"]))
    if "torch_cpu_rng" in states:
        torch.set_rng_state(states["torch_cpu_rng"])
    if "torch_cuda_rng" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["torch_cuda_rng"])


def find_latest_checkpoint(checkpoint_dir: Path, pattern: str = "epoch_*.pt") -> Path | None:
    """Find the latest checkpoint by epoch number in a directory.

    Args:
        checkpoint_dir: Directory to search.
        pattern: Glob pattern for checkpoint files.

    Returns:
        Path to latest checkpoint or None.
    """
    checkpoints = sorted(checkpoint_dir.glob(pattern))
    if not checkpoints:
        return None
    return checkpoints[-1]


def save_run_status(
    status_dir: Path,
    status: dict[str, Any],
) -> None:
    """Save run status JSON.

    Includes start/end timestamps, current state, and notes.
    """
    status_dir.mkdir(parents=True, exist_ok=True)
    path = status_dir / "status.json"
    with open(path, "w") as f:
        json.dump(status, f, indent=2, default=str)
    logger.info(f"Saved run status: {path}")
