"""Calibration evaluation for classifier predictions.

Computes:
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Brier score
- Reliability diagram bins
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_ece(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute Expected Calibration Error (ECE) and related metrics.

    Uses equal-width binning in the probability range.

    Args:
        labels: Ground-truth labels (0 or 1).
        probabilities: Predicted probabilities for the positive class.
        n_bins: Number of equal-width bins.

    Returns:
        Dict with ece, mce, bin_data, and brier_score.
    """
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)

    if len(labels) == 0:
        return {
            "ece": float("nan"),
            "mce": float("nan"),
            "brier_score": float("nan"),
            "n_bins": n_bins,
            "n_samples": 0,
            "bin_data": [],
        }

    # Brier score
    brier = float(np.mean((probabilities - labels) ** 2))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(probabilities, bin_edges, right=False) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    mce = 0.0
    bin_data = []

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        bin_count = int(mask.sum())
        if bin_count == 0:
            continue

        bin_confidence = float(probabilities[mask].mean())
        bin_accuracy = float(labels[mask].mean())
        bin_weight = bin_count / max(len(labels), 1)
        bin_error = abs(bin_confidence - bin_accuracy)

        ece += bin_weight * bin_error
        mce = max(mce, bin_error)

        bin_data.append({
            "bin": bin_idx,
            "bin_lower": float(bin_edges[bin_idx]),
            "bin_upper": float(bin_edges[bin_idx + 1]),
            "n_samples": bin_count,
            "confidence": round(bin_confidence, 6),
            "accuracy": round(bin_accuracy, 6),
            "error": round(bin_error, 6),
        })

    return {
        "ece": round(ece, 6),
        "mce": round(mce, 6),
        "brier_score": round(brier, 6),
        "n_bins": n_bins,
        "n_samples": len(labels),
        "bin_data": bin_data,
    }


def compute_calibration_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Full calibration evaluation.

    Args:
        labels: Ground-truth labels (0 or 1).
        probabilities: Predicted probabilities for positive class.
        n_bins: Number of calibration bins.

    Returns:
        Dict with ECE, MCE, Brier, and bin data.
    """
    return compute_ece(labels, probabilities, n_bins=n_bins)


def evaluate_calibration_from_predictions(
    predictions_path: Path,
    prob_col: str = "prob_malignant",
    label_col: str = "label",
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute calibration metrics from saved predictions.

    Args:
        predictions_path: Path to parquet prediction file.
        prob_col: Column name for positive-class probability.
        label_col: Column name for ground-truth label.
        n_bins: Number of calibration bins.

    Returns:
        Dict with calibration metrics.
    """
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions not found: {predictions_path}")

    df = pd.read_parquet(predictions_path)
    labels = df[label_col].values
    probs = df[prob_col].values

    metrics = compute_ece(labels, probs, n_bins=n_bins)
    metrics["predictions_path"] = str(predictions_path)
    metrics["n_predictions"] = len(df)

    return metrics


def save_calibration_json(metrics: dict[str, Any], path: Path) -> None:
    """Save calibration metrics to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Saved calibration metrics: {path}")
