"""Classification evaluation metrics.

Computes standard diagnostic performance metrics from saved predictions.
All metrics are computed from prediction-level artifacts, not from
model re-execution.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    f1_score,
)

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
    positive_class: int = 1,
) -> dict[str, Any]:
    """Compute standard classification metrics.

    Args:
        labels: Ground-truth labels (0 or 1).
        probabilities: Predicted probabilities for the positive class.
        threshold: Decision threshold for binary classification.
        positive_class: Index of the positive class.

    Returns:
        Dict with metrics.
    """
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)

    if len(labels) == 0:
        return {"error": "empty input", "n_samples": 0}

    predictions = (probabilities >= threshold).astype(np.int64)

    metrics: dict[str, Any] = {}
    metrics["n_samples"] = int(len(labels))
    metrics["n_positive"] = int(labels.sum())
    metrics["n_negative"] = int(len(labels) - labels.sum())
    metrics["threshold"] = threshold

    # AUROC
    try:
        if len(np.unique(labels)) > 1:
            metrics["auroc"] = float(roc_auc_score(labels, probabilities))
        else:
            metrics["auroc"] = float("nan")
            logger.warning("Only one class present in labels; AUROC is NaN")
    except Exception as e:
        metrics["auroc"] = float("nan")
        logger.warning(f"AUROC computation failed: {e}")

    # Balanced accuracy
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(labels, predictions))

    # Accuracy
    metrics["accuracy"] = float((predictions == labels).mean())

    # Precision, Recall, F1
    metrics["precision"] = float(precision_score(labels, predictions, zero_division=0))
    metrics["recall"] = float(recall_score(labels, predictions, zero_division=0))
    metrics["f1"] = float(f1_score(labels, predictions, zero_division=0))

    # Specificity
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["true_positives"] = int(tp)
    metrics["specificity"] = float(tn / max(tn + fp, 1))
    metrics["sensitivity"] = float(tp / max(tp + fn, 1))

    # PR-AUC
    try:
        precision_curve, recall_curve, _ = precision_recall_curve(labels, probabilities)
        metrics["prauc"] = float(auc(recall_curve, precision_curve))
    except Exception as e:
        metrics["prauc"] = float("nan")
        logger.warning(f"PR-AUC computation failed: {e}")

    return metrics


def compute_youden_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """Compute Youden's J statistic optimal threshold.

    Returns:
        Threshold that maximizes sensitivity + specificity - 1.
    """
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    youden_j = tpr - fpr
    best_idx = int(np.argmax(youden_j))
    return float(thresholds[best_idx])


def load_predictions(path: Path) -> pd.DataFrame:
    """Load predictions from parquet file."""
    if not path.exists():
        raise FileNotFoundError(f"Predictions not found: {path}")
    return pd.read_parquet(path)


def evaluate_from_predictions(
    predictions_path: Path,
    prob_col: str = "prob_malignant",
    label_col: str = "label",
    threshold: float | None = None,
) -> dict[str, Any]:
    """Full evaluation pipeline from saved predictions.

    Args:
        predictions_path: Path to parquet prediction file.
        prob_col: Column name for positive-class probability.
        label_col: Column name for ground-truth label.
        threshold: Decision threshold. If None, computed via Youden's J.

    Returns:
        Dict with metrics and metadata.
    """
    df = load_predictions(predictions_path)
    labels = df[label_col].values
    probs = df[prob_col].values

    if threshold is None:
        threshold = compute_youden_threshold(labels, probs)
        logger.info(f"Computed Youden threshold: {threshold:.4f}")

    metrics = compute_classification_metrics(labels, probs, threshold=threshold)
    metrics["threshold_source"] = "youden_j" if threshold is None else "provided"
    metrics["predictions_path"] = str(predictions_path)
    metrics["n_predictions"] = len(df)

    return metrics


def save_metrics_json(metrics: dict[str, Any], path: Path) -> None:
    """Save metrics to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Saved metrics: {path}")
