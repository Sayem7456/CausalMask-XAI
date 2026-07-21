"""Tests for classification evaluation metrics."""

import numpy as np

from causalmask.evaluation.classification import (
    compute_classification_metrics,
    compute_youden_threshold,
)


def test_perfect_classification():
    labels = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_classification_metrics(labels, probs, threshold=0.5)
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["sensitivity"] == 1.0


def test_all_wrong():
    labels = np.array([0, 0, 1, 1])
    probs = np.array([0.9, 0.8, 0.2, 0.1])
    metrics = compute_classification_metrics(labels, probs, threshold=0.5)
    assert metrics["accuracy"] == 0.0
    assert metrics["f1"] == 0.0


def test_auroc_binary():
    labels = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_classification_metrics(labels, probs, threshold=0.5)
    assert metrics["auroc"] == 1.0


def test_auroc_single_class():
    labels = np.array([0, 0, 0, 0])
    probs = np.array([0.1, 0.2, 0.3, 0.4])
    metrics = compute_classification_metrics(labels, probs, threshold=0.5)
    assert np.isnan(metrics["auroc"])


def test_youden_threshold():
    labels = np.array([0, 0, 0, 1, 1, 1])
    probs = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
    thresh = compute_youden_threshold(labels, probs)
    assert 0.3 <= thresh <= 0.6


def test_metric_ranges():
    labels = np.array([0, 0, 1, 1, 1, 0, 0, 1])
    probs = np.array([0.1, 0.3, 0.7, 0.6, 0.9, 0.2, 0.4, 0.8])
    metrics = compute_classification_metrics(labels, probs, threshold=0.5)
    for key in ["accuracy", "balanced_accuracy", "f1", "precision", "recall", "specificity", "sensitivity"]:
        assert 0.0 <= metrics[key] <= 1.0, f"{key}={metrics[key]} out of [0, 1]"
