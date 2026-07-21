"""Tests for calibration metrics."""

import numpy as np

from causalmask.evaluation.calibration import compute_ece


def test_perfect_calibration():
    labels = np.array([0, 0, 1, 1])
    probs = np.array([0.0, 0.0, 1.0, 1.0])
    result = compute_ece(labels, probs, n_bins=5)
    assert result["ece"] == 0.0
    assert result["mce"] == 0.0
    assert result["brier_score"] == 0.0


def test_maximally_miscalibrated():
    labels = np.array([0, 0, 1, 1])
    probs = np.array([1.0, 1.0, 0.0, 0.0])
    result = compute_ece(labels, probs, n_bins=5)
    assert result["ece"] > 0.0
    assert result["brier_score"] > 0.0


def test_ece_bounds():
    rng = np.random.default_rng(42)
    labels = rng.integers(0, 2, size=100)
    probs = rng.uniform(0, 1, size=100)
    result = compute_ece(labels, probs, n_bins=10)
    assert 0.0 <= result["ece"] <= 1.0
    assert 0.0 <= result["mce"] <= 1.0
    assert 0.0 <= result["brier_score"] <= 1.0
    assert result["n_samples"] == 100


def test_bin_data_structure():
    labels = np.array([0, 0, 0, 1, 1, 1])
    probs = np.array([0.1, 0.3, 0.5, 0.6, 0.8, 0.9])
    result = compute_ece(labels, probs, n_bins=5)
    assert "bin_data" in result
    assert len(result["bin_data"]) > 0
    for bin_entry in result["bin_data"]:
        assert "bin" in bin_entry
        assert "n_samples" in bin_entry
        assert "confidence" in bin_entry
        assert "accuracy" in bin_entry


def test_empty_input():
    labels = np.array([], dtype=np.int64)
    probs = np.array([], dtype=np.float64)
    result = compute_ece(labels, probs, n_bins=10)
    assert np.isnan(result["ece"])
    assert result["n_samples"] == 0
