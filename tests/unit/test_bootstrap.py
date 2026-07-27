"""Unit tests for group-aware bootstrap confidence intervals."""

import numpy as np
import pytest

from causalmask.statistics.bootstrap import (
    group_aware_bootstrap_ci,
    bootstrap_mean_ci,
    paired_bootstrap_diff,
)


class TestGroupAwareBootstrap:
    def test_output_structure(self):
        values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        groups = np.array([0, 0, 1, 1, 2, 2])
        ci = group_aware_bootstrap_ci(values, groups, n_bootstrap=200, seed=42)
        assert "point_estimate" in ci
        assert "ci_lower" in ci
        assert "ci_upper" in ci
        assert ci["n_valid"] == 6
        assert ci["n_groups"] == 3

    def test_point_estimate_is_mean(self):
        values = np.array([0.1, 0.2, 0.3, 0.4])
        groups = np.array([0, 0, 1, 1])
        ci = group_aware_bootstrap_ci(values, groups, n_bootstrap=200, seed=42)
        expected = (0.1 + 0.2 + 0.3 + 0.4) / 4
        assert ci["point_estimate"] == pytest.approx(expected)

    def test_ci_bounds_contain_point(self):
        rng = np.random.default_rng(42)
        values = rng.normal(0.5, 0.1, 200)
        groups = np.repeat(np.arange(20), 10)
        ci = group_aware_bootstrap_ci(values, groups, n_bootstrap=200, seed=42)
        assert ci["ci_lower"] <= ci["point_estimate"] <= ci["ci_upper"]

    def test_single_group(self):
        values = np.array([0.1, 0.2, 0.3])
        groups = np.array([0, 0, 0])
        ci = group_aware_bootstrap_ci(values, groups, n_bootstrap=100, seed=42)
        assert ci["n_groups"] == 1

    def test_all_nan_handled(self):
        values = np.array([float("nan"), float("nan")])
        groups = np.array([0, 1])
        ci = group_aware_bootstrap_ci(values, groups, seed=42)
        assert np.isnan(ci["point_estimate"])
        assert ci["n_valid"] == 0

    def test_with_nan_mixed(self):
        values = np.array([0.5, float("nan"), 0.7])
        groups = np.array([0, 1, 2])
        ci = group_aware_bootstrap_ci(values, groups, seed=42)
        assert ci["n_valid"] == 2

    def test_median_statistic(self):
        values = np.array([0.1, 0.9, 0.9, 0.9, 0.9])
        groups = np.array([0, 1, 1, 1, 1])
        ci = group_aware_bootstrap_ci(values, groups, statistic=np.median,
                                      n_bootstrap=100, seed=42)
        assert ci["point_estimate"] == 0.9

    def test_deterministic_given_seed(self):
        values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        groups = np.array([0, 1, 2, 3, 4])
        ci1 = group_aware_bootstrap_ci(values, groups, n_bootstrap=100, seed=42)
        ci2 = group_aware_bootstrap_ci(values, groups, n_bootstrap=100, seed=42)
        assert ci1["ci_lower"] == ci2["ci_lower"]
        assert ci1["ci_upper"] == ci2["ci_upper"]


class TestBootstrapMeanCI:
    def test_output_structure(self):
        ci = bootstrap_mean_ci(np.arange(10, dtype=float), n_bootstrap=100, seed=42)
        assert ci["n_valid"] == 10
        assert ci["ci_lower"] <= ci["point_estimate"] <= ci["ci_upper"]

    def test_single_value(self):
        ci = bootstrap_mean_ci(np.array([0.5]), n_bootstrap=50, seed=42)
        assert ci["point_estimate"] == 0.5
        assert ci["ci_lower"] == 0.5
        assert ci["ci_upper"] == 0.5

    def test_empty_values(self):
        ci = bootstrap_mean_ci(np.array([]), seed=42)
        assert np.isnan(ci["point_estimate"])


class TestPairedBootstrapDiff:
    def test_output_structure(self):
        a = np.arange(10, dtype=float)
        b = np.arange(10, dtype=float) + 0.5
        ci = paired_bootstrap_diff(a, b, n_bootstrap=100, seed=42)
        assert ci["paired_n"] == 10
        assert "p_approx" in ci

    def test_zero_difference(self):
        a = np.array([0.5, 0.5, 0.5])
        b = np.array([0.5, 0.5, 0.5])
        ci = paired_bootstrap_diff(a, b, n_bootstrap=100, seed=42)
        assert ci["mean_diff"] == pytest.approx(0.0)

    def test_positive_difference(self):
        a = np.array([0.8, 0.9, 0.7])
        b = np.array([0.2, 0.3, 0.1])
        ci = paired_bootstrap_diff(a, b, n_bootstrap=100, seed=42)
        assert ci["mean_diff"] > 0.3

    def test_with_nan(self):
        a = np.array([0.8, float("nan"), 0.7])
        b = np.array([0.2, 0.3, 0.1])
        ci = paired_bootstrap_diff(a, b, n_bootstrap=100, seed=42)
        assert ci["paired_n"] == 2

    def test_group_aware(self):
        a = np.array([0.8, 0.7, 0.9, 0.85])
        b = np.array([0.2, 0.3, 0.1, 0.15])
        groups = np.array([0, 0, 1, 1])
        ci = paired_bootstrap_diff(a, b, group_ids=groups, n_bootstrap=100, seed=42)
        assert ci["ci_lower"] <= ci["mean_diff"] <= ci["ci_upper"]
