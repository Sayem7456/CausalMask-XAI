"""Unit tests for causal faithfulness metrics (Phase 7)."""

import numpy as np
import pytest

from causalmask.evaluation.faithfulness import (
    raw_lesion_necessity,
    normalized_lesion_necessity,
    lesion_sufficiency,
    background_invariance,
    prediction_flip_rate,
    donor_stratified_invariance,
    lesion_vs_sham_difference,
    ensure_confidence_for_target,
    compute_per_sample_causal_metrics,
)
from causalmask.evaluation.causalmask_score import (
    harmonic_mean,
    arithmetic_mean,
    geometric_mean,
    compute_causalmask_harmonic,
    compute_causalmask_arithmetic,
    compute_causalmask_geometric,
    compute_all_aggregations,
    compute_aggregation_sensitivity,
)


@pytest.fixture
def p_orig():
    return np.array([0.2, 0.8], dtype=np.float32)


@pytest.fixture
def p_suff():
    return np.array([0.3, 0.7], dtype=np.float32)


@pytest.fixture
def p_removed_t():
    return np.array([0.6, 0.4], dtype=np.float32)


@pytest.fixture
def p_removed_ns():
    return np.array([0.55, 0.45], dtype=np.float32)


@pytest.fixture
def p_sham():
    return np.array([0.3, 0.7], dtype=np.float32)


# ---------------------------------------------------------------------------
# target confidence
# ---------------------------------------------------------------------------

class TestEnsureConfidenceForTarget:
    def test_array_input(self):
        probs = np.array([0.2, 0.8])
        assert ensure_confidence_for_target(probs, 0) == 0.2
        assert ensure_confidence_for_target(probs, 1) == 0.8

    def test_scalar_input(self):
        assert ensure_confidence_for_target(np.array([0.7]), 0) == 0.7

    def test_out_of_bounds(self):
        assert np.isnan(ensure_confidence_for_target(np.array([0.2, 0.8]), 5))


# ---------------------------------------------------------------------------
# raw lesion necessity
# ---------------------------------------------------------------------------

class TestRawLesionNecessity:
    def test_positive_necessity(self):
        n = raw_lesion_necessity(0.8, 0.3)
        assert n == pytest.approx(0.5)

    def test_negative_necessity(self):
        n = raw_lesion_necessity(0.3, 0.8)
        assert n == pytest.approx(-0.5)

    def test_no_change(self):
        n = raw_lesion_necessity(0.5, 0.5)
        assert n == pytest.approx(0.0)

    def test_range(self):
        assert -1.0 <= raw_lesion_necessity(1.0, 0.0) <= 1.0
        assert -1.0 <= raw_lesion_necessity(0.0, 1.0) <= 1.0


# ---------------------------------------------------------------------------
# normalized lesion necessity
# ---------------------------------------------------------------------------

class TestNormalizedLesionNecessity:
    def test_high_necessity(self):
        n = normalized_lesion_necessity(0.8, 0.2)
        assert 0.5 < n <= 1.0

    def test_no_change(self):
        n = normalized_lesion_necessity(0.5, 0.5)
        assert n == pytest.approx(0.0)

    def test_confidence_increases(self):
        n = normalized_lesion_necessity(0.3, 0.8)
        assert n == pytest.approx(0.0)

    def test_zero_original(self):
        n = normalized_lesion_necessity(0.0, 0.5)
        assert n == pytest.approx(0.0)
        assert not np.isnan(n)

    def test_range(self):
        for _ in range(100):
            po = np.random.uniform(0, 1)
            pr = np.random.uniform(0, 1)
            n = normalized_lesion_necessity(po, pr)
            assert 0.0 <= n <= 1.0, f"n={n}, po={po}, pr={pr}"


# ---------------------------------------------------------------------------
# lesion sufficiency
# ---------------------------------------------------------------------------

class TestLesionSufficiency:
    def test_perfect_sufficiency(self):
        s = lesion_sufficiency(0.8, 0.8)
        assert s == pytest.approx(1.0)

    def test_small_difference(self):
        s = lesion_sufficiency(0.8, 0.75)
        assert 0.9 < s < 1.0

    def test_large_difference(self):
        s = lesion_sufficiency(0.8, 0.2)
        assert s == pytest.approx(0.4)

    def test_range(self):
        for _ in range(100):
            po = np.random.uniform(0, 1)
            ps = np.random.uniform(0, 1)
            s = lesion_sufficiency(po, ps)
            assert 0.0 <= s <= 1.0, f"s={s}"


# ---------------------------------------------------------------------------
# background invariance
# ---------------------------------------------------------------------------

class TestBackgroundInvariance:
    def test_perfect_invariance(self):
        r = background_invariance(0.8, [0.8, 0.8, 0.8])
        assert r["invariant"] == pytest.approx(1.0)

    def test_partial_difference(self):
        r = background_invariance(0.8, [0.6, 0.7, 0.5])
        assert 0.5 < r["invariant"] < 1.0

    def test_empty_list(self):
        r = background_invariance(0.8, [])
        assert np.isnan(r["invariant"])
        assert r["n_donors"] == 0

    def test_n_donors_counted(self):
        r = background_invariance(0.8, [0.7, 0.75])
        assert r["n_donors"] == 2


# ---------------------------------------------------------------------------
# prediction flip rate
# ---------------------------------------------------------------------------

class TestPredictionFlipRate:
    def test_no_flips(self):
        r = prediction_flip_rate(1, [1, 1, 1])
        assert r["flip_rate"] == 0.0
        assert r["n_flips"] == 0

    def test_all_flips(self):
        r = prediction_flip_rate(1, [0, 0, 0])
        assert r["flip_rate"] == 1.0
        assert r["n_flips"] == 3

    def test_empty(self):
        r = prediction_flip_rate(1, [])
        assert np.isnan(r["flip_rate"])


# ---------------------------------------------------------------------------
# donor stratified invariance
# ---------------------------------------------------------------------------

class TestDonorStratifiedInvariance:
    def test_separate_groups(self):
        r = donor_stratified_invariance(
            0.8,
            same_class_swaps=[0.8, 0.79],
            opposite_class_swaps=[0.5, 0.4],
        )
        assert r["same_class"]["invariant"] > r["opposite_class"]["invariant"]
        assert r["same_class"]["n_donors"] == 2
        assert r["opposite_class"]["n_donors"] == 2


# ---------------------------------------------------------------------------
# lesion vs sham difference
# ---------------------------------------------------------------------------

class TestLesionVsShamDifference:
    def test_lesion_more_effect(self):
        r = lesion_vs_sham_difference(0.3, 0.7)
        assert r["lesion_conf_drop"] == 0.3
        assert r["sham_conf_drop"] == 0.7
        assert r["difference"] == pytest.approx(-0.4)


# ---------------------------------------------------------------------------
# per-sample causal metrics
# ---------------------------------------------------------------------------

class TestPerSampleCausalMetrics:
    def test_full_output_structure(self, p_orig, p_suff, p_removed_t, p_removed_ns, p_sham):
        result = compute_per_sample_causal_metrics(
            p_original=p_orig,
            p_sufficient=p_suff,
            p_removed_telea=p_removed_t,
            p_removed_navier=p_removed_ns,
            p_swaps_same=[np.array([0.25, 0.75]), np.array([0.28, 0.72])],
            p_swaps_opposite=[np.array([0.6, 0.4])],
            p_sham_removed=p_sham,
            true_class=1,
        )
        assert result["predicted_class"] == 1
        assert result["target_class"] == 1
        assert result["is_correct"] is True

    def test_predicted_class_metrics_present(self, p_orig, p_suff, p_removed_t, p_removed_ns, p_sham):
        result = compute_per_sample_causal_metrics(
            p_original=p_orig,
            p_sufficient=p_suff,
            p_removed_telea=p_removed_t,
            p_removed_navier=p_removed_ns,
            p_swaps_same=[np.array([0.25, 0.75])],
            p_swaps_opposite=[],
            p_sham_removed=p_sham,
            true_class=1,
        )
        assert np.isfinite(result["predicted_raw_necessity_telea"])
        assert np.isfinite(result["predicted_norm_necessity_telea"])
        assert 0.0 <= result["predicted_norm_necessity_telea"] <= 1.0
        assert 0.0 <= result["predicted_sufficiency"] <= 1.0

    def test_incorrect_prediction_true_metrics_nan(self, p_orig, p_suff, p_removed_t, p_removed_ns, p_sham):
        result = compute_per_sample_causal_metrics(
            p_original=p_orig,
            p_sufficient=p_suff,
            p_removed_telea=p_removed_t,
            p_removed_navier=p_removed_ns,
            p_swaps_same=[],
            p_swaps_opposite=[],
            p_sham_removed=None,
            true_class=0,
        )
        assert result["is_correct"] is False
        assert np.isnan(result["true_norm_necessity_telea"])
        assert np.isnan(result["true_sufficiency"])

    def test_no_sham_handled(self, p_orig, p_suff, p_removed_t, p_removed_ns):
        result = compute_per_sample_causal_metrics(
            p_original=p_orig,
            p_sufficient=p_suff,
            p_removed_telea=p_removed_t,
            p_removed_navier=p_removed_ns,
            p_swaps_same=[],
            p_swaps_opposite=[],
            p_sham_removed=None,
            true_class=1,
        )
        assert "predicted_lesion_vs_sham_diff_telea" not in result

    def test_raw_necessity_direction(self):
        """Lesion removal should reduce confidence in the target class."""
        p_orig = np.array([0.1, 0.9])
        p_removed = np.array([0.6, 0.4])
        n_raw = raw_lesion_necessity(0.9, 0.4)
        assert n_raw > 0.0

    def test_sufficiency_high_when_similar(self):
        s = lesion_sufficiency(0.9, 0.85)
        assert s > 0.9


# ---------------------------------------------------------------------------
# composite score tests
# ---------------------------------------------------------------------------

class TestCompositeScores:
    def test_harmonic_perfect(self):
        s = compute_causalmask_harmonic(1.0, 1.0, 1.0)
        assert 0.9 < s <= 1.0

    def test_harmonic_zero_one(self):
        s = compute_causalmask_harmonic(0.0, 0.0, 0.0)
        assert 0.0 <= s < 0.1

    def test_harmonic_uneven(self):
        s_perfect = compute_causalmask_harmonic(1.0, 1.0, 1.0)
        s_uneven = compute_causalmask_harmonic(1.0, 0.1, 0.2)
        assert s_uneven < s_perfect

    def test_harmonic_penalizes_zero(self):
        s = compute_causalmask_harmonic(1.0, 0.0, 1.0)
        assert s < 0.1

    def test_arithmetic_perfect(self):
        s = compute_causalmask_arithmetic(1.0, 1.0, 1.0)
        assert s == pytest.approx(1.0)

    def test_geometric_perfect(self):
        s = compute_causalmask_geometric(1.0, 1.0, 1.0)
        assert s == pytest.approx(1.0)

    def test_harmonic_le_arithmetic(self):
        """Harmonic mean is always <= arithmetic mean."""
        for _ in range(50):
            n = max(0.01, np.random.uniform(0, 1))
            s = max(0.01, np.random.uniform(0, 1))
            b = max(0.01, np.random.uniform(0, 1))
            harm = compute_causalmask_harmonic(n, s, b)
            arith = compute_causalmask_arithmetic(n, s, b)
            assert harm <= arith + 1e-10, f"h={harm}, a={arith}"

    def test_all_aggregations_returns_three(self):
        r = compute_all_aggregations(0.8, 0.6, 0.7)
        assert "harmonic" in r
        assert "arithmetic" in r
        assert "geometric" in r

    def test_nan_propagation(self):
        s_nan = compute_causalmask_harmonic(float("nan"), 0.5, 0.5)
        assert np.isnan(s_nan)


# ---------------------------------------------------------------------------
# harmonic / arithmetic / geometric mean helpers
# ---------------------------------------------------------------------------

class TestMeanFunctions:
    def test_harmonic_mean(self):
        assert harmonic_mean([1.0, 2.0, 3.0]) == pytest.approx(3.0 / (1.0 + 0.5 + 1.0/3.0))

    def test_harmonic_mean_with_nan(self):
        v = harmonic_mean([1.0, float("nan"), 2.0])
        assert not np.isnan(v)

    def test_arithmetic_mean(self):
        assert arithmetic_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_geometric_mean(self):
        assert geometric_mean([1.0, 1.0, 1.0]) == pytest.approx(1.0)

    def test_geometric_mean_with_zero(self):
        v = geometric_mean([0.0, 0.5, 0.5])
        assert v == 0.0


# ---------------------------------------------------------------------------
# aggregation sensitivity
# ---------------------------------------------------------------------------

class TestAggregationSensitivity:
    def test_output_structure(self):
        rng = np.random.default_rng(42)
        N = rng.uniform(0.3, 0.9, 100)
        S = rng.uniform(0.3, 0.9, 100)
        B = rng.uniform(0.3, 0.9, 100)
        result = compute_aggregation_sensitivity(N, S, B)
        assert result["n_valid"] == 100
        assert "harmonic" in result
        assert "arithmetic" in result
        assert "geometric" in result
        assert "spearman_correlations" in result

    def test_high_correlation_expected(self):
        rng = np.random.default_rng(42)
        N = rng.uniform(0.3, 0.9, 200)
        S = rng.uniform(0.3, 0.9, 200)
        B = rng.uniform(0.3, 0.9, 200)
        result = compute_aggregation_sensitivity(N, S, B)
        corr = result["spearman_correlations"]["harmonic_vs_arithmetic"]
        assert corr > 0.5

    def test_empty_handled(self):
        result = compute_aggregation_sensitivity(
            np.array([]), np.array([]), np.array([])
        )
        assert result["n_valid"] == 0
