"""CausalMask composite score and aggregation sensitivity.

The composite CausalMask score combines necessity (N), sufficiency (S),
and background invariance (B) using harmonic, arithmetic, or geometric mean.
The harmonic mean is the default because a model cannot obtain a strong
score by performing well on only one property.

Aggregation weights are preregistered as equal (no weighting observed).
Sensitivity analysis compares harmonic, arithmetic, and geometric aggregation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_EPS = 1e-8


def harmonic_mean(values: list[float]) -> float:
    """Harmonic mean of values, ignoring NaN.

    Returns NaN if all values are NaN or zero.
    """
    valid = [v for v in values if np.isfinite(v) and not np.isnan(v)]
    if not valid:
        return float("nan")
    if any(v <= 0 for v in valid):
        return 0.0
    return float(len(valid) / sum(1.0 / v for v in valid))


def arithmetic_mean(values: list[float]) -> float:
    """Arithmetic mean of values, ignoring NaN."""
    valid = [v for v in values if np.isfinite(v) and not np.isnan(v)]
    if not valid:
        return float("nan")
    return float(np.mean(valid))


def geometric_mean(values: list[float]) -> float:
    """Geometric mean of values, ignoring NaN.

    Returns NaN if all values are NaN or zero.
    """
    valid = [v for v in values if np.isfinite(v) and not np.isnan(v)]
    if not valid:
        return float("nan")
    if any(v <= 0 for v in valid):
        return 0.0
    return float(np.exp(np.mean(np.log(valid))))


def compute_causalmask_harmonic(
    necessity: float,
    sufficiency: float,
    background_invariance: float,
    epsilon: float = _EPS,
) -> float:
    """Equal-weight harmonic mean CausalMask score (3 components).

    CausalMask_3 = 3 / (1/(N+ε) + 1/(S+ε) + 1/(B+ε))
    """
    values = [
        max(float(necessity), epsilon),
        max(float(sufficiency), epsilon),
        max(float(background_invariance), epsilon),
    ]
    if any(np.isnan(v) for v in values):
        return float("nan")
    return float(3.0 / sum(1.0 / v for v in values))


def compute_causalmask_arithmetic(
    necessity: float,
    sufficiency: float,
    background_invariance: float,
) -> float:
    """Equal-weight arithmetic mean CausalMask score."""
    vals = [float(necessity), float(sufficiency), float(background_invariance)]
    valid = [v for v in vals if np.isfinite(v) and not np.isnan(v)]
    if not valid:
        return float("nan")
    return float(np.mean(valid))


def compute_causalmask_geometric(
    necessity: float,
    sufficiency: float,
    background_invariance: float,
    epsilon: float = _EPS,
) -> float:
    """Equal-weight geometric mean CausalMask score."""
    vals = [
        max(float(necessity), epsilon),
        max(float(sufficiency), epsilon),
        max(float(background_invariance), epsilon),
    ]
    if any(np.isnan(v) for v in vals):
        return float("nan")
    return float(np.exp(np.mean(np.log(vals))))


def compute_all_aggregations(
    necessity: float,
    sufficiency: float,
    background_invariance: float,
) -> dict[str, float]:
    """Compute harmonic, arithmetic, and geometric CausalMask scores.

    Returns dict with keys: harmonic, arithmetic, geometric.
    """
    return {
        "harmonic": compute_causalmask_harmonic(necessity, sufficiency, background_invariance),
        "arithmetic": compute_causalmask_arithmetic(necessity, sufficiency, background_invariance),
        "geometric": compute_causalmask_geometric(necessity, sufficiency, background_invariance),
    }


def compute_aggregation_sensitivity(
    n_values: np.ndarray,
    s_values: np.ndarray,
    b_values: np.ndarray,
) -> dict[str, Any]:
    """Aggregation sensitivity analysis across samples.

    Computes per-sample composite scores using all three aggregation
    methods, then reports mean and bootstrap-percentile summary.

    Args:
        n_values: Normalized necessity values per sample.
        s_values: Sufficiency values per sample.
        b_values: Background invariance values per sample.

    Returns:
        Dict with harmonic, arithmetic, geometric summaries and
        Spearman rank correlations between aggregation methods.
    """
    n = np.asarray(n_values, dtype=np.float64)
    s = np.asarray(s_values, dtype=np.float64)
    b = np.asarray(b_values, dtype=np.float64)

    valid = np.isfinite(n) & np.isfinite(s) & np.isfinite(b)
    n = n[valid]
    s = s[valid]
    b = b[valid]

    if len(n) == 0:
        return {"n_valid": 0, "note": "no valid samples"}

    harmonic_scores = np.array([
        compute_causalmask_harmonic(float(n[i]), float(s[i]), float(b[i]))
        for i in range(len(n))
    ])
    arithmetic_scores = np.array([
        compute_causalmask_arithmetic(float(n[i]), float(s[i]), float(b[i]))
        for i in range(len(n))
    ])
    geometric_scores = np.array([
        compute_causalmask_geometric(float(n[i]), float(s[i]), float(b[i]))
        for i in range(len(n))
    ])

    valid_all = (np.isfinite(harmonic_scores)
                 & np.isfinite(arithmetic_scores)
                 & np.isfinite(geometric_scores))

    from scipy.stats import spearmanr as _spearmanr

    corr_ha = float(_spearmanr(harmonic_scores[valid_all], arithmetic_scores[valid_all])[0])  # type: ignore[arg-type]
    corr_hg = float(_spearmanr(harmonic_scores[valid_all], geometric_scores[valid_all])[0])  # type: ignore[arg-type]
    corr_ag = float(_spearmanr(arithmetic_scores[valid_all], geometric_scores[valid_all])[0])  # type: ignore[arg-type]

    return {
        "n_valid": int(len(n)),
        "harmonic": {
            "mean": float(np.mean(harmonic_scores[valid_all])),
            "median": float(np.median(harmonic_scores[valid_all])),
            "std": float(np.std(harmonic_scores[valid_all])),
        },
        "arithmetic": {
            "mean": float(np.mean(arithmetic_scores[valid_all])),
            "median": float(np.median(arithmetic_scores[valid_all])),
            "std": float(np.std(arithmetic_scores[valid_all])),
        },
        "geometric": {
            "mean": float(np.mean(geometric_scores[valid_all])),
            "median": float(np.median(geometric_scores[valid_all])),
            "std": float(np.std(geometric_scores[valid_all])),
        },
        "spearman_correlations": {
            "harmonic_vs_arithmetic": float(corr_ha),
            "harmonic_vs_geometric": float(corr_hg),
            "arithmetic_vs_geometric": float(corr_ag),
        },
    }
