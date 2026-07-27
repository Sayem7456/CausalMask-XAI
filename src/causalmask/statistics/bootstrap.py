"""Group-aware bootstrap confidence intervals for paired medical-imaging metrics.

Resamples at the group (patient/duplicate-cluster) level to avoid
underestimating uncertainty from within-group correlation.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def group_aware_bootstrap_ci(
    values: np.ndarray,
    group_ids: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 2_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute group-aware bootstrap confidence interval.

    Resamples entire groups with replacement, then collects all
    observations from selected groups. Avoids within-group
    pseudoreplication.

    Args:
        values: Metric values, shape (n_samples,).
        group_ids: Group identifier per sample, shape (n_samples,).
        statistic: Statistic function (e.g. np.mean, np.median).
        n_bootstrap: Number of bootstrap replicates.
        alpha: Significance level (0.05 -> 95% CI).
        seed: Random seed.

    Returns:
        Dict with keys: point_estimate, ci_lower, ci_upper, n_valid, n_groups,
        n_bootstrap, alpha, seed.
    """
    values = np.asarray(values, dtype=np.float64)
    group_ids = np.asarray(group_ids)

    valid = np.isfinite(values)
    values = values[valid]
    group_ids = group_ids[valid]

    if len(values) == 0:
        return {
            "point_estimate": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "n_valid": 0,
            "n_groups": 0,
            "n_bootstrap": n_bootstrap,
            "alpha": alpha,
            "seed": seed,
            "bootstrap_distribution": [],
            "denominator_note": "no valid samples",
        }

    unique_groups = np.unique(group_ids)
    groups = [values[group_ids == g] for g in unique_groups]

    rng = np.random.default_rng(seed)
    replicates = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        sampled_idx = rng.integers(0, len(groups), size=len(groups))
        sampled_values = np.concatenate([groups[j] for j in sampled_idx])
        replicates[i] = statistic(sampled_values)

    point = statistic(values)
    lower = np.percentile(replicates, 100 * alpha / 2)
    upper = np.percentile(replicates, 100 * (1 - alpha / 2))

    return {
        "point_estimate": float(point),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "n_valid": int(len(values)),
        "n_groups": int(len(unique_groups)),
        "n_bootstrap": n_bootstrap,
        "alpha": float(alpha),
        "seed": seed,
        "bootstrap_distribution": [float(v) for v in replicates],
        "denominator_note": None,
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int = 2_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap CI for the mean (no grouping — use with caution).

    Use only when group information is unavailable, and state that
    it may underestimate uncertainty.

    Args:
        values: Metric values, shape (n_samples,).
        n_bootstrap: Number of bootstrap replicates.
        alpha: Significance level.
        seed: Random seed.

    Returns:
        Dict with ci_lower, ci_upper, point_estimate, etc.
    """
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    values = values[valid]

    if len(values) == 0:
        return {
            "point_estimate": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "n_valid": 0,
            "n_bootstrap": n_bootstrap,
            "alpha": float(alpha),
            "seed": seed,
            "denominator_note": "no valid samples",
        }

    rng = np.random.default_rng(seed)
    n = len(values)
    replicates = np.array([
        values[rng.integers(0, n, size=n)].mean()
        for _ in range(n_bootstrap)
    ])

    point = values.mean()
    lower = np.percentile(replicates, 100 * alpha / 2)
    upper = np.percentile(replicates, 100 * (1 - alpha / 2))

    return {
        "point_estimate": float(point),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "n_valid": int(n),
        "n_bootstrap": n_bootstrap,
        "alpha": float(alpha),
        "seed": seed,
        "bootstrap_distribution": [float(v) for v in replicates],
        "denominator_note": "image-level bootstrap (may underestimate uncertainty)",
    }


def paired_bootstrap_diff(
    values_a: np.ndarray,
    values_b: np.ndarray,
    group_ids: Optional[np.ndarray] = None,
    n_bootstrap: int = 2_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap CI for the mean paired difference (A - B).

    Args:
        values_a: First set of metric values.
        values_b: Second set of metric values (paired with A).
        group_ids: Optional group IDs for group-aware resampling.
        n_bootstrap: Replicates.
        alpha: Significance level.
        seed: Random seed.

    Returns:
        Dict with point_estimate, ci_lower, ci_upper, p_approx.
    """
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)

    paired = (np.isfinite(values_a)) & (np.isfinite(values_b))
    values_a = values_a[paired]
    values_b = values_b[paired]
    diffs = values_a - values_b

    if group_ids is not None:
        group_ids = np.asarray(group_ids)[paired]
        ci = group_aware_bootstrap_ci(diffs, group_ids, np.mean,
                                      n_bootstrap, alpha, seed)
    else:
        ci = bootstrap_mean_ci(diffs, n_bootstrap, alpha, seed)

    ci["paired_n"] = int(len(diffs))
    ci["mean_diff"] = float(diffs.mean())
    ci["median_diff"] = float(np.median(diffs))

    if ci.get("bootstrap_distribution"):
        dist = np.array(ci["bootstrap_distribution"])
        ci["p_approx"] = float(2 * min(
            (dist <= 0).mean(),
            (dist >= 0).mean(),
        ))

    return ci
