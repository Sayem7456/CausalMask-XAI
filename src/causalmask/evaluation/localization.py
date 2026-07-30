"""XAI localization metrics.

Measures how much attribution mass concentrates within the lesion
and lesion-plus-margin regions.

Metrics:
  - attribution_mass_inside_lesion
  - attribution_mass_inside_lesion_plus_margin
  - pointing_game_accuracy
  - soft_dice
  - saliency_iou (using fixed declared threshold rule)

All metrics accept non-negative normalized attribution maps [H, W]
and binary masks [H, W]. Threshold rule is preregistered and never
chosen by looking at test performance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_EPS = 1e-8

DEFAULT_IOU_THRESHOLD = 0.5


def attribution_mass_inside_mask(
    attribution: NDArray[np.float64],
    mask: NDArray[np.float64],
) -> float:
    """Fraction of total attribution mass inside the mask.

    L = Σ(A ⊙ M) / (ΣA + ε)

    Args:
        attribution: Non-negative normalized [H, W] attribution.
        mask: Binary [H, W] mask (0 or 1).

    Returns:
        Float in [0, 1]. Higher = more attribution inside mask.
    """
    attr = np.asarray(attribution, dtype=np.float64)
    m = np.asarray(mask, dtype=np.float64)
    if attr.shape != m.shape:
        raise ValueError(
            f"Shape mismatch: attribution {attr.shape}, mask {m.shape}"
        )
    total = attr.sum() + _EPS
    inside = (attr * m).sum()
    return float(inside / total)


def attribution_mass_inside_lesion(
    attribution: NDArray[np.float64],
    lesion_mask: NDArray[np.float64],
) -> float:
    """Attribution mass strictly inside the lesion (not margin)."""
    return attribution_mass_inside_mask(attribution, lesion_mask)


def attribution_mass_inside_lesion_plus_margin(
    attribution: NDArray[np.float64],
    lesion_plus_margin_mask: NDArray[np.float64],
) -> float:
    """Attribution mass inside lesion plus dilated margin."""
    return attribution_mass_inside_mask(attribution, lesion_plus_margin_mask)


def pointing_game_accuracy(
    attribution: NDArray[np.float64],
    mask: NDArray[np.float64],
) -> float:
    """Pointing-game accuracy: is the maximum attribution point inside the mask?

    Returns 1.0 if argmax of attribution falls on a mask pixel, else 0.0.
    """
    attr = np.asarray(attribution, dtype=np.float64)
    m = np.asarray(mask, dtype=np.float64)
    if attr.shape != m.shape:
        raise ValueError(f"Shape mismatch: attribution {attr.shape}, mask {m.shape}")
    max_idx = np.unravel_index(int(np.argmax(attr)), attr.shape)
    return 1.0 if m[max_idx] > 0.5 else 0.0


def soft_dice(
    attribution: NDArray[np.float64],
    mask: NDArray[np.float64],
) -> float:
    """Soft (continuous) Dice coefficient between attribution and mask.

    sDice = 2 * Σ(A ⊙ M) / (ΣA + ΣM + ε)

    Both A and M should be non-negative and of similar magnitude.
    For binary mask and normalized attribution [0,1], this is
    equivalent to an overlap measure.

    Args:
        attribution: Non-negative [H, W] normalized attribution.
        mask: Binary [H, W] mask.

    Returns:
        Float in [0, 1].
    """
    attr = np.asarray(attribution, dtype=np.float64)
    m = np.asarray(mask, dtype=np.float64)
    if attr.shape != m.shape:
        raise ValueError(f"Shape mismatch: attribution {attr.shape}, mask {m.shape}")
    intersection = (attr * m).sum()
    denom = attr.sum() + m.sum() + _EPS
    return float(2.0 * intersection / denom)


def saliency_iou(
    attribution: NDArray[np.float64],
    mask: NDArray[np.float64],
    threshold: float = DEFAULT_IOU_THRESHOLD,
) -> float:
    """IoU between thresholded saliency and binary mask.

    Uses a FIXED declared threshold. The threshold must NOT be chosen
    by looking at test performance.

    For normalized [0,1] attributions, pixels >= threshold form the
    saliency region. IoU = intersection / union.

    Args:
        attribution: Non-negative normalized [H, W] attribution.
        mask: Binary [H, W] ground-truth mask.
        threshold: Fixed threshold in [0, 1]. Default 0.5.

    Returns:
        IoU in [0, 1].
    """
    attr = np.asarray(attribution, dtype=np.float64)
    m = np.asarray(mask, dtype=np.float64)
    if attr.shape != m.shape:
        raise ValueError(f"Shape mismatch: attribution {attr.shape}, mask {m.shape}")
    saliency = (attr >= threshold).astype(np.float64)
    intersection = (saliency * m).sum()
    union = ((saliency + m) > 0).sum() + _EPS
    return float(intersection / union)


@dataclass
class LocalizationResult:
    sample_id: str = ""
    mass_lesion: float = float("nan")
    mass_lesion_margin: float = float("nan")
    pointing_game: float = float("nan")
    soft_dice: float = float("nan")
    iou: float = float("nan")
    iou_threshold: float = DEFAULT_IOU_THRESHOLD
    failure_flag: str = ""


def compute_localization_metrics(
    attribution: NDArray[np.float64],
    lesion_mask: NDArray[np.float64],
    lesion_plus_margin_mask: NDArray[np.float64],
    sample_id: str = "",
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> LocalizationResult:
    """Compute all localization metrics for one sample.

    Args:
        attribution: Non-negative normalized [H, W] attribution.
        lesion_mask: Binary [H, W] lesion mask.
        lesion_plus_margin_mask: Binary [H, W] lesion-plus-margin mask.
        sample_id: Sample identifier for provenance.
        iou_threshold: Fixed threshold for IoU computation.

    Returns:
        LocalizationResult with all metrics.
    """
    result = LocalizationResult(
        sample_id=sample_id,
        iou_threshold=iou_threshold,
    )

    attr = np.asarray(attribution, dtype=np.float64)
    lesion = np.asarray(lesion_mask, dtype=np.float64)
    margin_mask = np.asarray(lesion_plus_margin_mask, dtype=np.float64)

    if len(attr.shape) != 2:
        result.failure_flag = f"attribution_not_2d:{str(attr.shape)}"
        return result

    if not np.isfinite(attr).all():
        n_bad = (~np.isfinite(attr)).sum()
        result.failure_flag = f"non_finite_values:{n_bad}"
        logger.warning(f"{sample_id}: {result.failure_flag}")
        return result

    if attr.max() == 0.0:
        result.failure_flag = "all_zero_attribution"
        return result

    try:
        result.mass_lesion = attribution_mass_inside_lesion(attr, lesion)
    except Exception as e:
        logger.warning(f"mass_lesion failed for {sample_id}: {e}")

    try:
        result.mass_lesion_margin = attribution_mass_inside_lesion_plus_margin(attr, margin_mask)
    except Exception as e:
        logger.warning(f"mass_lesion_margin failed for {sample_id}: {e}")

    try:
        result.pointing_game = pointing_game_accuracy(attr, margin_mask)
    except Exception as e:
        logger.warning(f"pointing_game failed for {sample_id}: {e}")

    try:
        result.soft_dice = soft_dice(attr, margin_mask)
    except Exception as e:
        logger.warning(f"soft_dice failed for {sample_id}: {e}")

    try:
        result.iou = saliency_iou(attr, margin_mask, threshold=iou_threshold)
    except Exception as e:
        logger.warning(f"iou failed for {sample_id}: {e}")

    return result


def compute_localization_batch(
    attributions: NDArray[np.float64],
    lesion_masks: NDArray[np.float64],
    margin_masks: NDArray[np.float64],
    sample_ids: Optional[list[str]] = None,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict[str, Any]:
    """Compute localization metrics for a batch of samples.

    Args:
        attributions: [N, H, W] normalized attributions.
        lesion_masks: [N, H, W] binary lesion masks.
        margin_masks: [N, H, W] binary lesion-plus-margin masks.
        sample_ids: Optional list of sample IDs.
        iou_threshold: Fixed IoU threshold.

    Returns:
        Dict with per-metric arrays and summary statistics.
    """
    n = len(attributions)
    if sample_ids is None:
        sample_ids = [f"sample_{i:04d}" for i in range(n)]

    mass_lesion = []
    mass_margin = []
    pointing = []
    s_dice = []
    iou_vals = []
    failures = []

    for i in range(n):
        res = compute_localization_metrics(
            attributions[i],
            lesion_masks[i],
            margin_masks[i],
            sample_id=sample_ids[i],
            iou_threshold=iou_threshold,
        )
        mass_lesion.append(res.mass_lesion)
        mass_margin.append(res.mass_lesion_margin)
        pointing.append(res.pointing_game)
        s_dice.append(res.soft_dice)
        iou_vals.append(res.iou)
        if res.failure_flag:
            failures.append({"sample_id": sample_ids[i], "failure": res.failure_flag})

    mass_lesion_arr = np.array(mass_lesion, dtype=np.float64)
    mass_margin_arr = np.array(mass_margin, dtype=np.float64)
    pointing_arr = np.array(pointing, dtype=np.float64)
    s_dice_arr = np.array(s_dice, dtype=np.float64)
    iou_arr = np.array(iou_vals, dtype=np.float64)

    valid = np.isfinite(mass_lesion_arr)

    return {
        "n_samples": n,
        "n_valid": int(valid.sum()),
        "n_failed": len(failures),
        "failures": failures,
        "iou_threshold": iou_threshold,
        "mass_lesion": mass_lesion_arr.tolist(),
        "mass_lesion_mean": float(np.nanmean(mass_lesion_arr)),
        "mass_lesion_median": float(np.nanmedian(mass_lesion_arr)),
        "mass_lesion_margin": mass_margin_arr.tolist(),
        "mass_lesion_margin_mean": float(np.nanmean(mass_margin_arr)),
        "mass_lesion_margin_median": float(np.nanmedian(mass_margin_arr)),
        "pointing_game_accuracy": pointing_arr.tolist(),
        "pointing_game_mean": float(np.nanmean(pointing_arr)),
        "soft_dice": s_dice_arr.tolist(),
        "soft_dice_mean": float(np.nanmean(s_dice_arr)),
        "soft_dice_median": float(np.nanmedian(s_dice_arr)),
        "iou": iou_arr.tolist(),
        "iou_mean": float(np.nanmean(iou_arr)),
        "iou_median": float(np.nanmedian(iou_arr)),
    }
