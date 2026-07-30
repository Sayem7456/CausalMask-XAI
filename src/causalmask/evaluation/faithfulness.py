"""Causal faithfulness component metrics.

Computes lesion necessity (raw and normalized), lesion sufficiency,
background invariance, prediction-flip rate, donor-stratified invariance,
lesion-intervention versus sham-control difference, and
insertion/deletion AUC for XAI faithfulness.

All functions accept predicted-class or true-class target definitions.
The caller is responsible for sending the correct confidence values.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_EPS = 1e-8


def _auc_trapezoid(y: np.ndarray, dx: float = 1.0) -> float:
    """Trapezoidal AUC of y sampled at uniform intervals dx."""
    y = np.asarray(y, dtype=np.float64)
    return float((y[1:] + y[:-1]).sum() * dx / 2.0)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def raw_lesion_necessity(
    p_original: float,
    p_removed: float,
) -> float:
    """N_raw = p_y(x) - p_y(x_removed).

    Positive: confidence decreased after removal (lesion was necessary).
    Negative: confidence increased (model attends elsewhere).
    Range approximately [-1, 1].
    """
    return float(p_original) - float(p_removed)


def normalized_lesion_necessity(
    p_original: float,
    p_removed: float,
) -> float:
    """N = clip((p_y(x) - p_y(x_removed)) / max(p_y(x), epsilon), 0, 1).

    Range [0, 1]. Higher = lesion more necessary.
    """
    diff = float(p_original) - float(p_removed)
    denom = max(float(p_original), _EPS)
    return _clip01(diff / denom)


def lesion_sufficiency(
    p_original: float,
    p_sufficient: float,
) -> float:
    """S = clip(1 - |p_y(x) - p_y(x_sufficient)|, 0, 1).

    Range [0, 1]. Higher = lesion+margin alone preserves prediction.
    """
    diff = abs(float(p_original) - float(p_sufficient))
    return _clip01(1.0 - diff)


def background_invariance(
    p_original: float,
    p_swaps: list[float],
) -> dict[str, Any]:
    """B = clip(1 - mean(|p_y(x) - p_y(x_swap_j)|), 0, 1).

    Args:
        p_original: Original confidence in target class.
        p_swaps: Confidences from each swap donor.

    Returns:
        Dict with keys: invariant, mean_abs_diff, n_donors, p_swaps.
    """
    p_original = float(p_original)
    if not p_swaps:
        return {
            "invariant": float("nan"),
            "mean_abs_diff": float("nan"),
            "n_donors": 0,
            "p_swaps": [],
        }
    diffs = [abs(p_original - float(pj)) for pj in p_swaps]
    mean_diff = float(np.mean(diffs))
    return {
        "invariant": _clip01(1.0 - mean_diff),
        "mean_abs_diff": mean_diff,
        "n_donors": len(p_swaps),
        "p_swaps": [float(pj) for pj in p_swaps],
    }


def prediction_flip_rate(
    original_predicted_class: int,
    counterfactual_predicted_classes: list[int],
) -> dict[str, Any]:
    """Fraction of counterfactuals where predicted class differs from original.

    Args:
        original_predicted_class: argmax of original prediction.
        counterfactual_predicted_classes: List of argmax predictions for
            each counterfactual variant.

    Returns:
        Dict with flip_rate, n_total, n_flips.
    """
    if not counterfactual_predicted_classes:
        return {"flip_rate": float("nan"), "n_total": 0, "n_flips": 0}
    n = len(counterfactual_predicted_classes)
    flips = sum(1 for c in counterfactual_predicted_classes if c != original_predicted_class)
    return {"flip_rate": flips / n, "n_total": n, "n_flips": flips}


def donor_stratified_invariance(
    p_original: float,
    same_class_swaps: list[float],
    opposite_class_swaps: list[float],
) -> dict[str, Any]:
    """Invariance stratified by donor class.

    Returns:
        Dict with same_class, opposite_class sub-dicts.
    """
    same_result = background_invariance(p_original, same_class_swaps)
    opp_result = background_invariance(p_original, opposite_class_swaps)
    return {
        "same_class": same_result,
        "opposite_class": opp_result,
    }


def lesion_vs_sham_difference(
    p_removed: float,
    p_sham_removed: float,
) -> dict[str, Any]:
    """Difference between lesion removal effect and sham removal effect.

    Returns:
        Dict with lesion_conf_drop, sham_conf_drop, difference.
        Positive difference: lesion removal has larger effect than sham
        (lesion is more informative than a random region of same area).
    """
    lesion_drop = float(p_removed)
    sham_drop = float(p_sham_removed)
    return {
        "lesion_conf_drop": lesion_drop,
        "sham_conf_drop": sham_drop,
        "difference": lesion_drop - sham_drop,
    }


def ensure_confidence_for_target(
    probabilities: np.ndarray,
    target_class: int,
) -> float:
    """Extract confidence for a specific target class from probability array.

    Args:
        probabilities: Array of class probabilities [n_classes].
        target_class: Index of target class.

    Returns:
        Confidence value for the target class.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim == 0:
        return float(probs)
    idx = int(target_class)
    if idx < 0 or idx >= len(probs):
        return float("nan")
    return float(probs[idx])


def compute_per_sample_causal_metrics(
    p_original: np.ndarray,
    p_sufficient: np.ndarray,
    p_removed_telea: np.ndarray,
    p_removed_navier: np.ndarray,
    p_swaps_same: list[np.ndarray],
    p_swaps_opposite: list[np.ndarray],
    p_sham_removed: np.ndarray | None = None,
    true_class: int | None = None,
    target_class: int | None = None,
) -> dict[str, Any]:
    """Compute all causal metrics for one sample.

    Args:
        p_original: Original class probabilities [n_classes].
        p_sufficient: Probabilities for sufficient image.
        p_removed_telea: Probabilities for Telea-removed image.
        p_removed_navier: Probabilities for NS-removed image.
        p_swaps_same: List of probability arrays from same-class swaps.
        p_swaps_opposite: List of probability arrays from opposite-class swaps.
        p_sham_removed: Probabilities for sham-removed image.
        true_class: Ground-truth class index (0 or 1).
        target_class: Target class index for decision-faithfulness.
            If None, uses np.argmax(p_original).

    Returns:
        Dict with all component metrics for both target definitions.
    """
    if target_class is None:
        target_class = int(np.argmax(p_original))

    result: dict[str, Any] = {}
    result["predicted_class"] = int(np.argmax(p_original))
    result["target_class"] = target_class
    result["true_class"] = true_class
    result["is_correct"] = (true_class is not None and result["predicted_class"] == true_class)

    # --- predicted-class versions (primary decision-faithfulness target) ---
    pc_original = ensure_confidence_for_target(p_original, target_class)
    pc_sufficient = ensure_confidence_for_target(p_sufficient, target_class)
    pc_removed_t = ensure_confidence_for_target(p_removed_telea, target_class)
    pc_removed_ns = ensure_confidence_for_target(p_removed_navier, target_class)
    pc_swaps_same = [ensure_confidence_for_target(pj, target_class) for pj in p_swaps_same]
    pc_swaps_opposite = [ensure_confidence_for_target(pj, target_class) for pj in p_swaps_opposite]
    pc_sham = (ensure_confidence_for_target(p_sham_removed, target_class)
               if p_sham_removed is not None else None)

    result["predicted_raw_necessity_telea"] = raw_lesion_necessity(pc_original, pc_removed_t)
    result["predicted_raw_necessity_navier"] = raw_lesion_necessity(pc_original, pc_removed_ns)
    result["predicted_norm_necessity_telea"] = normalized_lesion_necessity(pc_original, pc_removed_t)
    result["predicted_norm_necessity_navier"] = normalized_lesion_necessity(pc_original, pc_removed_ns)
    result["predicted_sufficiency"] = lesion_sufficiency(pc_original, pc_sufficient)

    bg_inv = background_invariance(pc_original, pc_swaps_same + pc_swaps_opposite)
    result["predicted_background_invariance"] = bg_inv["invariant"]
    result["predicted_bg_n_donors"] = bg_inv["n_donors"]
    result["predicted_bg_mean_abs_diff"] = bg_inv["mean_abs_diff"]

    donor_inv = donor_stratified_invariance(pc_original, pc_swaps_same, pc_swaps_opposite)
    result["predicted_same_class_invariance"] = donor_inv["same_class"]["invariant"]
    result["predicted_opposite_class_invariance"] = donor_inv["opposite_class"]["invariant"]

    if pc_sham is not None:
        sham_diff_t = lesion_vs_sham_difference(pc_original - pc_removed_t, pc_original - pc_sham)
        result["predicted_lesion_vs_sham_diff_telea"] = sham_diff_t["difference"]

    # --- true-class versions (only for correctly classified samples) ---
    true_class_idx = true_class
    tc_original = ensure_confidence_for_target(p_original, true_class_idx) if true_class_idx is not None else None

    if true_class_idx is not None and result["is_correct"]:
        assert tc_original is not None
        tc_sufficient = ensure_confidence_for_target(p_sufficient, true_class_idx)
        tc_removed_t = ensure_confidence_for_target(p_removed_telea, true_class_idx)
        tc_removed_ns = ensure_confidence_for_target(p_removed_navier, true_class_idx)
        tc_swaps_same = [ensure_confidence_for_target(pj, true_class_idx) for pj in p_swaps_same]
        tc_swaps_opposite = [ensure_confidence_for_target(pj, true_class_idx) for pj in p_swaps_opposite]

        result["true_raw_necessity_telea"] = raw_lesion_necessity(tc_original, tc_removed_t)
        result["true_raw_necessity_navier"] = raw_lesion_necessity(tc_original, tc_removed_ns)
        result["true_norm_necessity_telea"] = normalized_lesion_necessity(tc_original, tc_removed_t)
        result["true_norm_necessity_navier"] = normalized_lesion_necessity(tc_original, tc_removed_ns)
        result["true_sufficiency"] = lesion_sufficiency(tc_original, tc_sufficient)

        bg_inv_tc = background_invariance(tc_original, tc_swaps_same + tc_swaps_opposite)
        result["true_background_invariance"] = bg_inv_tc["invariant"]
        result["true_bg_n_donors"] = bg_inv_tc["n_donors"]
        result["true_bg_mean_abs_diff"] = bg_inv_tc["mean_abs_diff"]

        donor_inv_tc = donor_stratified_invariance(tc_original, tc_swaps_same, tc_swaps_opposite)
        result["true_same_class_invariance"] = donor_inv_tc["same_class"]["invariant"]
        result["true_opposite_class_invariance"] = donor_inv_tc["opposite_class"]["invariant"]
    else:
        result["true_raw_necessity_telea"] = float("nan")
        result["true_raw_necessity_navier"] = float("nan")
        result["true_norm_necessity_telea"] = float("nan")
        result["true_norm_necessity_navier"] = float("nan")
        result["true_sufficiency"] = float("nan")
        result["true_background_invariance"] = float("nan")
        result["true_bg_n_donors"] = 0
        result["true_bg_mean_abs_diff"] = float("nan")
        result["true_same_class_invariance"] = float("nan")
        result["true_opposite_class_invariance"] = float("nan")

    return result


def insertion_auc(
    model: Any,
    image: NDArray[np.float64],
    attribution: NDArray[np.float64],
    mask: NDArray[np.float64],
    target_class: int,
    baseline: NDArray[np.float64],
    n_steps: int = 30,
    device: Any = None,
) -> dict[str, Any]:
    """Insertion AUC: add most-important pixels and measure confidence rise.

    Starting from a baseline image (e.g. blurred), incrementally insert
    pixels in order of decreasing attribution. A faithful explanation
    should produce a steep increase in the target-class confidence.

    Args:
        model: PyTorch model (callable).
        image: Original image [C, H, W] as numpy array.
        attribution: Non-negative normalized [H, W] attribution.
        mask: Binary [H, W] lesion-plus-margin mask for region-of-interest.
            If None, uses the full image. (Not used in standard insertion,
            kept for compatibility.)
        target_class: Target class index for confidence tracking.
        baseline: Baseline image [C, H, W] (e.g. blurred or zero).
        n_steps: Number of insertion steps.
        device: torch device.

    Returns:
        Dict with insertion_auc, curve, n_steps.
    """
    import torch

    if device is None:
        device = torch.device("cpu")

    attr = np.asarray(attribution, dtype=np.float64)
    img = np.asarray(image, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)

    h, w = attr.shape

    sorted_indices = np.dstack(np.unravel_index(
        np.argsort(attr.ravel())[::-1], (h, w)
    ))[0]

    total_pixels = len(sorted_indices)
    step_size = max(1, total_pixels // n_steps)
    n_steps_actual = min(n_steps, total_pixels // step_size)

    curve = np.zeros(n_steps_actual + 1)
    current = baseline.copy()

    current_t = torch.from_numpy(current).float().unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(current_t)
        probs = torch.softmax(logits, dim=1)
        curve[0] = float(probs[0, target_class].cpu().numpy())

    for step in range(1, n_steps_actual + 1):
        start_idx = (step - 1) * step_size
        end_idx = min(step * step_size, total_pixels)
        for idx in range(start_idx, end_idx):
            y, x_coord = sorted_indices[idx]
            current[:, y, x_coord] = img[:, y, x_coord]

        current_t = torch.from_numpy(current).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(current_t)
            probs = torch.softmax(logits, dim=1)
            curve[step] = float(probs[0, target_class].cpu().numpy())

    auc_val = float(_auc_trapezoid(curve, dx=1.0 / n_steps_actual))

    return {
        "insertion_auc": auc_val,
        "curve": curve.tolist(),
        "n_steps": n_steps_actual,
    }


def deletion_auc(
    model: Any,
    image: NDArray[np.float64],
    attribution: NDArray[np.float64],
    mask: NDArray[np.float64],
    target_class: int,
    baseline: NDArray[np.float64],
    n_steps: int = 30,
    device: Any = None,
) -> dict[str, Any]:
    """Deletion AUC: remove most-important pixels and measure confidence drop.

    Starting from the original image, incrementally replace pixels with
    a baseline (e.g. zero or blurred) in order of decreasing attribution.
    A faithful explanation should produce a steep decrease.

    Args:
        model: PyTorch model (callable).
        image: Original image [C, H, W] as numpy array.
        attribution: Non-negative normalized [H, W] attribution.
        mask: Binary [H, W] lesion-plus-margin mask (for ROI).
        target_class: Target class index.
        baseline: Baseline value or array to replace removed pixels.
        n_steps: Number of deletion steps.
        device: torch device.

    Returns:
        Dict with deletion_auc, curve, n_steps.
    """
    import torch

    if device is None:
        device = torch.device("cpu")

    attr = np.asarray(attribution, dtype=np.float64)
    img = np.asarray(image, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)

    h, w = attr.shape

    sorted_indices = np.dstack(np.unravel_index(
        np.argsort(attr.ravel())[::-1], (h, w)
    ))[0]

    total_pixels = len(sorted_indices)
    step_size = max(1, total_pixels // n_steps)
    n_steps_actual = min(n_steps, total_pixels // step_size)

    curve = np.zeros(n_steps_actual + 1)
    current = img.copy()

    current_t = torch.from_numpy(current).float().unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(current_t)
        probs = torch.softmax(logits, dim=1)
        curve[0] = float(probs[0, target_class].cpu().numpy())

    for step in range(1, n_steps_actual + 1):
        start_idx = (step - 1) * step_size
        end_idx = min(step * step_size, total_pixels)
        for idx in range(start_idx, end_idx):
            y, x_coord = sorted_indices[idx]
            current[:, y, x_coord] = baseline[:, y, x_coord]

        current_t = torch.from_numpy(current).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(current_t)
            probs = torch.softmax(logits, dim=1)
            curve[step] = float(probs[0, target_class].cpu().numpy())

    auc_val = float(_auc_trapezoid(curve, dx=1.0 / n_steps_actual))

    return {
        "deletion_auc": auc_val,
        "curve": curve.tolist(),
        "n_steps": n_steps_actual,
    }


def compute_faithfulness_insertion_deletion(
    model: Any,
    images: NDArray[np.float64],
    attributions: NDArray[np.float64],
    lesion_masks: NDArray[np.float64],
    baseline_images: NDArray[np.float64],
    target_classes: list[int],
    n_steps: int = 30,
    device: Any = None,
) -> dict[str, Any]:
    """Batch insertion/deletion AUC computation.

    Args:
        model: PyTorch model.
        images: [N, C, H, W] original images.
        attributions: [N, H, W] normalized attribution maps.
        lesion_masks: [N, H, W] binary lesion-plus-margin masks.
        baseline_images: [N, C, H, W] baseline images (e.g. blurred).
        target_classes: List of target class indices per sample.
        n_steps: Number of perturbation steps.
        device: torch device.

    Returns:
        Dict with per-sample and aggregate insertion/deletion AUCs.
    """
    n = len(images)
    insertion_vals = []
    deletion_vals = []
    failures = []

    for i in range(n):
        try:
            ins = insertion_auc(
                model, images[i], attributions[i], lesion_masks[i],
                target_classes[i], baseline_images[i],
                n_steps=n_steps, device=device,
            )
            insertion_vals.append(ins["insertion_auc"])
        except Exception as e:
            insertion_vals.append(float("nan"))
            failures.append({"sample": i, "metric": "insertion", "error": str(e)})

        try:
            dell = deletion_auc(
                model, images[i], attributions[i], lesion_masks[i],
                target_classes[i], baseline_images[i],
                n_steps=n_steps, device=device,
            )
            deletion_vals.append(dell["deletion_auc"])
        except Exception as e:
            deletion_vals.append(float("nan"))
            failures.append({"sample": i, "metric": "deletion", "error": str(e)})

    ins_arr = np.array(insertion_vals, dtype=np.float64)
    del_arr = np.array(deletion_vals, dtype=np.float64)

    valid = np.isfinite(ins_arr) & np.isfinite(del_arr)

    return {
        "n_samples": n,
        "n_valid": int(valid.sum()),
        "n_failed": len(failures),
        "failures": failures,
        "n_steps": n_steps,
        "insertion_auc_mean": float(np.nanmean(ins_arr)),
        "insertion_auc_median": float(np.nanmedian(ins_arr)),
        "insertion_auc_std": float(np.nanstd(ins_arr)),
        "deletion_auc_mean": float(np.nanmean(del_arr)),
        "deletion_auc_median": float(np.nanmedian(del_arr)),
        "deletion_auc_std": float(np.nanstd(del_arr)),
        "insertion_auc": ins_arr.tolist(),
        "deletion_auc": del_arr.tolist(),
    }
