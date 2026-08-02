"""Explanation robustness evaluation.

Measures saliency-map stability under diagnosis-preserving transformations:
  - horizontal flip (with geometric inverse)
  - mild contrast adjustment
  - mild gamma correction
  - small translation (with geometric inverse)
  - mild speckle noise

For geometric transforms:
  1. transform the image
  2. compute attribution
  3. invert the attribution geometry
  4. compare in the original coordinate space

Reports: prediction stability, probability change, Spearman rho,
SSIM, top-k attribution overlap, localization change.

Important: Do not interpret explanation instability without also
reporting whether the model prediction changed.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from causalmask.evaluation.localization import attribution_mass_inside_mask

logger = logging.getLogger(__name__)

_EPS = 1e-8


def _ensure_numpy(x: NDArray[np.floating] | Tensor) -> NDArray[np.float64]:
    if isinstance(x, Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _spearman_correlation(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    from scipy.stats import spearmanr

    a = a.ravel()
    b = b.ravel()
    if np.std(a) < _EPS or np.std(b) < _EPS:
        return float("nan")
    rho, _ = spearmanr(a, b)
    return float(rho)


def _ssim(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    from skimage.metrics import structural_similarity

    a_min, a_max = a.min(), a.max()
    b_min, b_max = b.min(), b.max()
    dr = max(a_max - a_min, b_max - b_min, _EPS)
    ssim_val, _ = structural_similarity(a, b, full=True, data_range=dr)
    return float(ssim_val)


def _top_k_overlap(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    k: int = 100,
) -> float:
    a_flat = a.ravel()
    b_flat = b.ravel()
    top_a = set(np.argsort(a_flat)[::-1][:k])
    top_b = set(np.argsort(b_flat)[::-1][:k])
    if not top_a or not top_b:
        return float("nan")
    return len(top_a & top_b) / k


# ---------------------------------------------------------------------------
# Robustness transforms
# ---------------------------------------------------------------------------


def apply_horizontal_flip(
    image: NDArray[np.float64],
    attribution_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    predict_func: Callable[[NDArray[np.float64]], tuple[int, NDArray[np.float64]]],
) -> dict[str, Any]:
    """Horizontal flip robustness test.

    1. Flip image horizontally.
    2. Compute new prediction and attribution on flipped image.
    3. Flip attribution back to original coordinate space.
    4. Compare.

    Args:
        image: Original image [C, H, W] or [H, W, C] in numpy.
        attribution_func: Function(image_np) → attribution_np [H, W].
        predict_func: Function(image_np) → (predicted_class, probabilities).

    Returns:
        Dict with robustness metrics.
    """
    from skimage.transform import resize as skresize

    img = _ensure_numpy(image)
    img_for_flip: NDArray[np.float64]
    if img.ndim == 3 and img.shape[0] in (1, 3):
        c, h, w = img.shape
        img_for_flip = img.transpose(1, 2, 0)
    elif img.ndim == 3 and img.shape[2] in (1, 3):
        img_for_flip = img
        c = img.shape[2]
        h, w = img.shape[0], img.shape[1]
    else:
        h, w = img.shape[0], img.shape[1]
        c = 1
        img_for_flip = img

    flipped_img_np = np.fliplr(img_for_flip)

    orig_pred, orig_probs = predict_func(img_for_flip)
    flip_pred, flip_probs = predict_func(flipped_img_np)

    orig_conf = orig_probs[orig_pred]
    flip_conf = flip_probs[orig_pred]

    orig_attr = attribution_func(img_for_flip)
    flip_attr = attribution_func(flipped_img_np)
    flip_attr_inv = np.fliplr(flip_attr)

    return _compare_attributions(
        orig_attr,
        flip_attr_inv,
        orig_pred=orig_pred,
        flip_pred=flip_pred,
        orig_conf=float(orig_conf),
        flip_conf=float(flip_conf),
        transform_name="horizontal_flip",
    )


def apply_contrast_adjustment(
    image: NDArray[np.float64],
    attribution_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    predict_func: Callable[[NDArray[np.float64]], tuple[int, NDArray[np.float64]]],
    factor: float = 1.1,
) -> dict[str, Any]:
    """Mild contrast adjustment."""
    img = _ensure_numpy(image)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img_hwc = img.transpose(1, 2, 0)
    elif img.ndim == 3 and img.shape[2] in (1, 3):
        img_hwc = img
    else:
        img_hwc = img

    mean_val = img_hwc.mean()
    adjusted = (img_hwc - mean_val) * factor + mean_val
    adjusted = np.clip(adjusted, 0, 1) if img_hwc.max() <= 1.0 else np.clip(adjusted, 0, 255)

    orig_pred, orig_probs = predict_func(img_hwc)
    adj_pred, adj_probs = predict_func(adjusted)

    orig_conf = orig_probs[orig_pred]
    adj_conf = adj_probs[orig_pred]

    orig_attr = attribution_func(img_hwc)
    adj_attr = attribution_func(adjusted)

    return _compare_attributions(
        orig_attr,
        adj_attr,
        orig_pred=orig_pred,
        flip_pred=adj_pred,
        orig_conf=float(orig_conf),
        flip_conf=float(adj_conf),
        transform_name="contrast",
    )


def apply_gamma_correction(
    image: NDArray[np.float64],
    attribution_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    predict_func: Callable[[NDArray[np.float64]], tuple[int, NDArray[np.float64]]],
    gamma: float = 1.1,
) -> dict[str, Any]:
    """Mild gamma correction."""
    img = _ensure_numpy(image)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img_hwc = img.transpose(1, 2, 0)
    elif img.ndim == 3 and img.shape[2] in (1, 3):
        img_hwc = img
    else:
        img_hwc = img

    vmax = img_hwc.max()
    img_norm = img_hwc / max(vmax, _EPS)
    corrected = np.power(img_norm, gamma)
    if vmax > 1.0:
        corrected = corrected * vmax
    corrected = np.clip(corrected, 0, 1) if img_hwc.max() <= 1.0 else np.clip(corrected, 0, 255)

    orig_pred, orig_probs = predict_func(img_hwc)
    gamma_pred, gamma_probs = predict_func(corrected)

    orig_conf = orig_probs[orig_pred]
    gamma_conf = gamma_probs[orig_pred]

    orig_attr = attribution_func(img_hwc)
    gamma_attr = attribution_func(corrected)

    return _compare_attributions(
        orig_attr,
        gamma_attr,
        orig_pred=orig_pred,
        flip_pred=gamma_pred,
        orig_conf=float(orig_conf),
        flip_conf=float(gamma_conf),
        transform_name="gamma",
    )


def apply_small_translation(
    image: NDArray[np.float64],
    attribution_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    predict_func: Callable[[NDArray[np.float64]], tuple[int, NDArray[np.float64]]],
    shift_x: int = 5,
    shift_y: int = 5,
) -> dict[str, Any]:
    """Small translation robustness test.

    1. Translate image by (shift_x, shift_y).
    2. Compute prediction and attribution.
    3. Translate attribution back by (-shift_x, -shift_y).
    4. Crop to valid (non-padded) overlap region for comparison.
    """
    img = _ensure_numpy(image)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        c, h, w = img.shape
        img_hwc = img.transpose(1, 2, 0)
    elif img.ndim == 3 and img.shape[2] in (1, 3):
        h, w, c = img.shape
        img_hwc = img
    else:
        h, w = img.shape[0], img.shape[1]
        c = 1
        img_hwc = img

    shift_x = int(shift_x)
    shift_y = int(shift_y)

    translated = np.zeros_like(img_hwc)
    x_src_start = max(0, -shift_x)
    x_src_end = w - max(0, shift_x)
    x_dst_start = max(0, shift_x)
    x_dst_end = w + min(0, shift_x)
    y_src_start = max(0, -shift_y)
    y_src_end = h - max(0, shift_y)
    y_dst_start = max(0, shift_y)
    y_dst_end = h + min(0, shift_y)

    translated[y_dst_start:y_dst_end, x_dst_start:x_dst_end] = img_hwc[y_src_start:y_src_end, x_src_start:x_src_end]

    orig_pred, orig_probs = predict_func(img_hwc)
    trans_pred, trans_probs = predict_func(translated)

    orig_conf = orig_probs[orig_pred]
    trans_conf = trans_probs[orig_pred]

    orig_attr = attribution_func(img_hwc)
    trans_attr = attribution_func(translated)

    trans_attr_inv = np.zeros_like(trans_attr)
    trans_attr_inv[y_src_start:y_src_end, x_src_start:x_src_end] = trans_attr[y_dst_start:y_dst_end, x_dst_start:x_dst_end]

    valid_mask = np.zeros((h, w), dtype=bool)
    valid_mask[y_src_start:y_src_end, x_src_start:x_src_end] = True

    return _compare_attributions(
        orig_attr,
        trans_attr_inv,
        orig_pred=orig_pred,
        flip_pred=trans_pred,
        orig_conf=float(orig_conf),
        flip_conf=float(trans_conf),
        transform_name="translation",
        valid_mask=valid_mask,
    )


def apply_speckle_noise(
    image: NDArray[np.float64],
    attribution_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    predict_func: Callable[[NDArray[np.float64]], tuple[int, NDArray[np.float64]]],
    std: float = 0.02,
) -> dict[str, Any]:
    """Mild multiplicative speckle noise."""
    img = _ensure_numpy(image)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img_hwc = img.transpose(1, 2, 0)
    elif img.ndim == 3 and img.shape[2] in (1, 3):
        img_hwc = img
    else:
        img_hwc = img

    noise = np.random.normal(0, std, img_hwc.shape)
    noisy = img_hwc + noise * img_hwc
    if img_hwc.max() <= 1.0:
        noisy = np.clip(noisy, 0, 1)
    else:
        noisy = np.clip(noisy, 0, 255)

    orig_pred, orig_probs = predict_func(img_hwc)
    noisy_pred, noisy_probs = predict_func(noisy)

    orig_conf = orig_probs[orig_pred]
    noisy_conf = noisy_probs[orig_pred]

    orig_attr = attribution_func(img_hwc)
    noisy_attr = attribution_func(noisy)

    return _compare_attributions(
        orig_attr,
        noisy_attr,
        orig_pred=orig_pred,
        flip_pred=noisy_pred,
        orig_conf=float(orig_conf),
        flip_conf=float(noisy_conf),
        transform_name="speckle",
    )


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------


def _compare_attributions(
    orig_attr: NDArray[np.float64],
    trans_attr: NDArray[np.float64],
    orig_pred: int,
    flip_pred: int,
    orig_conf: float,
    flip_conf: float,
    transform_name: str,
    valid_mask: Optional[NDArray[np.bool_]] = None,
) -> dict[str, Any]:
    """Compare original and transformed attribution maps."""
    orig_attr = _ensure_numpy(orig_attr)
    trans_attr = _ensure_numpy(trans_attr)

    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        orig_flat = orig_attr[valid_mask]
        trans_flat = trans_attr[valid_mask]
        spearman_val = _spearman_correlation(orig_flat.reshape(-1, 1), trans_flat.reshape(-1, 1)) if len(orig_flat) > 2 else float("nan")
    else:
        spearman_val = _spearman_correlation(orig_attr, trans_attr)

    if orig_attr.shape != trans_attr.shape:
        ssim_val = float("nan")
    else:
        ssim_val = _ssim(orig_attr, trans_attr)

    topk = _top_k_overlap(orig_attr, trans_attr, k=100)

    prob_change = abs(orig_conf - flip_conf)
    prediction_stable = int(orig_pred == flip_pred)

    return {
        "transform": transform_name,
        "prediction_stable": prediction_stable,
        "probability_change": float(prob_change),
        "orig_confidence": float(orig_conf),
        "transformed_confidence": float(flip_conf),
        "spearman_rho": spearman_val,
        "ssim": ssim_val,
        "top_k_overlap": topk,
    }


def compute_robustness_for_sample(
    image: NDArray[np.float64],
    attribution_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    predict_func: Callable[[NDArray[np.float64]], tuple[int, NDArray[np.float64]]],
    sample_id: str = "",
    transforms: Optional[list[str]] = None,
    shift_pixels: int = 5,
    contrast_factor: float = 1.1,
    gamma_val: float = 1.1,
    speckle_std: float = 0.02,
) -> list[dict[str, Any]]:
    """Compute robustness metrics for one sample across specified transforms.

    Args:
        image: Image as numpy array.
        attribution_func: Function that returns attribution map.
        predict_func: Function that returns (class, probabilities).
        sample_id: Sample identifier for output rows.
        transforms: List of transform names. Defaults to all five.
        shift_pixels: Translation pixels.
        contrast_factor: Contrast multiplication factor.
        gamma_val: Gamma value for gamma correction.
        speckle_std: Speckle noise standard deviation.

    Returns:
        List of dicts, one per transform.
    """
    if transforms is None:
        transforms = sorted(_TRANSFORM_MAP.keys())

    rows: list[dict[str, Any]] = []
    for tname in transforms:
        try:
            if tname == "horizontal_flip":
                result = apply_horizontal_flip(image, attribution_func, predict_func)
            elif tname == "contrast":
                result = apply_contrast_adjustment(image, attribution_func, predict_func, factor=contrast_factor)
            elif tname == "gamma":
                result = apply_gamma_correction(image, attribution_func, predict_func, gamma=gamma_val)
            elif tname == "translation":
                result = apply_small_translation(image, attribution_func, predict_func, shift_x=shift_pixels, shift_y=shift_pixels)
            elif tname == "speckle":
                result = apply_speckle_noise(image, attribution_func, predict_func, std=speckle_std)
            else:
                result = {"transform": tname, "error": f"unknown_transform:{tname}"}
        except Exception as e:
            result = {"transform": tname, "error": str(e)}

        result["sample_id"] = sample_id
        rows.append(result)

    return rows


_TRANSFORM_MAP: dict[str, Callable] = {
    "horizontal_flip": apply_horizontal_flip,
    "contrast": apply_contrast_adjustment,
    "gamma": apply_gamma_correction,
    "translation": apply_small_translation,
    "speckle": apply_speckle_noise,
}


def compute_robustness_batch(
    images: list[NDArray[np.float64]],
    sample_ids: list[str],
    attribution_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    predict_func: Callable[[NDArray[np.float64]], tuple[int, NDArray[np.float64]]],
    transforms: Optional[list[str]] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Batch robustness evaluation.

    Returns:
        Dict with summary statistics and per-sample rows.
    """
    import pandas as pd

    all_rows: list[dict[str, Any]] = []
    for img, sid in zip(images, sample_ids):
        rows = compute_robustness_for_sample(
            img, attribution_func, predict_func, sample_id=sid, transforms=transforms, **kwargs
        )
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    if df.empty:
        return {
            "n_samples": 0,
            "n_transforms": 0,
            "summary": {},
            "dataframe": df,
        }

    summary: dict[str, Any] = {}
    for tname in df["transform"].unique():
        sub = df[df["transform"] == tname]
        if "error" in sub.columns:
            err_series = sub["error"].astype(str)
            n_err = int((err_series != "").sum())
            n_ok = len(sub) - n_err
        else:
            n_err = 0
            n_ok = len(sub)

        summary[tname] = {
            "n_total": len(sub),
            "n_errors": n_err,
            "n_valid": n_ok,
        }
        for col in ["prediction_stable", "probability_change", "spearman_rho", "ssim", "top_k_overlap"]:
            if col in sub.columns:
                vals = sub[col].dropna()
                if len(vals) > 0:
                    summary[tname][f"{col}_mean"] = float(vals.mean())
                    summary[tname][f"{col}_median"] = float(vals.median())
                    summary[tname][f"{col}_std"] = float(vals.std())

    return {
        "n_samples": len(sample_ids),
        "n_transforms": len(summary),
        "summary": summary,
        "dataframe": df,
    }


def compute_localization_change(
    orig_attr: NDArray[np.float64],
    trans_attr: NDArray[np.float64],
    mask: NDArray[np.float64],
) -> dict[str, Any]:
    """Compute change in localization metrics between original and transformed attributions.

    Args:
        orig_attr: Original attribution [H, W].
        trans_attr: Transformed attribution [H, W].
        mask: Binary lesion-plus-margin mask [H, W].

    Returns:
        Dict with mass_inside change for both orig and trans.
    """
    orig_attr = _ensure_numpy(orig_attr)
    trans_attr = _ensure_numpy(trans_attr)
    mask = _ensure_numpy(mask)

    orig_mass = attribution_mass_inside_mask(orig_attr, mask)
    trans_mass = attribution_mass_inside_mask(trans_attr, mask)

    return {
        "orig_mass_inside": float(orig_mass),
        "trans_mass_inside": float(trans_mass),
        "mass_change": float(trans_mass - orig_mass),
    }
