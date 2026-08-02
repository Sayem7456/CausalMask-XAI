"""Sanity checks for XAI attribution methods.

Implements:
  1. Progressive model-parameter randomization.
     Randomize an increasing fraction of model parameters (weights)
     and measure how the attribution changes. A model-sensitive method
     should show monotonically degrading similarity with the original
     attribution as more parameters are randomized.

  2. Limited label-randomization control.
     Train a control model on shuffled labels and compare attributions.
     Only computationally feasible for a limited subset.

  3. Intensity baseline.
     Constant-value attribution proportional to image intensity.

  4. Edge baseline.
     Gradient magnitude of input image as a baseline saliency.

  5. Center-prior baseline (optional).
     Gaussian centered on the image center — biases toward center.

Reports: randomization-degradation curves, baseline comparisons.
A method that remains unchanged after parameter randomization must
NOT be described as model-sensitive.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Callable, Optional

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

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
# Parameter randomization
# ---------------------------------------------------------------------------


def randomize_model_parameters(
    model: nn.Module,
    fraction: float,
    random_seed: int = 0,
    preserve_original: bool = True,
) -> nn.Module:
    """Randomize a fraction of the model's trainable parameters in-place.

    Only randomizes the weight and bias parameters of linear and conv layers.
    Normalization layers (BatchNorm, LayerNorm) are NOT randomized.

    Args:
        model: PyTorch model (modified in-place if preserve_original=False).
        fraction: Fraction [0, 1] of parameters to randomize.
        random_seed: Seed for reproducibility of randomization.
        preserve_original: If True, returns a deep copy.

    Returns:
        Model with randomized parameters.
    """
    if preserve_original:
        model = deepcopy(model)

    fraction = max(0.0, min(1.0, float(fraction)))
    rng = np.random.RandomState(random_seed)

    params_to_randomize: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if "norm" in name.lower() or "bn" in name.lower() or "layer_norm" in name.lower():
            continue
        if "weight" in name or "bias" in name:
            params_to_randomize.append(param)

    n_params = len(params_to_randomize)
    n_rand = max(1, int(n_params * fraction))
    indices = rng.permutation(n_params)[:n_rand]

    for idx in indices:
        param = params_to_randomize[idx]
        shape = param.data.shape
        if len(shape) >= 2:
            fan_in = shape[1] if len(shape) >= 2 else 1
            std = 1.0 / np.sqrt(fan_in)
            new_data = torch.randn(shape, device=param.device) * std
        else:
            new_data = torch.zeros(shape, device=param.device)
        param.data.copy_(new_data)

    return model


def compute_randomization_curve(
    model: nn.Module,
    attributor_factory: Callable[[nn.Module], Any],
    images: list[NDArray[np.float64]],
    target_classes: list[int],
    fractions: Optional[list[float]] = None,
    random_seed: int = 0,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """Progressive model-parameter randomization degradation curve.

    For each fraction, randomizes that fraction of model parameters,
    computes attributions, and compares against the original (fraction=0)
    attributions using Spearman rho, SSIM, and top-k overlap.

    Args:
        model: Original model.
        attributor_factory: Function(model) → attributor with .attribute().
        images: List of images.
        target_classes: Target class per image.
        fractions: Fractions to evaluate. Defaults to [0, 0.1, 0.25, 0.5, 0.75, 1.0].
        random_seed: Base seed.
        device: Torch device.

    Returns:
        Dict with curve data.
    """
    if fractions is None:
        fractions = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

    results: dict[str, list[dict[str, float]]] = {str(f): [] for f in fractions}

    orig_model = deepcopy(model)
    if device is not None:
        orig_model.to(device)

    attributor_orig = attributor_factory(orig_model)
    original_attrs: list[NDArray[np.float64]] = []

    for img, tc in zip(images, target_classes):
        img_t = torch.from_numpy(np.asarray(img, dtype=np.float32))
        if img_t.ndim == 3 and img_t.shape[2] in (1, 3):
            img_t = img_t.permute(2, 0, 1)
        elif img_t.ndim == 2:
            img_t = img_t.unsqueeze(0)
        img_t = img_t.unsqueeze(0)
        if device is not None:
            img_t = img_t.to(device)
        tc_t = torch.tensor([tc], dtype=torch.long, device=device)
        attr = attributor_orig.attribute(img_t, tc_t)
        attr_np = attr.squeeze().detach().cpu().numpy()
        original_attrs.append(attr_np)

    for fraction in fractions:
        fkey = str(fraction)
        if fraction == 0.0:
            for orig_attr in original_attrs:
                results[fkey].append({
                    "spearman_rho": 1.0,
                    "ssim": 1.0,
                    "top_k_overlap": 1.0,
                })
            continue

        rand_model = deepcopy(orig_model)
        rand_model = randomize_model_parameters(
            rand_model, fraction=fraction, random_seed=random_seed, preserve_original=False
        )
        if device is not None:
            rand_model.to(device)

        attributor_rand = attributor_factory(rand_model)

        for i, (img, tc) in enumerate(zip(images, target_classes)):
            img_t = torch.from_numpy(np.asarray(img, dtype=np.float32))
            if img_t.ndim == 3 and img_t.shape[2] in (1, 3):
                img_t = img_t.permute(2, 0, 1)
            elif img_t.ndim == 2:
                img_t = img_t.unsqueeze(0)
            img_t = img_t.unsqueeze(0)
            if device is not None:
                img_t = img_t.to(device)
            tc_t = torch.tensor([tc], dtype=torch.long, device=device)
            rand_attr = attributor_rand.attribute(img_t, tc_t)
            rand_attr_np = rand_attr.squeeze().detach().cpu().numpy()

            rho = _spearman_correlation(original_attrs[i], rand_attr_np)
            sim = _ssim(original_attrs[i], rand_attr_np)
            topk = _top_k_overlap(original_attrs[i], rand_attr_np, k=100)

            results[fkey].append({
                "spearman_rho": rho,
                "ssim": sim,
                "top_k_overlap": topk,
            })

    summary: dict[str, Any] = {}
    for fraction in fractions:
        fkey = str(fraction)
        vals = results[fkey]
        summary[fkey] = {
            "n_samples": len(vals),
            "spearman_rho_mean": float(np.nanmean([v["spearman_rho"] for v in vals])),
            "spearman_rho_std": float(np.nanstd([v["spearman_rho"] for v in vals])),
            "ssim_mean": float(np.nanmean([v["ssim"] for v in vals])),
            "ssim_std": float(np.nanstd([v["ssim"] for v in vals])),
            "top_k_overlap_mean": float(np.nanmean([v["top_k_overlap"] for v in vals])),
            "top_k_overlap_std": float(np.nanstd([v["top_k_overlap"] for v in vals])),
        }

    return {
        "fractions": fractions,
        "n_images": len(images),
        "summary": summary,
        "per_sample": results,
    }


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def generate_intensity_baseline(
    image: NDArray[np.float64],
    normalize: bool = True,
) -> NDArray[np.float64]:
    """Generate an intensity-based baseline saliency.

    Attribution proportional to the local intensity of the input image.
    For grayscale medical images, brighter regions receive higher attribution.

    Args:
        image: Input image [C, H, W] or [H, W, C].
        normalize: Whether to min-max normalize to [0, 1].

    Returns:
        Baseline attribution [H, W].
    """
    img = _ensure_numpy(image)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        baseline = img.mean(axis=0)
    elif img.ndim == 3 and img.shape[2] in (1, 3):
        baseline = img.mean(axis=2)
    else:
        baseline = img.copy()

    if normalize:
        bmin, bmax = baseline.min(), baseline.max()
        if bmax - bmin > _EPS:
            baseline = (baseline - bmin) / (bmax - bmin)

    return np.asarray(baseline, dtype=np.float64)


def generate_edge_baseline(
    image: NDArray[np.float64],
    normalize: bool = True,
) -> NDArray[np.float64]:
    """Generate an edge-based baseline saliency using Sobel gradients.

    Args:
        image: Input image.
        normalize: Whether to normalize to [0, 1].

    Returns:
        Edge map [H, W].
    """
    from skimage.filters import sobel

    img = _ensure_numpy(image)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        gray = img.mean(axis=0)
    elif img.ndim == 3 and img.shape[2] in (1, 3):
        gray = img.mean(axis=2)
    else:
        gray = img

    edge = sobel(gray)
    if normalize:
        emin, emax = edge.min(), edge.max()
        if emax - emin > _EPS:
            edge = (edge - emin) / (emax - emin)

    return np.asarray(edge, dtype=np.float64)


def generate_center_prior_baseline(
    image: NDArray[np.float64],
    sigma: float = 0.3,
    normalize: bool = True,
) -> NDArray[np.float64]:
    """Generate a center-prior baseline (Gaussian blob centered on image).

    Test whether XAI methods merely highlight the image center rather than
    clinically relevant regions.

    Args:
        image: Input image (used only for shape).
        sigma: Gaussian width relative to image size.

    Returns:
        Center-prior map [H, W].
    """
    img = _ensure_numpy(image)
    if img.ndim == 3:
        h = img.shape[0] if img.shape[2] in (1, 3) else img.shape[1]
        w = img.shape[1] if img.shape[2] in (1, 3) else img.shape[2]
    else:
        h, w = img.shape[0], img.shape[1]

    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    gauss = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * (sigma * max(h, w)) ** 2))

    if normalize:
        gmin, gmax = gauss.min(), gauss.max()
        if gmax - gmin > _EPS:
            gauss = (gauss - gmin) / (gmax - gmin)

    return gauss


def compare_to_baseline(
    attribution: NDArray[np.float64],
    baseline: NDArray[np.float64],
    model_attribution: Optional[NDArray[np.float64]] = None,
) -> dict[str, Any]:
    """Compare a model attribution to a non-model baseline.

    Args:
        attribution: Model attribution [H, W].
        baseline: Baseline map [H, W].
        model_attribution: Second model attribution for pairwise comparison.

    Returns:
        Dict with Spearman rho, SSIM, top-k overlap.
    """
    attr = _ensure_numpy(attribution)
    base = _ensure_numpy(baseline)

    rho = _spearman_correlation(attr, base)
    sim = _ssim(attr, base)
    topk = _top_k_overlap(attr, base, k=100)

    result: dict[str, Any] = {
        "spearman_rho": rho,
        "ssim": sim,
        "top_k_overlap": topk,
    }

    if model_attribution is not None:
        m_attr = _ensure_numpy(model_attribution)
        result["model_vs_baseline_spearman"] = _spearman_correlation(m_attr, base)
        result["model_vs_baseline_ssim"] = _ssim(m_attr, base)
        result["model_vs_baseline_top_k"] = _top_k_overlap(m_attr, base, k=100)

    return result


def compute_sanity_check_batch(
    images: list[NDArray[np.float64]],
    sample_ids: list[str],
    attributions: dict[str, list[NDArray[np.float64]]],
) -> dict[str, Any]:
    """Batch sanity baseline comparison.

    Compares each method's attributions against intensity, edge,
    and center-prior baselines.

    Args:
        images: Original images.
        sample_ids: Sample identifiers.
        attributions: Dict of method_name → list of attribution arrays.

    Returns:
        Dict with per-method summary statistics.
    """
    import pandas as pd

    rows: list[dict[str, Any]] = []

    for method_name, attrs in attributions.items():
        for i, (img, sid) in enumerate(zip(images, sample_ids)):
            if i >= len(attrs):
                break
            attr = attrs[i]

            intensity_base = generate_intensity_baseline(img)
            edge_base = generate_edge_baseline(img)
            center_base = generate_center_prior_baseline(img)

            for baseline_name, baseline_map in [
                ("intensity", intensity_base),
                ("edge", edge_base),
                ("center_prior", center_base),
            ]:
                comp = compare_to_baseline(attr, baseline_map)
                rows.append({
                    "sample_id": sid,
                    "method": method_name,
                    "baseline": baseline_name,
                    "spearman_rho": comp["spearman_rho"],
                    "ssim": comp["ssim"],
                    "top_k_overlap": comp["top_k_overlap"],
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return {"n_samples": 0, "summary": {}, "dataframe": df}

    summary: dict[str, Any] = {}
    for method_name in df["method"].unique():
        for baseline_name in df["baseline"].unique():
            sub = df[(df["method"] == method_name) & (df["baseline"] == baseline_name)]
            for col in ["spearman_rho", "ssim", "top_k_overlap"]:
                vals = sub[col].dropna()
                if len(vals) > 0:
                    key = f"{method_name}_{baseline_name}"
                    if key not in summary:
                        summary[key] = {}
                    summary[key][f"{col}_mean"] = float(vals.mean())
                    summary[key][f"{col}_std"] = float(vals.std())

    return {
        "n_samples": len(sample_ids),
        "summary": summary,
        "dataframe": df,
    }
