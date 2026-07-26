"""Lesion-plus-margin masks for counterfactual interventions.

Margin is defined relative to the lesion bounding-box scale,
not an undocumented constant pixel count. This ensures margins
generalise across images of different resolutions and lesion sizes.

For a margin ratio r (e.g. 0.05 = 5%), the dilation kernel size is:
    k = max(1, round(r * max(box_w, box_h)))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import cv2


def compute_lesion_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Return (x_min, y_min, x_max, y_max) of non-zero region in *mask*.

    If mask is empty, returns (0, 0, mask_w, mask_h).
    """
    if mask.ndim == 3:
        mask = mask.squeeze(-1) if mask.shape[-1] == 1 else mask
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        h, w = mask.shape[:2]
        return 0, 0, w, h
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return int(x_min), int(y_min), int(x_max + 1), int(y_max + 1)


@dataclass(frozen=True)
class MarginConfig:
    """Configuration for lesion-plus-margin mask generation.

    Attributes:
        margin_ratio: Dilation radius as fraction of max bbox dimension.
            Use 0.0, 0.05, 0.10, or 0.20.
        min_kernel: Minimum kernel radius (pixels).
        max_kernel: Maximum kernel radius cap (pixels).  If None, no cap.
        iterations: Number of dilation iterations.
        feathered_blend_px: Width of Gaussian-feathered boundary for
            alpha blending.  If 0, produces a binary mask.  The alpha
            channel is for blending only — never used as a localization
            ground-truth mask.
    """
    margin_ratio: float = 0.05
    min_kernel: int = 1
    max_kernel: Optional[int] = None
    iterations: int = 1
    feathered_blend_px: int = 0


def _kernel_size_from_bbox(
    x_min: int, y_min: int, x_max: int, y_max: int,
    margin_ratio: float,
    min_kernel: int = 1,
    max_kernel: Optional[int] = None,
) -> int:
    box_w = max(1, x_max - x_min)
    box_h = max(1, y_max - y_min)
    k = int(round(margin_ratio * max(box_w, box_h)))
    k = max(min_kernel, k)
    if max_kernel is not None:
        k = min(max_kernel, k)
    return k


def _binary_lesion_mask(mask: np.ndarray) -> np.ndarray:
    """Ensure *mask* is a 2-d uint8 binary array [H, W] in {0, 1}."""
    if mask.ndim == 3:
        mask = mask[:, :, 0] if mask.shape[2] == 1 else mask.mean(axis=2)
    return (mask > 0).astype(np.uint8)


def dilate_mask(
    mask: np.ndarray,
    kernel_size: int,
    iterations: int = 1,
) -> np.ndarray:
    """Dilate binary mask with a circular structuring element."""
    if kernel_size <= 0:
        return _binary_lesion_mask(mask)
    binary = _binary_lesion_mask(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kernel_size + 1, 2 * kernel_size + 1))
    dilated = cv2.dilate(binary, kernel, iterations=iterations)
    return (dilated > 0).astype(np.uint8)


def lesion_plus_margin(
    mask: np.ndarray,
    config: MarginConfig | None = None,
    image_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Create binary lesion-plus-margin mask M⁺.

    Args:
        mask: Binary lesion mask, any shape [H,W], [H,W,1], [H,W,3].
        config: MarginConfig. Defaults to 5%.
        image_shape: (H, W) to clip output to image bounds.

    Returns:
        Binary uint8 numpy array (same H,W as input or image_shape).
    """
    if config is None:
        config = MarginConfig()

    binary = _binary_lesion_mask(mask)
    x_min, y_min, x_max, y_max = compute_lesion_bbox(binary)
    k = _kernel_size_from_bbox(x_min, y_min, x_max, y_max,
                               config.margin_ratio,
                               config.min_kernel,
                               config.max_kernel)
    dilated = dilate_mask(binary, k, config.iterations)

    if image_shape is not None:
        h, w = image_shape
        if dilated.shape[0] != h or dilated.shape[1] != w:
            dilated = cv2.resize(dilated.astype(np.float32), (w, h),
                                 interpolation=cv2.INTER_NEAREST).astype(np.uint8)

    return dilated


def lesion_plus_margin_feathered(
    mask: np.ndarray,
    config: MarginConfig | None = None,
    image_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Return a float32 alpha mask [H, W] with feathered margin boundary.

    The inner lesion + margin region is 1.0, transitioning smoothly
    to 0.0 over *feathered_blend_px* pixels.  Use for blending only.
    """
    binary = lesion_plus_margin(mask, config, image_shape)
    feather = config.feathered_blend_px if config else 0
    if feather <= 0:
        return binary.astype(np.float32)
    blur_sigma = feather / 2.0
    if blur_sigma < 0.5:
        blur_sigma = 0.5
    alpha = cv2.GaussianBlur(binary.astype(np.float32), (0, 0), blur_sigma)
    return np.clip(alpha, 0.0, 1.0)
