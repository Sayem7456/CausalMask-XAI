"""Lesion-removed counterfactual images.

Remove the lesion + margin region and fill it using
OpenCV inpainting methods (Telea and Navier-Stokes).

IMPORTANT: This is a causal intervention, not a medical reconstruction.
The filled region is a mathematical completion — it is not claimed to
represent normal, healthy, or anatomically realistic tissue.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import cv2
import numpy as np

from causalmask.counterfactuals.masks import (
    lesion_plus_margin,
    MarginConfig,
)


class RemovalOperator(Enum):
    TELEA = "telea"
    NAVIER_STOKES = "navier_stokes"


@dataclass(frozen=True)
class RemovalConfig:
    """Configuration for lesion-removed image generation.

    Attributes:
        margin_config: Margin dilation settings.
        operator: Inpainting operator.
        inpaint_radius: Inpainting radius in pixels for OpenCV.
            None means auto-scale relative to image diagonal.
        auto_radius_scale: Fraction of image diagonal used when
            inpaint_radius is None.
        feather_boundary_px: Width of Gaussian blur on the inpainting
            mask boundary to reduce hard edges.
    """
    margin_config: MarginConfig = MarginConfig(margin_ratio=0.05)
    operator: RemovalOperator = RemovalOperator.TELEA
    inpaint_radius: int | None = None
    auto_radius_scale: float = 0.015
    feather_boundary_px: int = 0


def _auto_radius(image_shape: Tuple[int, int], scale: float) -> int:
    h, w = image_shape[:2]
    diag = (h ** 2 + w ** 2) ** 0.5
    return max(1, int(diag * scale))


def generate_lesion_removed(
    image: np.ndarray,
    mask: np.ndarray,
    config: RemovalConfig | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a lesion-removed image via inpainting.

    x_removed = x ⊙ (1 − M⁺) + inpaint(x, M⁺) ⊙ M⁺

    Args:
        image: Input image [H, W, C] uint8 (or convertible).
        mask: Binary lesion mask.
        config: RemovalConfig.

    Returns:
        (removed_image, mask_plus) — removed image [H,W,C] uint8,
        mask_plus [H,W] binary.
    """
    if config is None:
        config = RemovalConfig()

    h, w = image.shape[:2]
    image_u8 = image.astype(np.uint8) if image.dtype != np.uint8 else image.copy()
    mask_plus = lesion_plus_margin(mask, config.margin_config, (h, w))

    inpaint_mask = mask_plus.copy()
    if config.feather_boundary_px > 0:
        k = 2 * config.feather_boundary_px + 1
        inpaint_mask = cv2.GaussianBlur(
            inpaint_mask.astype(np.float32), (k, k), config.feather_boundary_px
        )
        inpaint_mask = (inpaint_mask > 0).astype(np.uint8) * 255

    radius = config.inpaint_radius
    if radius is None:
        radius = _auto_radius((h, w), config.auto_radius_scale)

    if config.operator == RemovalOperator.TELEA:
        flag = cv2.INPAINT_TELEA
    elif config.operator == RemovalOperator.NAVIER_STOKES:
        flag = cv2.INPAINT_NS
    else:
        raise ValueError(f"Unknown removal operator: {config.operator}")

    inpainted = cv2.inpaint(image_u8, inpaint_mask, radius, flag)

    return inpainted, mask_plus
