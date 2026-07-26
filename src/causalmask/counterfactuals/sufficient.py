"""Lesion-sufficient counterfactual images.

Preserve lesion + margin pixels; blur the exterior.
The operator is an intervention — it does not generate
anatomically meaningful tissue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from causalmask.counterfactuals.masks import (
    lesion_plus_margin,
    lesion_plus_margin_feathered,
    MarginConfig,
)


@dataclass(frozen=True)
class SufficientConfig:
    """Configuration for lesion-sufficient image generation.

    Attributes:
        margin_config: Margin dilation settings.
        blur_method: 'gaussian' (default) — future: 'median', 'bilateral'.
        blur_sigma: Gaussian sigma for exterior blur.
        blur_kernel_truncate: Kernel truncation multiplier for Gaussian.
        use_feathered_blend: If True, blend with alpha at margin boundary.
            If False, produce a hard-edge composite.
        clamp_to_input_range: If True, clip output to [0, 255].
            If False, let values run (for float images).
    """
    margin_config: MarginConfig = MarginConfig(margin_ratio=0.05)
    blur_method: str = "gaussian"
    blur_sigma: float = 20.0
    blur_kernel_truncate: float = 3.0
    use_feathered_blend: bool = True
    clamp_to_input_range: bool = True


def _blur_image(image: np.ndarray, config: SufficientConfig) -> np.ndarray:
    """Apply configurable blur to the full image."""
    if config.blur_method == "gaussian":
        k = max(1, int(config.blur_sigma * config.blur_kernel_truncate))
        if k % 2 == 0:
            k += 1
        blurred = cv2.GaussianBlur(image, (k, k), config.blur_sigma)
        return blurred
    raise ValueError(f"Unknown blur method: {config.blur_method}")


def generate_lesion_sufficient(
    image: np.ndarray,
    mask: np.ndarray,
    config: SufficientConfig | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a lesion-sufficient image.

    x_sufficient = x ⊙ M⁺ + B(x) ⊙ (1 − M⁺)

    Args:
        image: Input image [H, W, C] uint8 or float32.
        mask: Binary lesion mask (any compatible shape).
        config: SufficientConfig.

    Returns:
        (sufficient_image, mask_plus) — image [H,W,C] same dtype,
        mask_plus [H,W] binary.
    """
    if config is None:
        config = SufficientConfig()

    h, w = image.shape[:2]
    mask_plus = lesion_plus_margin(mask, config.margin_config, (h, w))
    blurred = _blur_image(image, config)

    if config.use_feathered_blend:
        alpha = lesion_plus_margin_feathered(
            mask, config.margin_config, (h, w)
        )
        alpha = np.expand_dims(alpha, axis=-1)
        result = (image.astype(np.float32) * alpha
                  + blurred.astype(np.float32) * (1.0 - alpha))
    else:
        mask_3c = np.broadcast_to(
            np.expand_dims(mask_plus, axis=-1), image.shape
        ).astype(np.float32)
        result = (image.astype(np.float32) * mask_3c
                  + blurred.astype(np.float32) * (1.0 - mask_3c))

    if image.dtype == np.uint8 or config.clamp_to_input_range:
        result = np.clip(result, 0, 255).astype(np.uint8)
    elif image.dtype == np.float32:
        result = result.astype(np.float32)
    else:
        result = result.astype(image.dtype)

    return result, mask_plus
