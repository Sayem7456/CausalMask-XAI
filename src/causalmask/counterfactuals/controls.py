"""Sham / negative controls for counterfactual interventions.

Controls ensure the lesion intervention effect is not confused
with generic image corruption.

Three control types:
1. Same-area random-region removal outside the lesion.
2. Same-area random-region preservation.
3. Shifted-mask control with minimal lesion overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from causalmask.counterfactuals.masks import (
    compute_lesion_bbox,
    _binary_lesion_mask,
    MarginConfig,
)


@dataclass(frozen=True)
class ControlsConfig:
    """Configuration for sham / negative controls.

    Attributes:
        seed: Deterministic seed for random region placement.
        removal_operator: Inpainting method for random removal.
        removal_radius: Inpainting radius (pixels). None = auto.
        removal_auto_scale: Diagonal fraction for auto radius.
        shifted_overlap_max: Max allowed IoU of shifted mask with
            the original lesion for the shifted-mask control.
        shifted_attempts: Max attempts to find a valid shift.
    """
    seed: int = 42
    removal_operator: str = "telea"
    removal_radius: int | None = None
    removal_auto_scale: float = 0.015
    shifted_overlap_max: float = 0.1
    shifted_attempts: int = 100


def sham_mask_area(mask: np.ndarray) -> int:
    """Return the number of foreground pixels in the binary mask."""
    binary = _binary_lesion_mask(mask)
    return int(binary.sum())


def _random_region(
    shape: Tuple[int, int],
    area_pixels: int,
    excluded: np.ndarray | None,
    rng: np.random.Generator,
    max_attempts: int = 200,
) -> Tuple[int, int, int, int]:
    """Find a random rectangular region of ~*area_pixels* pixels
    that does not overlap *excluded*.

    Returns (x_min, y_min, x_max, y_max) or raises RuntimeError.
    """
    h, w = shape[:2]

    side_len = max(1, int(area_pixels ** 0.5))
    half_w = side_len // 2
    half_h = side_len // 2

    for _ in range(max_attempts):
        cx = int(rng.integers(half_w, max(half_w + 1, w - half_w)))
        cy = int(rng.integers(half_h, max(half_h + 1, h - half_h)))
        x_min, x_max = max(cx - half_w, 0), min(cx + half_w, w)
        y_min, y_max = max(cy - half_h, 0), min(cy + half_h, h)

        if excluded is not None:
            region = excluded[y_min:y_max, x_min:x_max]
            if region.any():
                continue

        return x_min, int(y_min), x_max, int(y_max)

    raise RuntimeError(
        f"Could not find random region outside exclusion after {max_attempts} attempts"
    )


def _make_region_mask(
    shape: Tuple[int, int],
    x_min: int, y_min: int, x_max: int, y_max: int,
) -> np.ndarray:
    """Create a binary mask with a filled rectangle region."""
    result = np.zeros(shape[:2], dtype=np.uint8)
    result[y_min:y_max, x_min:x_max] = 1
    return result


def generate_random_region_removal(
    image: np.ndarray,
    lesion_mask: np.ndarray,
    config: ControlsConfig | None = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Remove a random region of same area as the lesion, located
    outside the lesion. Uses Telea inpainting.

    Args:
        image: Input image [H, W, C] uint8.
        lesion_mask: Binary lesion mask.
        config: ControlsConfig.

    Returns:
        (removed_image, control_mask) — image with random region
        inpainted, and the control mask identifying the removed region.
    """
    if config is None:
        config = ControlsConfig()

    h, w = image.shape[:2]
    area = sham_mask_area(lesion_mask)
    rng = np.random.default_rng(config.seed)

    dilated_lesion = cv2.dilate(
        _binary_lesion_mask(lesion_mask),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=2,
    )
    excluded = (dilated_lesion > 0)

    x_min, y_min, x_max, y_max = _random_region(
        (h, w), area, excluded, rng
    )
    control_mask = _make_region_mask((h, w), x_min, y_min, x_max, y_max)

    radius = config.removal_radius
    if radius is None:
        diag = (h ** 2 + w ** 2) ** 0.5
        radius = max(1, int(diag * config.removal_auto_scale))

    flag = cv2.INPAINT_TELEA if config.removal_operator == "telea" else cv2.INPAINT_NS
    inpaint_mask = control_mask * 255
    removed = cv2.inpaint(image.astype(np.uint8), inpaint_mask, radius, flag)

    actual_area = control_mask.sum()
    return removed, control_mask, actual_area


def generate_random_region_preservation(
    image: np.ndarray,
    lesion_mask: np.ndarray,
    config: ControlsConfig | None = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Preserve a random region of same area as the lesion, located
    outside the lesion. Blur the rest of the image.

    Args:
        image: Input image [H, W, C] uint8.
        lesion_mask: Binary lesion mask.
        config: ControlsConfig.

    Returns:
        (preserved_image, control_mask).
    """
    if config is None:
        config = ControlsConfig()

    h, w = image.shape[:2]
    area = sham_mask_area(lesion_mask)
    rng = np.random.default_rng(config.seed)

    dilated_lesion = cv2.dilate(
        _binary_lesion_mask(lesion_mask),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=2,
    )
    excluded = (dilated_lesion > 0)

    x_min, y_min, x_max, y_max = _random_region(
        (h, w), area, excluded, rng
    )
    control_mask = _make_region_mask((h, w), x_min, y_min, x_max, y_max)

    blurred = cv2.GaussianBlur(image, (51, 51), 25)
    mask_3c = np.broadcast_to(
        np.expand_dims(control_mask, axis=-1), image.shape
    ).astype(np.float32)
    result = (image.astype(np.float32) * mask_3c
              + blurred.astype(np.float32) * (1.0 - mask_3c))

    actual_area = control_mask.sum()
    return np.clip(result, 0, 255).astype(np.uint8), control_mask, actual_area


def generate_shifted_mask_control(
    image: np.ndarray,
    lesion_mask: np.ndarray,
    config: ControlsConfig | None = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Shift the lesion mask to a nearby region with minimal overlap.

    Args:
        image: Input image [H, W, C] uint8.
        lesion_mask: Binary lesion mask.
        config: ControlsConfig.

    Returns:
        (inpainted_image, shifted_mask, info_dict).
        info_dict contains: 'shift_x', 'shift_y', 'overlap_iou',
        'found' (bool).
    """
    if config is None:
        config = ControlsConfig()

    h, w = image.shape[:2]
    binary = _binary_lesion_mask(lesion_mask)
    bbox = compute_lesion_bbox(binary)
    box_w = bbox[2] - bbox[0]
    box_h = bbox[3] - bbox[1]
    lesion_pixels = binary.sum()

    rng = np.random.default_rng(config.seed)
    best_iou = 1.0
    best_shift = (0, 0)

    for _ in range(config.shifted_attempts):
        sx = rng.integers(box_w, w)
        sy = rng.integers(box_h, h)
        shifted = np.roll(np.roll(binary, sx, axis=1), sy, axis=0)
        intersection = (binary & shifted).sum()
        union = lesion_pixels + shifted.sum() - intersection
        iou = intersection / max(union, 1)
        if iou < best_iou:
            best_iou = iou
            best_shift = (sx, sy)
        if iou <= config.shifted_overlap_max:
            break

    sx, sy = best_shift
    shifted_mask = np.roll(np.roll(binary, sx, axis=1), sy, axis=0)

    found = best_iou <= config.shifted_overlap_max

    radius = config.removal_radius
    if radius is None:
        diag = (h ** 2 + w ** 2) ** 0.5
        radius = max(1, int(diag * config.removal_auto_scale))

    flag = cv2.INPAINT_TELEA if config.removal_operator == "telea" else cv2.INPAINT_NS
    inpaint_mask = shifted_mask * 255
    inpainted = cv2.inpaint(image.astype(np.uint8), inpaint_mask, radius, flag)

    info = {
        "shift_x": int(sx),
        "shift_y": int(sy),
        "overlap_iou": float(best_iou),
        "found": found,
    }
    return inpainted, shifted_mask, info
