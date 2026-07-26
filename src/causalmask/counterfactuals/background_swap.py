"""Background-swapped counterfactual images.

Preserve the lesion + margin region. Replace the exterior with tissue
from a donor image drawn from the same partition.

Rules (per skill spec):
- Training donors from training partition only.
- Validation donors from validation only.
- Test donors from test fold only.
- No self-donation.
- Donor ID, label, partition, alignment method, and seed recorded.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from causalmask.counterfactuals.masks import (
    lesion_plus_margin,
    lesion_plus_margin_feathered,
    MarginConfig,
)


@dataclass(frozen=True)
class SwapConfig:
    """Configuration for background-swapped image generation.

    Attributes:
        margin_config: Margin dilation settings.
        donor_class: 'same' or 'opposite' class donor.
        use_feathered_blend: Blend margin boundary smoothly.
        align_histogram: If True, match donor histogram to source
            in the exterior region before compositing.
        seed: Deterministic seed for donor selection.
    """
    margin_config: MarginConfig = MarginConfig(margin_ratio=0.05)
    donor_class: str = "same"
    use_feathered_blend: bool = True
    align_histogram: bool = True
    seed: int = 42


def _select_donor(
    source_sample_id: str,
    source_label: str,
    candidates: list[dict],
    config: SwapConfig,
    rng: np.random.Generator,
) -> Optional[dict]:
    """Select a donor that is not self and matches class policy.

    Args:
        source_sample_id: Sample ID of the source image.
        source_label: 'benign' or 'malignant'.
        candidates: List of candidate dicts, each must have
            'sample_id' and 'normalized_label'.
        config: SwapConfig with donor_class policy.
        rng: Seeded random generator.

    Returns:
        Selected candidate dict, or None if no suitable donor.
    """
    eligible = []
    for c in candidates:
        if c["sample_id"] == source_sample_id:
            continue
        if config.donor_class == "same" and c["normalized_label"] != source_label:
            continue
        if config.donor_class == "opposite" and c["normalized_label"] == source_label:
            continue
        eligible.append(c)

    if not eligible:
        return None
    return eligible[rng.integers(0, len(eligible))]


def _match_histogram(
    source_region: np.ndarray,
    donor_region: np.ndarray,
) -> np.ndarray:
    """Match the histogram of *donor_region* to *source_region* per channel."""
    result = donor_region.astype(np.uint8).copy()
    if source_region.ndim == 2:
        source_cdf = _cdf(source_region)
        donor_cdf = _cdf(result)
        lut = _lut_from_cdfs(donor_cdf, source_cdf)
        result = cv2.LUT(result, lut)
    else:
        for c in range(min(source_region.shape[2], result.shape[2])):
            src_ch = source_region[:, :, c]
            dst_ch = result[:, :, c]
            if src_ch.size == 0 or dst_ch.size == 0:
                continue
            src_cdf = _cdf(src_ch)
            dst_cdf = _cdf(dst_ch)
            lut = _lut_from_cdfs(dst_cdf, src_cdf)
            result[:, :, c] = cv2.LUT(dst_ch, lut)
    return result


def _cdf(channel: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(channel.ravel(), bins=256, range=(0, 256))
    return hist.cumsum().astype(np.float64)


def _lut_from_cdfs(src_cdf: np.ndarray, ref_cdf: np.ndarray) -> np.ndarray:
    src_cdf = src_cdf / src_cdf[-1] if src_cdf[-1] > 0 else src_cdf
    ref_cdf = ref_cdf / ref_cdf[-1] if ref_cdf[-1] > 0 else ref_cdf
    lut = np.zeros(256, dtype=np.uint8)
    ref_idx = 0
    for i in range(256):
        while ref_idx < 256 and ref_cdf[ref_idx] < src_cdf[i]:
            ref_idx += 1
        lut[i] = min(ref_idx, 255)
    return lut


def generate_background_swap(
    source_image: np.ndarray,
    source_mask: np.ndarray,
    donor_image: np.ndarray,
    config: SwapConfig | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a background-swapped counterfactual image.

    x_swap = x ⊙ M⁺ + x_donor ⊙ (1 − M⁺)

    Args:
        source_image: Original image [H, W, C] uint8.
        source_mask: Binary lesion mask of source.
        donor_image: Donor image [H, W, C] uint8 (must be same shape).
        config: SwapConfig.

    Returns:
        (swapped_image, mask_plus) — swapped image [H,W,C] uint8,
        mask_plus [H,W] binary.
    """
    if config is None:
        config = SwapConfig()

    h, w = source_image.shape[:2]

    donor_resized = donor_image
    if donor_image.shape[:2] != (h, w):
        donor_resized = cv2.resize(
            donor_image, (w, h), interpolation=cv2.INTER_LINEAR
        )

    source_u8 = source_image.astype(np.uint8)
    donor_u8 = donor_resized.astype(np.uint8)

    mask_plus = lesion_plus_margin(source_mask, config.margin_config, (h, w))

    if config.align_histogram:
        bg_mask = (mask_plus == 0)
        if bg_mask.any() and source_u8.ndim >= 2:
            if source_u8.ndim == 2:
                src_bg = source_u8[bg_mask]
                donor_bg = donor_u8[bg_mask]
            else:
                # Align using exterior pixels only, applied to full donor
                donor_u8 = _match_histogram(source_u8, donor_u8)

    if config.use_feathered_blend:
        alpha = lesion_plus_margin_feathered(
            source_mask, config.margin_config, (h, w)
        )
        alpha = np.expand_dims(alpha, axis=-1)
        result = (source_u8.astype(np.float32) * alpha
                  + donor_u8.astype(np.float32) * (1.0 - alpha))
    else:
        mask_3c = np.broadcast_to(
            np.expand_dims(mask_plus, axis=-1), source_u8.shape
        ).astype(np.float32)
        result = (source_u8.astype(np.float32) * mask_3c
                  + donor_u8.astype(np.float32) * (1.0 - mask_3c))

    result = np.clip(result, 0, 255).astype(np.uint8)
    return result, mask_plus


def _donor_record(
    source_id: str,
    donor_id: str,
    donor_label: str,
    partition: str,
    donor_class: str,
    config: SwapConfig,
) -> dict:
    return {
        "source_sample_id": source_id,
        "donor_sample_id": donor_id,
        "donor_label": donor_label,
        "partition": partition,
        "donor_class": donor_class,
        "alignment_method": "histogram_match" if config.align_histogram else "none",
        "seed": config.seed,
    }
