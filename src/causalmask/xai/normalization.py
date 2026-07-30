"""Attribution normalization and caching.

Normalization:
  - normalize_minmax_per_sample: min-max per sample to [0, 1].
  - normalize_percentile_per_sample: percentile-based per-sample normalization.
  - safe_normalize: handles NaN, inf, all-zero maps.

Caching:
  - AttributionCache: keyed by method-config digest, stores/retrieves
    attributions to avoid recomputation.
  - AttributionCacheEntry: single cache entry.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from causalmask.xai.base import AttributionMetadata

logger = logging.getLogger(__name__)

_EPS = 1e-8


def normalize_minmax_per_sample(attr: Tensor, clamp: bool = True) -> Tensor:
    """Min-max normalization per sample to [0, 1].

    Args:
        attr: [B, 1, H, W] or [B, H, W] raw attribution.
        clamp: If False, return unclamped values.

    Returns:
        Normalized tensor same shape as input, [0, 1] per sample.
    """
    was_3d = attr.ndim == 3
    if was_3d:
        attr = attr.unsqueeze(1)
    b = attr.shape[0]
    flat = attr.view(b, -1)
    mins = flat.min(dim=1, keepdim=True).values.view(b, 1, 1, 1)
    maxs = flat.max(dim=1, keepdim=True).values.view(b, 1, 1, 1)
    denom = (maxs - mins).clamp(min=_EPS)
    norm = (attr - mins) / denom
    if clamp:
        norm = norm.clamp(0.0, 1.0)
    if was_3d:
        norm = norm.squeeze(1)
    return norm


def normalize_percentile_per_sample(
    attr: Tensor,
    low_pct: float = 1.0,
    high_pct: float = 99.0,
) -> Tensor:
    """Percentile-based normalization per sample to approximately [0, 1].

    Clips to [low_pct, high_pct] percentile range then min-max normalizes.
    Reduces influence of outlier pixels on the dynamic range.

    Args:
        attr: [B, 1, H, W] or [B, H, W].
        low_pct: Lower percentile for clipping (0-100).
        high_pct: Upper percentile for clipping (0-100).

    Returns:
        Normalized tensor same shape as input.
    """
    was_3d = attr.ndim == 3
    if was_3d:
        attr = attr.unsqueeze(1)
    b = attr.shape[0]
    norm = torch.empty_like(attr)
    for i in range(b):
        sample = attr[i]
        flat = sample.flatten()
        low = float(np.percentile(flat.cpu().numpy(), low_pct))
        high = float(np.percentile(flat.cpu().numpy(), high_pct))
        clipped = sample.clamp(low, high)
        denom = max(high - low, _EPS)
        norm[i] = (clipped - low) / denom
    norm = norm.clamp(0.0, 1.0)
    if was_3d:
        norm = norm.squeeze(1)
    return norm


def safe_normalize(
    attr: Tensor,
    method: str = "minmax",
    input_h: Optional[int] = None,
    input_w: Optional[int] = None,
) -> tuple[Tensor, str]:
    """Safely normalize attribution with NaN/inf/all-zero handling.

    Args:
        attr: [B, 1, H, W] or [B, H, W] raw attribution.
        method: 'minmax' or 'percentile'.
        input_h, input_w: If provided, resize attribution to this size
            after normalization (bilinear interpolation).

    Returns:
        (normalized, failure_flag) tuple.
          - normalized: [B, 1, H, W] normalized and optionally resized.
          - failure_flag: empty string if OK, otherwise error description.

    Raises:
        ValueError: If method is unknown.
    """
    if method not in ("minmax", "percentile"):
        raise ValueError(f"Unknown normalization method: {method}")

    failure_flag = ""

    if not torch.isfinite(attr).all():
        n_bad = (~torch.isfinite(attr)).sum().item()
        failure_flag = f"non_finite_values:{n_bad}"
        logger.warning(f"safe_normalize: {failure_flag}")
        attr = torch.nan_to_num(attr, nan=0.0, posinf=0.0, neginf=0.0)

    b = attr.shape[0]
    flat = attr.view(b, -1)
    all_zero = (flat.abs().max(dim=1).values == 0)

    if all_zero.any():
        n_zero = all_zero.sum().item()
        if failure_flag:
            failure_flag += f";all_zero_maps:{n_zero}"
        else:
            failure_flag = f"all_zero_maps:{n_zero}"
        logger.warning(f"safe_normalize: {n_zero}/{b} maps are all-zero")

    if method == "minmax":
        normalized = normalize_minmax_per_sample(attr, clamp=True)
    else:
        normalized = normalize_percentile_per_sample(attr)

    normalized = torch.nan_to_num(normalized, nan=0.0)

    if input_h is not None and input_w is not None:
        was_3d = normalized.ndim == 3
        if was_3d:
            normalized = normalized.unsqueeze(1)
        if normalized.shape[2] != input_h or normalized.shape[3] != input_w:
            normalized = torch.nn.functional.interpolate(
                normalized,
                size=(input_h, input_w),
                mode="bilinear",
                align_corners=False,
            )
        if was_3d:
            normalized = normalized.squeeze(1)

    return normalized, failure_flag


@dataclass
class AttributionCacheEntry:
    attribution: Tensor
    metadata: AttributionMetadata
    raw_attribution: Optional[Tensor] = None


class AttributionCache:
    """Filesystem-backed attribution cache keyed by method-config digest.

    Cache keys are computed from: checkpoint digest, sample ID, target class,
    method config, and normalization. This avoids recomputation across
    evaluation passes and notebook reruns.

    Usage:
        cache = AttributionCache(Path("artifacts/cache/xai"))
        key = cache.make_key(metadata)
        cached = cache.get(key)
        if cached is None:
            attr = compute(...)
            cache.put(key, attr, metadata)
    """

    def __init__(self, cache_dir: Path):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._cache_dir / "index.parquet"
        self._memory: dict[str, AttributionCacheEntry] = {}

    def make_key(self, metadata: AttributionMetadata) -> str:
        return metadata.digest()

    def get(self, key: str) -> Optional[AttributionCacheEntry]:
        if key in self._memory:
            return self._memory[key]
        pt_path = self._cache_dir / f"{key}.pt"
        if pt_path.exists():
            try:
                data = torch.load(pt_path, map_location="cpu", weights_only=False)
                entry = AttributionCacheEntry(
                    attribution=data["attribution"],
                    metadata=data.get("metadata", AttributionMetadata()),
                    raw_attribution=data.get("raw_attribution"),
                )
                self._memory[key] = entry
                return entry
            except Exception as e:
                logger.warning(f"Cache read failed for {key}: {e}")
        return None

    def put(
        self,
        key: str,
        attribution: Tensor,
        metadata: AttributionMetadata,
        raw_attribution: Optional[Tensor] = None,
    ) -> None:
        entry = AttributionCacheEntry(
            attribution=attribution.cpu(),
            metadata=metadata,
            raw_attribution=raw_attribution.cpu() if raw_attribution is not None else None,
        )
        self._memory[key] = entry
        pt_path = self._cache_dir / f"{key}.pt"
        torch.save(
            {
                "attribution": entry.attribution,
                "metadata": entry.metadata,
                "raw_attribution": entry.raw_attribution,
            },
            pt_path,
        )

    def contains(self, key: str) -> bool:
        if key in self._memory:
            return True
        return (self._cache_dir / f"{key}.pt").exists()

    def clear_memory(self) -> None:
        self._memory.clear()


def compute_checkpoint_digest(checkpoint_path: Path, hash_algorithm: str = "sha256") -> str:
    """Compute a stable digest of a checkpoint file for cache keying.

    Uses the first 2KB and file size as a lightweight fingerprint,
    suitable for distinguishing different training runs without
    loading the full state dict.
    """
    h = hashlib.new(hash_algorithm)
    file_size = checkpoint_path.stat().st_size
    with open(checkpoint_path, "rb") as f:
        header = f.read(2048)
    h.update(header)
    h.update(str(file_size).encode())
    return h.hexdigest()[:12]
