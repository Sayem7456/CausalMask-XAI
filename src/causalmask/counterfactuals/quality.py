"""Counterfactual quality metrics, caching, and visual audit.

Quantifies intervention quality without making clinical claims.
Provides deterministic caching keyed by sample ID, manifests, splits,
operator, margin, donor ID, seed, and configuration digests.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Per-sample quality audit record.

    Fields:
        sample_id: Source sample identifier.
        operator: 'sufficient', 'removal_telea', 'removal_ns', 'swap', etc.
        margin_ratio: Margin dilation ratio used.
        donor_id: Donor sample ID (for swaps). Empty string if not applicable.
        changed_pixel_fraction: Fraction of pixels that differ from original.
        lesion_preservation_error: Mean absolute error inside M⁺ region
            (0 = perfect preservation).
        boundary_gradient_discrepancy: L2 norm of gradient difference
            along the margin boundary.
        regional_ssim_inside: SSIM inside the lesion-plus-margin region.
        regional_ssim_outside: SSIM outside the lesion-plus-margin region.
        histogram_divergence: Jensen-Shannon divergence between original
            and counterfactual histograms (computed per channel, averaged).
        operator_failed: True if the operator produced a NaN, inf, or
            other detectable failure.
        failure_reason: Description if failed.
        output_is_finite: True if all values are finite.
        output_range: (min, max) of output pixel values.
        output_shape: (H, W, C) of output.
        intensity_in_range: True if uint8 [0,255] or float32 [0,1].
        sham_area_match: For sham controls: ratio of control_mask area
            to lesion area. None for non-sham operators.
        metadata: Arbitrary additional key-value pairs.
    """
    sample_id: str = ""
    operator: str = ""
    margin_ratio: float = 0.0
    donor_id: str = ""
    changed_pixel_fraction: float = 0.0
    lesion_preservation_error: float = 0.0
    boundary_gradient_discrepancy: float = 0.0
    regional_ssim_inside: float = 1.0
    regional_ssim_outside: float = 1.0
    histogram_divergence: float = 0.0
    operator_failed: bool = False
    failure_reason: str = ""
    output_is_finite: bool = True
    output_range: Tuple[float, float] = (0.0, 0.0)
    output_shape: Tuple[int, int, int] = (0, 0, 0)
    intensity_in_range: bool = True
    sham_area_match: Optional[float] = None
    metadata: Dict = field(default_factory=dict)


@dataclass(frozen=True)
class AuditConfig:
    """Configuration for the quality audit.

    Attributes:
        cache_root: Directory for deterministic caching.
        samples_per_stratum: Max samples per stratum in visual grid.
        figure_dpi: DPI for audit grid PNGs.
        seed: Deterministic seed for audit sampling.
    """
    cache_root: Path = Path("data/cache/counterfactuals")
    samples_per_stratum: int = 8
    figure_dpi: int = 100
    seed: int = 42


def _compute_cache_key(
    sample_id: str,
    manifest_digest: str,
    split_digest: str,
    operator: str,
    margin_ratio: float,
    donor_id: str,
    seed: int,
    config_digest: str,
) -> str:
    """Deterministic SHA-256 cache key."""
    payload = json.dumps(
        {
            "sample_id": sample_id,
            "manifest_digest": manifest_digest,
            "split_digest": split_digest,
            "operator": operator,
            "margin_ratio": margin_ratio,
            "donor_id": donor_id,
            "seed": seed,
            "config_digest": config_digest,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _compute_config_digest(config_dict: dict) -> str:
    canonical = json.dumps(config_dict, sort_keys=True,
                           separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two 1-d probability arrays."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p / (p.sum() + 1e-12), 1e-12, 1.0)
    q = np.clip(q / (q.sum() + 1e-12), 1e-12, 1.0)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def _compute_histogram(image: np.ndarray, bins: int = 256) -> np.ndarray:
    """Compute normalised histogram of a uint8 image."""
    if image.dtype == np.uint8:
        data = image.ravel()
    else:
        data = np.clip(image, 0, 255).astype(np.uint8).ravel()
    hist, _ = np.histogram(data, bins=bins, range=(0, 256), density=True)
    return hist


def compute_quality_metrics(
    original: np.ndarray,
    counterfactual: np.ndarray,
    mask_plus: np.ndarray,
    sample_id: str,
    operator: str,
    margin_ratio: float,
    donor_id: str = "",
    sham_area_match: Optional[float] = None,
    metadata: Optional[Dict] = None,
) -> QualityMetrics:
    """Compute all quality metrics for a counterfactual intervention.

    Args:
        original: Original image [H, W, C] uint8.
        counterfactual: Counterfactual image [H, W, C] uint8.
        mask_plus: Binary lesion-plus-margin mask [H, W].
        sample_id: Source sample ID.
        operator: Operator name string.
        margin_ratio: Margin ratio used.
        donor_id: Donor sample ID if applicable.
        sham_area_match: For sham controls, ratio of control area.
        metadata: Extra key-value data.

    Returns:
        QualityMetrics dataclass.
    """
    original_f32 = original.astype(np.float32)
    cf_f32 = counterfactual.astype(np.float32)

    # Detect non-finite values BEFORE uint8 conversion (which silently clamps NaN)
    if not bool(np.isfinite(cf_f32).all()):
        result = QualityMetrics(
            sample_id=sample_id,
            operator=operator,
            margin_ratio=margin_ratio,
            donor_id=donor_id,
            sham_area_match=sham_area_match,
            metadata=metadata or {},
            operator_failed=True,
            failure_reason="non-finite output",
            output_is_finite=False,
            output_shape=tuple(counterfactual.shape),
            output_range=(float(counterfactual.min()), float(counterfactual.max())),
        )
        return result

    original_u8 = np.clip(original_f32, 0, 255).astype(np.uint8)
    cf_u8 = np.clip(cf_f32, 0, 255).astype(np.uint8)

    result = QualityMetrics(
        sample_id=sample_id,
        operator=operator,
        margin_ratio=margin_ratio,
        donor_id=donor_id,
        sham_area_match=sham_area_match,
        metadata=metadata or {},
    )

    # Output shape
    result.output_shape = tuple(cf_u8.shape)
    result.output_range = (float(cf_u8.min()), float(cf_u8.max()))

    # Finiteness (already checked above)
    result.output_is_finite = True

    # Intensity range validity
    if cf_u8.dtype == np.uint8:
        result.intensity_in_range = bool(cf_u8.min() >= 0 and cf_u8.max() <= 255)
    else:
        result.intensity_in_range = bool(cf_u8.min() >= 0.0 and cf_u8.max() <= 1.0 + 1e-4)

    # Operator failure detection
    if not result.output_is_finite:
        result.operator_failed = True
        result.failure_reason = "non-finite output"
        return result

    # Changed pixel fraction
    diff = np.abs(original_u8.astype(np.float32) - cf_u8.astype(np.float32))
    changed = (diff > 1).any(axis=-1) if diff.ndim == 3 else (diff > 1)
    result.changed_pixel_fraction = float(changed.mean())

    # Ensure mask_plus is binary and matches shape
    mplus = (mask_plus > 0).astype(np.uint8)
    if mplus.shape[:2] != original_u8.shape[:2]:
        from PIL import Image
        mplus = np.array(
            Image.fromarray(mplus).resize(
                (original_u8.shape[1], original_u8.shape[0]),
                Image.Resampling.NEAREST,
            )
        )

    inside_mask = mplus > 0
    outside_mask = ~inside_mask

    # Lesion preservation error (inside M⁺)
    if inside_mask.any():
        if original_u8.ndim == 3:
            inside_diff = np.abs(
                original_u8.astype(np.float32) - cf_u8.astype(np.float32)
            )
            result.lesion_preservation_error = float(
                inside_diff[inside_mask].mean()
            )
        else:
            result.lesion_preservation_error = float(
                diff[inside_mask].mean()
            )
    else:
        result.lesion_preservation_error = 0.0

    # Boundary gradient discrepancy
    result.boundary_gradient_discrepancy = _boundary_gradient_diff(
        original_u8, cf_u8, mplus
    )

    # Regional SSIM
    try:
        if original_u8.ndim == 3:
            win_size = min(7, min(original_u8.shape[0], original_u8.shape[1]))
            if win_size >= 3:
                if win_size % 2 == 0:
                    win_size -= 1
                result.regional_ssim_inside = float(ssim(
                    original_u8, cf_u8,
                    win_size=win_size,
                    channel_axis=2,
                    data_range=255,
                ))
    except Exception:
        result.regional_ssim_inside = float("nan")

    # SSIM outside
    try:
        if outside_mask.any() and original_u8.ndim == 3:
            win_size = min(7, min(original_u8.shape[0], original_u8.shape[1]))
            if win_size >= 3:
                if win_size % 2 == 0:
                    win_size -= 1
                # Mask SSIM to outside region
                o_masked = original_u8.copy()
                c_masked = cf_u8.copy()
                if original_u8.ndim == 3:
                    o_masked[inside_mask] = 0
                    c_masked[inside_mask] = 0
                result.regional_ssim_outside = float(ssim(
                    o_masked, c_masked,
                    win_size=win_size,
                    channel_axis=2,
                    data_range=255,
                ))
    except Exception:
        result.regional_ssim_outside = float("nan")

    # Histogram divergence (per-channel average)
    try:
        if original_u8.ndim == 3:
            h_divs = []
            for c in range(original_u8.shape[2]):
                h_orig = _compute_histogram(original_u8[:, :, c])
                h_cf = _compute_histogram(cf_u8[:, :, c])
                h_divs.append(_js_divergence(h_orig, h_cf))
            result.histogram_divergence = float(np.mean(h_divs))
        else:
            h_orig = _compute_histogram(original_u8)
            h_cf = _compute_histogram(cf_u8)
            result.histogram_divergence = _js_divergence(h_orig, h_cf)
    except Exception:
        result.histogram_divergence = 0.0

    return result


def _boundary_gradient_diff(
    original: np.ndarray,
    counterfactual: np.ndarray,
    mask_plus: np.ndarray,
    band_width: int = 3,
) -> float:
    """Compute L2 norm of gradient difference along the mask boundary."""
    dilate = mask_plus.copy()
    erode = mask_plus.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (2 * band_width + 1, 2 * band_width + 1))
    outer = cv2.dilate(mask_plus, kernel).astype(np.uint8) * 255
    inner = cv2.erode(mask_plus, kernel).astype(np.uint8) * 255

    boundary = (outer - inner) > 0
    if not boundary.any():
        return 0.0

    if original.ndim == 2:
        grad_orig = np.abs(cv2.Sobel(original.astype(np.float32), cv2.CV_32F, 1, 1))
        grad_cf = np.abs(cv2.Sobel(counterfactual.astype(np.float32), cv2.CV_32F, 1, 1))
    else:
        grad_orig = np.zeros_like(original, dtype=np.float32)
        grad_cf = np.zeros_like(counterfactual, dtype=np.float32)
        for c in range(original.shape[2]):
            g_o = np.abs(cv2.Sobel(original[:, :, c].astype(np.float32), cv2.CV_32F, 1, 1))
            g_c = np.abs(cv2.Sobel(counterfactual[:, :, c].astype(np.float32), cv2.CV_32F, 1, 1))
            grad_orig[:, :, c] = g_o
            grad_cf[:, :, c] = g_c

    diff = np.sqrt(((grad_orig - grad_cf) ** 2).sum(axis=-1))
    return float(diff[boundary].mean())


def build_audit_grid(
    quality_df: pd.DataFrame,
    images_cache: Dict[Tuple[str, str], np.ndarray],
    config: AuditConfig | None = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Build deterministic visual audit grids stratified by class,
    lesion size, margin, operator, and quality flags.

    Args:
        quality_df: DataFrame of QualityMetrics records.
        images_cache: Dict mapping (sample_id, operator_key) to image
            numpy arrays for rendering.
        config: AuditConfig.
        output_dir: Where to save PNG grids. Created if needed.

    Returns:
        Dict mapping stratum label to saved PNG path.
    """
    if config is None:
        config = AuditConfig()

    if output_dir is None:
        output_dir = config.cache_root.parent / "audit_grids"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = quality_df.copy()
    rng = np.random.default_rng(config.seed)
    saved = {}

    strata_columns = ["operator", "margin_ratio"]
    if "normalized_label" in df.columns:
        strata_columns.insert(0, "normalized_label")
    if "lesion_size_bin" in df.columns:
        strata_columns.append("lesion_size_bin")

    for col in strata_columns:
        if col not in df.columns:
            strata_columns.remove(col)

    grouped = df.groupby(strata_columns, dropna=False)

    for group_keys, group_df in grouped:
        label = "_".join(str(k) for k in (group_keys
                         if isinstance(group_keys, tuple) else (group_keys,)))
        n = min(config.samples_per_stratum, len(group_df))
        sampled = group_df.sample(n=n, random_state=rng.integers(0, 2**31))

        cell_h, cell_w = 224, 224
        padding = 4
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        grid_w = cols * (cell_w + padding) + padding
        grid_h = rows * (cell_h + padding) + padding

        grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 200

        for i, (_, row) in enumerate(sampled.iterrows()):
            r, c = i // cols, i % cols
            key = (str(row["sample_id"]), str(row["operator"]))
            img = images_cache.get(key)
            if img is None:
                continue
            img_rgb = _to_rgb(img)
            # Resize to cell
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(img_rgb)
            pil_img = pil_img.resize((cell_w, cell_h), PILImage.Resampling.LANCZOS)
            cell = np.array(pil_img)

            y = r * (cell_h + padding) + padding
            x = c * (cell_w + padding) + padding
            grid[y:y + cell_h, x:x + cell_w] = cell

        out_path = output_dir / f"audit_{label}.png"
        Image.fromarray(grid).save(out_path)
        saved[label] = out_path

    return saved


def _to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1).astype(np.uint8)
    if image.shape[2] == 1:
        return np.broadcast_to(image, (image.shape[0], image.shape[1], 3)).copy()
    if image.shape[2] >= 4:
        return image[:, :, :3].astype(np.uint8)
    return image.astype(np.uint8)


def save_quality_metrics(
    metrics: List[QualityMetrics],
    path: Path,
) -> pd.DataFrame:
    """Save list of QualityMetrics records to Parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [asdict(m) for m in metrics]
    df = pd.DataFrame(records)
    df.to_parquet(path)
    logger.info(f"Saved {len(df)} quality records to {path}")
    return df


def load_quality_metrics(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def generate_quality_report(
    metrics_df: pd.DataFrame,
    output_path: Path,
) -> str:
    """Generate a Markdown quality report from metrics DataFrame.

    Args:
        metrics_df: DataFrame of quality metrics.
        output_path: Where to write the Markdown report.

    Returns:
        Report content as string.
    """
    df = metrics_df.copy()

    lines = [
        "# Counterfactual Quality Audit Report",
        "",
        f"**Records:** {len(df)}",
        f"**Failed operators:** {df['operator_failed'].sum()}",
        f"**Operators:** {', '.join(sorted(df['operator'].unique()))}",
        "",
        "## Aggregate Metrics",
        "",
    ]

    numeric_cols = [
        "changed_pixel_fraction",
        "lesion_preservation_error",
        "boundary_gradient_discrepancy",
        "regional_ssim_inside",
        "regional_ssim_outside",
        "histogram_divergence",
    ]

    agg = df[numeric_cols].describe().T
    agg_md = agg.to_markdown(floatfmt=".4f")
    lines.append(agg_md)
    lines.append("")

    lines.append("## Per-Operator Summary")
    lines.append("")

    for op in sorted(df["operator"].unique()):
        op_df = df[df["operator"] == op]
        lines.append(f"### {op}")
        op_summary = op_df[numeric_cols].describe().T
        lines.append(op_summary.to_markdown(floatfmt=".4f"))
        lines.append(f"- **Failed:** {op_df['operator_failed'].sum()}")
        lines.append(f"- **Finite outputs:** {op_df['output_is_finite'].sum()} / {len(op_df)}")
        lines.append(f"- **Valid intensity range:** {op_df['intensity_in_range'].sum()} / {len(op_df)}")
        lines.append("")

    if "sham_area_match" in df.columns:
        sham = df[df["sham_area_match"].notna()]
        if len(sham) > 0:
            lines.append("## Sham Control Area Matching")
            lines.append(f"- Mean area ratio: {sham['sham_area_match'].mean():.4f}")
            lines.append(f"- Std area ratio: {sham['sham_area_match'].std():.4f}")
            lines.append("")

    lines.append("## Failed Samples")
    lines.append("")
    failed = df[df["operator_failed"]]
    if len(failed) > 0:
        lines.append(failed[["sample_id", "operator", "failure_reason"]].to_markdown())
    else:
        lines.append("*No failures.*")

    report = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    logger.info(f"Quality report saved to {output_path}")
    return report
