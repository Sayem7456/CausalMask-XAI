"""Dataset discovery, manifest generation, and quality validation for BUSI and BUS-UCLM.

This module provides reusable logic for:
- Scanning extracted dataset directories
- Pairing images with masks
- Normalizing labels
- Generating stable sample IDs
- Computing image statistics and hashes
- Validating image-mask compatibility
- Flagging quality issues
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Label normalization mapping
LABEL_NORMALIZATION = {
    "benign": "benign",
    "malignant": "malignant",
    "normal": "normal",
    # BUSI variations
    "Benign": "benign",
    "Malignant": "malignant",
    "Normal": "normal",
    "BENIGN": "benign",
    "MALIGNANT": "malignant",
    "NORMAL": "normal",
}

# Primary task labels
PRIMARY_TASK_LABELS = {"benign", "malignant"}


@dataclass
class SampleRecord:
    """A single sample record in the manifest."""
    sample_id: str
    dataset: str
    image_path: str  # relative to project root
    mask_path: str  # relative to project root, empty string if no mask
    raw_label: str
    normalized_label: str
    included_in_primary_task: bool
    patient_id: str
    provisional_group_id: str
    image_width: int
    image_height: int
    channels: int
    image_sha256: str
    mask_sha256: str  # empty string if no mask
    mask_area_fraction: float
    has_mask: bool
    quality_flags: list[str] = field(default_factory=list)


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def normalize_label(raw_label: str) -> str:
    """Normalize a label string to lowercase canonical form."""
    normalized = LABEL_NORMALIZATION.get(raw_label)
    if normalized is None:
        # Try case-insensitive match
        normalized = LABEL_NORMALIZATION.get(raw_label.lower())
    if normalized is None:
        logger.warning(f"Unknown label: {raw_label}")
        return raw_label.lower()
    return normalized


def generate_sample_id(dataset: str, label: str, image_stem: str) -> str:
    """Generate a stable sample ID from dataset, label, and image filename stem."""
    # Clean the stem to ensure consistency
    clean_stem = re.sub(r"[^a-zA-Z0-9_-]", "_", image_stem)
    return f"{dataset}_{label}_{clean_stem}"


def compute_mask_area_fraction(mask_array: np.ndarray) -> float:
    """Compute the fraction of non-zero pixels in a binary mask."""
    if mask_array.size == 0:
        return 0.0
    binary = (mask_array > 0).astype(np.float32)
    return float(binary.sum() / binary.size)


def detect_quality_flags(
    image_array: np.ndarray,
    mask_array: Optional[np.ndarray],
    image_path: Path,
) -> list[str]:
    """Detect quality issues in an image and optional mask."""
    flags = []

    # Check image dimensions
    h, w = image_array.shape[:2]
    if h < 50 or w < 50:
        flags.append("very_small_image")
    if h > 3000 or w > 3000:
        flags.append("very_large_image")

    # Check aspect ratio
    aspect = max(h, w) / max(min(h, w), 1)
    if aspect > 5.0:
        flags.append("extreme_aspect_ratio")

    # Check channels
    if image_array.ndim == 2:
        channels = 1
    elif image_array.ndim == 3:
        channels = image_array.shape[2]
    else:
        flags.append("unexpected_channels")
        channels = image_array.ndim

    if channels not in (1, 3):
        flags.append(f"unexpected_channel_count_{channels}")

    # Check for black borders (simple heuristic)
    if image_array.ndim == 3:
        gray = np.mean(image_array, axis=2)
    else:
        gray = image_array.astype(np.float32)

    # Check if top/bottom/left/right borders are mostly black
    border_width = max(2, min(10, min(h, w) // 20))
    borders = [
        gray[:border_width, :],  # top
        gray[-border_width:, :],  # bottom
        gray[:, :border_width],  # left
        gray[:, -border_width:],  # right
    ]
    black_border_count = sum(1 for b in borders if np.mean(b) < 10)
    if black_border_count >= 3:
        flags.append("possible_black_borders")

    # Check for very dark or very bright images
    mean_intensity = np.mean(gray)
    if mean_intensity < 20:
        flags.append("very_dark_image")
    elif mean_intensity > 240:
        flags.append("very_bright_image")

    # Check mask if provided
    if mask_array is not None:
        mask_area = compute_mask_area_fraction(mask_array)
        if mask_area < 0.001:
            flags.append("empty_mask")
        elif mask_area < 0.005:
            flags.append("very_small_mask")
        elif mask_area > 0.9:
            flags.append("very_large_mask")

    return flags


def discover_busi_files(
    extracted_dir: Path,
    dataset_name: str = "busi",
) -> list[dict]:
    """Discover BUSI dataset files from the extracted directory.

    BUSI structure: Dataset_BUSI_with_GT/<label>/<image>.png
    Masks are named: <image>_mask.png (or variations like _mask_1.png)
    """
    samples = []

    # Find the Dataset_BUSI_with_GT directory
    busi_root = extracted_dir
    possible_roots = list(extracted_dir.rglob("Dataset_BUSI_with_GT"))
    if possible_roots:
        busi_root = possible_roots[0]
    else:
        # Try to find label directories directly
        for d in extracted_dir.iterdir():
            if d.is_dir() and d.name.lower() in ("benign", "malignant", "normal"):
                busi_root = extracted_dir
                break

    label_dirs = []
    for d in busi_root.iterdir():
        if d.is_dir() and d.name.lower() in ("benign", "malignant", "normal"):
            label_dirs.append(d)

    for label_dir in sorted(label_dirs):
        raw_label = label_dir.name
        norm_label = normalize_label(raw_label)

        # Find all PNG images in this directory
        image_files = sorted(label_dir.glob("*.png"))

        for img_path in image_files:
            # Skip mask files
            if "_mask" in img_path.name.lower():
                continue

            # Find corresponding mask(s)
            stem = img_path.stem
            mask_patterns = [
                f"{stem}_mask.png",
                f"{stem}_mask_1.png",
                f"{stem}_mask_2.png",
            ]
            mask_paths = []
            for pattern in mask_patterns:
                mask_candidate = label_dir / pattern
                if mask_candidate.exists():
                    mask_paths.append(mask_candidate)

            # Also try wildcard
            if not mask_paths:
                mask_candidates = list(label_dir.glob(f"{stem}_mask*.png"))
                mask_paths = sorted(mask_candidates)

            samples.append({
                "raw_label": raw_label,
                "normalized_label": norm_label,
                "image_path": img_path,
                "mask_paths": mask_paths,
                "dataset": dataset_name,
            })

    return samples


def discover_bus_uclm_files(
    extracted_dir: Path,
    dataset_name: str = "bus_uclm",
) -> list[dict]:
    """Discover BUS-UCLM dataset files from the extracted directory.

    BUS-UCLM has multiple possible structures:
    - bus_uclm_separated/<label>/<image>.png with masks in same dir
    - BUS-UCLM Breast ultrasound lesion segmentation dataset/<label>/<image>.png
    """
    samples = []

    # Find the actual data directory
    possible_roots = [
        extracted_dir / "bus_uclm_separated",
        extracted_dir / "BUS-UCLM Breast ultrasound lesion segmentation dataset",
    ]

    # Also search recursively
    for d in extracted_dir.rglob("*"):
        if d.is_dir() and d.name.lower() in ("benign", "malignant", "normal"):
            if d.parent not in possible_roots:
                possible_roots.append(d.parent)

    data_root = None
    for root in possible_roots:
        if root.exists() and root.is_dir():
            # Check if it has label subdirectories
            label_dirs = [d for d in root.iterdir() if d.is_dir() and d.name.lower() in ("benign", "malignant", "normal")]
            if label_dirs:
                data_root = root
                break

    if data_root is None:
        logger.warning(f"Could not find BUS-UCLM data root in {extracted_dir}")
        return samples

    label_dirs = []
    for d in data_root.iterdir():
        if d.is_dir() and d.name.lower() in ("benign", "malignant", "normal"):
            label_dirs.append(d)

    for label_dir in sorted(label_dirs):
        raw_label = label_dir.name
        norm_label = normalize_label(raw_label)

        # Find all PNG images
        image_files = sorted(label_dir.glob("*.png"))

        for img_path in image_files:
            # Skip mask files
            if "_mask" in img_path.name.lower():
                continue

            stem = img_path.stem
            mask_patterns = [
                f"{stem}_mask.png",
                f"{stem}_mask_1.png",
            ]
            mask_paths = []
            for pattern in mask_patterns:
                mask_candidate = label_dir / pattern
                if mask_candidate.exists():
                    mask_paths.append(mask_candidate)

            if not mask_paths:
                mask_candidates = list(label_dir.glob(f"{stem}_mask*.png"))
                mask_paths = sorted(mask_candidates)

            samples.append({
                "raw_label": raw_label,
                "normalized_label": norm_label,
                "image_path": img_path,
                "mask_paths": mask_paths,
                "dataset": dataset_name,
            })

    return samples


def create_sample_record(
    sample_info: dict,
    project_root: Path,
    patient_id_prefix: str = "",
) -> SampleRecord:
    """Create a SampleRecord from discovered file information."""
    img_path: Path = sample_info["image_path"]
    mask_paths: list[Path] = sample_info["mask_paths"]
    raw_label: str = sample_info["raw_label"]
    norm_label: str = sample_info["normalized_label"]
    dataset: str = sample_info["dataset"]

    # Compute relative paths
    try:
        img_rel = img_path.relative_to(project_root)
    except ValueError:
        img_rel = img_path

    # Use first mask if available
    mask_path = mask_paths[0] if mask_paths else None
    if mask_path is not None:
        try:
            mask_rel = mask_path.relative_to(project_root)
        except ValueError:
            mask_rel = mask_path
    else:
        mask_rel = None

    # Generate sample ID
    sample_id = generate_sample_id(dataset, norm_label, img_path.stem)

    # Generate patient ID (use prefix + label for now, as BUSI doesn't have reliable patient IDs)
    patient_id = f"{patient_id_prefix}{norm_label}_{img_path.stem}" if patient_id_prefix else f"{dataset}_{norm_label}"

    # Provisional group ID (same as patient_id for now)
    provisional_group_id = patient_id

    # Read image and compute properties
    try:
        with Image.open(img_path) as img:
            img_array = np.array(img)
            image_width = img.width
            image_height = img.height
            channels = len(img.getbands())
    except Exception as e:
        logger.error(f"Failed to read image {img_path}: {e}")
        image_width = 0
        image_height = 0
        channels = 0
        img_array = None

    # Compute image SHA-256
    image_sha256 = compute_sha256(img_path)

    # Read mask and compute properties
    mask_array = None
    mask_sha256 = ""
    mask_area_fraction = 0.0
    has_mask = mask_path is not None and mask_path.exists()

    if has_mask and mask_path is not None:
        try:
            with Image.open(mask_path) as msk:
                mask_array = np.array(msk)
                mask_sha256 = compute_sha256(mask_path)
                mask_area_fraction = compute_mask_area_fraction(mask_array)
        except Exception as e:
            logger.error(f"Failed to read mask {mask_path}: {e}")
            has_mask = False

    # Determine if included in primary task
    included_in_primary_task = norm_label in PRIMARY_TASK_LABELS

    # Detect quality flags
    quality_flags = []
    if img_array is not None:
        quality_flags = detect_quality_flags(img_array, mask_array, img_path)

    # Check for multiple masks
    if len(mask_paths) > 1:
        quality_flags.append(f"multiple_masks_found_{len(mask_paths)}")

    # Check if abnormal case has mask
    if included_in_primary_task and not has_mask:
        quality_flags.append("missing_mask_for_abnormal_case")

    return SampleRecord(
        sample_id=sample_id,
        dataset=dataset,
        image_path=str(img_rel),
        mask_path=str(mask_rel) if mask_rel else "",
        raw_label=raw_label,
        normalized_label=norm_label,
        included_in_primary_task=included_in_primary_task,
        patient_id=patient_id,
        provisional_group_id=provisional_group_id,
        image_width=image_width,
        image_height=image_height,
        channels=channels,
        image_sha256=image_sha256,
        mask_sha256=mask_sha256,
        mask_area_fraction=mask_area_fraction,
        has_mask=has_mask,
        quality_flags=quality_flags,
    )


def build_manifest(
    samples: list[dict],
    project_root: Path,
    dataset_name: str,
    patient_id_prefix: str = "",
) -> list[SampleRecord]:
    """Build a complete manifest from discovered samples."""
    records = []
    for i, sample_info in enumerate(samples):
        record = create_sample_record(sample_info, project_root, patient_id_prefix)
        records.append(record)

        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(samples)} samples for {dataset_name}")

    logger.info(f"Built manifest with {len(records)} samples for {dataset_name}")
    return records


def validate_manifest(records: list[SampleRecord]) -> dict:
    """Validate a manifest and return validation results."""
    issues = {
        "duplicate_sample_ids": [],
        "duplicate_image_paths": [],
        "unrecognized_labels": [],
        "abnormal_without_mask": [],
        "empty_masks": [],
        "very_small_masks": [],
        "very_large_masks": [],
        "unreadable_files": [],
        "dimension_mismatches": [],
        "quality_flagged": [],
    }

    seen_ids = set()
    seen_paths = set()

    for record in records:
        # Check for duplicate sample IDs
        if record.sample_id in seen_ids:
            issues["duplicate_sample_ids"].append(record.sample_id)
        seen_ids.add(record.sample_id)

        # Check for duplicate image paths
        if record.image_path in seen_paths:
            issues["duplicate_image_paths"].append(record.image_path)
        seen_paths.add(record.image_path)

        # Check for unrecognized labels
        if record.normalized_label not in PRIMARY_TASK_LABELS and record.normalized_label != "normal":
            issues["unrecognized_labels"].append(record.raw_label)

        # Check for abnormal cases without masks
        if record.included_in_primary_task and not record.has_mask:
            issues["abnormal_without_mask"].append(record.sample_id)

        # Check mask quality
        if record.has_mask:
            if record.mask_area_fraction < 0.001:
                issues["empty_masks"].append(record.sample_id)
            elif record.mask_area_fraction < 0.005:
                issues["very_small_masks"].append(record.sample_id)
            elif record.mask_area_fraction > 0.9:
                issues["very_large_masks"].append(record.sample_id)

        # Check for quality flags
        if record.quality_flags:
            issues["quality_flagged"].append({
                "sample_id": record.sample_id,
                "flags": record.quality_flags,
            })

    # Summary
    summary = {
        "total_samples": len(records),
        "unique_sample_ids": len(seen_ids),
        "unique_image_paths": len(seen_paths),
        "issues": issues,
        "issue_counts": {k: len(v) for k, v in issues.items()},
    }

    return summary


def records_to_dicts(records: list[SampleRecord]) -> list[dict]:
    """Convert SampleRecord objects to dictionaries for serialization."""
    return [asdict(r) for r in records]


def save_manifest_parquet(
    records: list[SampleRecord],
    output_path: Path,
) -> None:
    """Save manifest as Parquet file."""
    import pandas as pd

    dicts = records_to_dicts(records)
    df = pd.DataFrame(dicts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved manifest to {output_path} ({len(df)} rows)")


def save_manifest_summary(
    validation_summary: dict,
    records: list[SampleRecord],
    dataset_name: str,
    output_path: Path,
) -> None:
    """Save manifest summary as JSON."""
    summary = {
        "dataset": dataset_name,
        "total_samples": len(records),
        "label_distribution": {},
        "primary_task_samples": sum(1 for r in records if r.included_in_primary_task),
        "validation_summary": validation_summary,
    }

    # Count labels
    for record in records:
        label = record.normalized_label
        summary["label_distribution"][label] = summary["label_distribution"].get(label, 0) + 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved manifest summary to {output_path}")


def atomic_write_parquet(
    records: list[SampleRecord],
    output_path: Path,
) -> None:
    """Atomically save manifest as Parquet file.
    
    Writes to a temporary file first, then renames to prevent corruption.
    """
    import pandas as pd
    import tempfile
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to temporary file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.parquet', dir=output_path.parent)
    temp_file = None
    try:
        os.close(temp_fd)
        temp_file = Path(temp_path)
        
        dicts = records_to_dicts(records)
        df = pd.DataFrame(dicts)
        df.to_parquet(temp_file, index=False)
        
        # Validate the file was written correctly
        if temp_file.stat().st_size == 0:
            raise ValueError(f"Empty file written: {temp_file}")
        
        # Read back to validate
        test_df = pd.read_parquet(temp_file)
        if len(test_df) != len(records):
            raise ValueError(f"Row count mismatch: wrote {len(records)}, read {len(test_df)}")
        
        # Rename to final destination
        temp_file.rename(output_path)
        logger.info(f"Atomically saved manifest to {output_path} ({len(df)} rows)")
    except Exception as e:
        # Clean up on failure
        if temp_file is not None and temp_file.exists():
            temp_file.unlink()
        raise e


def atomic_write_json(
    validation_summary: dict,
    records: list[SampleRecord],
    dataset_name: str,
    output_path: Path,
) -> None:
    """Atomically save manifest summary as JSON.
    
    Writes to a temporary file first, then renames to prevent corruption.
    """
    import tempfile
    
    summary = {
        "dataset": dataset_name,
        "total_samples": len(records),
        "label_distribution": {},
        "primary_task_samples": sum(1 for r in records if r.included_in_primary_task),
        "validation_summary": validation_summary,
    }

    # Count labels
    for record in records:
        label = record.normalized_label
        summary["label_distribution"][label] = summary["label_distribution"].get(label, 0) + 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to temporary file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.json', dir=output_path.parent)
    temp_file = None
    try:
        os.close(temp_fd)
        temp_file = Path(temp_path)
        
        with open(temp_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Validate the file
        if temp_file.stat().st_size == 0:
            raise ValueError(f"Empty file written: {temp_file}")
        
        # Read back to validate
        with open(temp_file) as f:
            test_data = json.load(f)
        if test_data["total_samples"] != len(records):
            raise ValueError(f"Sample count mismatch")
        
        # Rename to final destination
        temp_file.rename(output_path)
        logger.info(f"Atomically saved manifest summary to {output_path}")
    except Exception as e:
        # Clean up on failure
        if temp_file is not None and temp_file.exists():
            temp_file.unlink()
        raise e


def atomic_write_json_dict(
    data: dict,
    output_path: Path,
) -> None:
    """Atomically save a dictionary as JSON.
    
    Writes to a temporary file first, then renames to prevent corruption.
    """
    import tempfile
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to temporary file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.json', dir=output_path.parent)
    temp_file = None
    try:
        os.close(temp_fd)
        temp_file = Path(temp_path)
        
        with open(temp_file, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        # Validate the file
        if temp_file.stat().st_size == 0:
            raise ValueError(f"Empty file written: {temp_file}")
        
        # Read back to validate
        with open(temp_file) as f:
            json.load(f)
        
        # Rename to final destination
        temp_file.rename(output_path)
        logger.info(f"Atomically saved JSON to {output_path}")
    except Exception as e:
        # Clean up on failure
        if temp_file is not None and temp_file.exists():
            temp_file.unlink()
        raise e
