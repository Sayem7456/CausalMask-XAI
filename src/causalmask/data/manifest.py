import hashlib
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


@dataclass
class SampleRecord:
    sample_id: str
    dataset: str
    image_path: str
    mask_path: str
    raw_label: str
    normalized_label: str
    included_in_primary_task: bool
    patient_id: str
    provisional_group_id: str
    image_width: int
    image_height: int
    channels: int
    image_sha256: str
    mask_sha256: str
    mask_area_fraction: float
    has_mask: bool
    quality_flags: list = field(default_factory=list)


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_LABEL_MAP = {"benign": "benign", "malignant": "malignant", "normal": "normal"}


def normalize_label(label: str) -> str:
    return _LABEL_MAP.get(label.lower(), label.lower())


def generate_sample_id(dataset: str, label: str, image_stem: str) -> str:
    stem_clean = re.sub(r"[^\w\-]", "_", image_stem)
    return f"{dataset}_{label}_{stem_clean}"


def compute_mask_area_fraction(mask: np.ndarray) -> float:
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    binary = mask > 0
    total = binary.size
    if total == 0:
        return 0.0
    return float(binary.sum()) / float(total)


def read_image_safe(path: Path) -> Optional[np.ndarray]:
    try:
        img = Image.open(path)
        return np.array(img)
    except Exception:
        return None


def read_mask_safe(path: Path) -> Optional[np.ndarray]:
    try:
        m = Image.open(path)
        arr = np.array(m)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        return arr
    except Exception:
        return None


def binarize_mask(mask: np.ndarray, threshold: int = 127) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return (mask > threshold).astype(np.uint8) * 255


def detect_quality_flags(
    image: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    image_path: Path,
) -> list:
    flags = []

    if image is None:
        flags.append("unreadable_image")
        return flags

    h, w = image.shape[:2]

    if h < 64 or w < 64:
        flags.append("very_small_image")
    if h > 2000 or w > 2000:
        flags.append("very_large_image")

    aspect = max(w, h) / max(min(w, h), 1)
    if aspect > 4.0:
        flags.append("extreme_aspect_ratio")

    if image.ndim == 2:
        channels = 1
    else:
        channels = image.shape[2]
    if channels not in (1, 3, 4):
        flags.append("unexpected_channels")

    if mask is not None:
        mask_sum = mask.sum()
        if mask_sum == 0:
            flags.append("empty_mask")
        else:
            frac = compute_mask_area_fraction(mask)
            if frac < 0.01:
                flags.append("very_small_mask")
            if frac > 0.95:
                flags.append("very_large_mask")
    else:
        flags.append("missing_mask")

    return flags


def _discover_labeled_images(data_root: Path) -> list[dict]:
    samples = []
    valid_labels = {"benign", "malignant", "normal"}

    for label_dir in data_root.iterdir():
        if not label_dir.is_dir():
            continue
        raw_label = label_dir.name
        normalized = normalize_label(raw_label)
        if normalized not in valid_labels:
            continue

        image_files = sorted(label_dir.glob("*"))
        image_map: dict[str, dict] = {}
        for f in image_files:
            if not f.is_file():
                continue
            stem = f.stem
            is_mask = "_mask" in stem or stem.endswith("_mask")
            base_stem = stem.replace("_mask", "")

            if is_mask:
                entry = image_map.setdefault(
                    base_stem, {"image": None, "masks": []}
                )
                entry["masks"].append(f)
            else:
                entry = image_map.setdefault(
                    base_stem, {"image": None, "masks": []}
                )
                entry["image"] = f

        for base_stem, entry in image_map.items():
            if entry["image"] is None:
                continue
            samples.append(
                {
                    "raw_label": raw_label,
                    "normalized_label": normalized,
                    "image_path": entry["image"],
                    "mask_paths": entry["masks"],
                    "image_stem": base_stem,
                }
            )

    return samples


def discover_busi_files(busi_extract_dir: Path) -> list[dict]:
    data_root = busi_extract_dir / "Dataset_BUSI_with_GT"
    if not data_root.exists():
        alt = busi_extract_dir
        if any(alt.iterdir()):
            data_root = alt
        else:
            return []
    return _discover_labeled_images(data_root)


def discover_bus_uclm_files(uclm_extract_dir: Path) -> list[dict]:
    preferred = uclm_extract_dir / "bus_uclm_separated"
    if preferred.exists():
        return _discover_labeled_images(preferred)

    alt = uclm_extract_dir / "BUS-UCLM Breast ultrasound lesion segmentation dataset"
    if alt.exists():
        return _discover_labeled_images(alt)

    if any(uclm_extract_dir.iterdir()):
        return _discover_labeled_images(uclm_extract_dir)

    return []


def create_sample_record(
    sample_info: dict,
    project_root: Path,
    patient_id_prefix: str = "",
) -> Optional[SampleRecord]:
    image_path: Path = sample_info["image_path"]
    mask_paths: list[Path] = sample_info.get("mask_paths", [])
    dataset = sample_info.get("dataset", "busi")
    raw_label = sample_info["raw_label"]
    normalized_label = sample_info["normalized_label"]

    image_np = read_image_safe(image_path)
    if image_np is None:
        return None

    if image_np.ndim == 3:
        h, w, c = image_np.shape
    else:
        h, w = image_np.shape
        c = 1

    image_sha = compute_sha256(image_path)
    image_stem = sample_info.get("image_stem", image_path.stem)
    sample_id = generate_sample_id(dataset, normalized_label, image_stem)

    has_mask = len(mask_paths) > 0
    chosen_mask: Optional[Path] = None
    mask_np: Optional[np.ndarray] = None

    if has_mask:
        chosen_mask = mask_paths[0]
        mask_np = read_mask_safe(chosen_mask)

    mask_sha = compute_sha256(chosen_mask) if chosen_mask and chosen_mask.exists() else ""
    mask_area = 0.0
    if mask_np is not None:
        mask_bin = binarize_mask(mask_np)
        mask_area = compute_mask_area_fraction(mask_bin)

    patient_id = f"{patient_id_prefix}{image_stem}" if patient_id_prefix else image_stem
    provisional_group_id = patient_id

    included = normalized_label != "normal"

    quality_flags = detect_quality_flags(image_np, mask_np, image_path)

    if has_mask and len(mask_paths) > 1:
        quality_flags.append("multiple_masks")

    return SampleRecord(
        sample_id=sample_id,
        dataset=dataset,
        image_path=str(image_path.relative_to(project_root))
        if image_path.is_absolute()
        else str(image_path),
        mask_path=str(chosen_mask.relative_to(project_root))
        if chosen_mask and chosen_mask.is_absolute()
        else str(chosen_mask) if chosen_mask else "",
        raw_label=raw_label,
        normalized_label=normalized_label,
        included_in_primary_task=included,
        patient_id=patient_id,
        provisional_group_id=provisional_group_id,
        image_width=w,
        image_height=h,
        channels=c,
        image_sha256=image_sha,
        mask_sha256=mask_sha,
        mask_area_fraction=mask_area,
        has_mask=has_mask,
        quality_flags=quality_flags,
    )


def validate_manifest(records: list[SampleRecord]) -> dict:
    issues = {
        "duplicate_sample_ids": 0,
        "duplicate_image_paths": 0,
        "labels_not_recognized": 0,
        "missing_masks_for_primary": 0,
    }

    sample_ids = [r.sample_id for r in records]
    if len(sample_ids) != len(set(sample_ids)):
        issues["duplicate_sample_ids"] = len(sample_ids) - len(set(sample_ids))

    image_paths = [r.image_path for r in records]
    if len(image_paths) != len(set(image_paths)):
        issues["duplicate_image_paths"] = len(image_paths) - len(set(image_paths))

    for r in records:
        if r.normalized_label not in ("benign", "malignant", "normal"):
            issues["labels_not_recognized"] += 1

        if r.included_in_primary_task and not r.has_mask:
            issues["missing_masks_for_primary"] += 1

    total = len(records)
    primary = sum(1 for r in records if r.included_in_primary_task)

    return {
        "total_samples": total,
        "primary_task_samples": primary,
        "excluded_normal_samples": total - primary,
        "issue_counts": issues,
        "flagged_samples": [r.sample_id for r in records if r.quality_flags],
    }
