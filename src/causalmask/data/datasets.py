"""Dataset adapters for loading images and masks from manifests.

This module provides:
- PyTorch Dataset classes for BUSI and BUS-UCLM
- Image and mask loading utilities
- Transform application for paired image-mask data
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BreastUltrasoundDataset(Dataset):
    """PyTorch Dataset for breast ultrasound images with optional masks.

    Loads images and masks from a manifest DataFrame, applying transforms
    to both image and mask simultaneously for paired augmentation.
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        project_root: Path,
        transform=None,
        mask_transform=None,
        include_mask: bool = True,
        target_size: Optional[tuple[int, int]] = None,
    ):
        """Initialize the dataset.

        Args:
            manifest_df: DataFrame with columns from the manifest.
            project_root: Root directory of the project.
            transform: Transforms to apply to images.
            mask_transform: Transforms to apply to masks.
            include_mask: Whether to load masks.
            target_size: Optional (height, width) to resize images and masks.
        """
        self.manifest_df = manifest_df.copy()
        self.project_root = Path(project_root)
        self.transform = transform
        self.mask_transform = mask_transform
        self.include_mask = include_mask
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.manifest_df)

    def __getitem__(self, idx: int) -> dict:
        """Load a single sample.

        Returns:
            Dictionary with keys:
                - image: Tensor [C, H, W]
                - mask: Tensor [1, H, W] (if include_mask=True)
                - label: Integer label (0=benign, 1=malignant)
                - sample_id: String sample identifier
                - metadata: Additional information
        """
        row = self.manifest_df.iloc[idx]

        # Load image
        img_path = self.project_root / row["image_path"]
        image = Image.open(img_path).convert("RGB")

        # Load mask if available and requested
        mask = None
        if self.include_mask and row.get("has_mask", False) and row.get("mask_path", ""):
            mask_path = self.project_root / row["mask_path"]
            if mask_path.exists():
                mask = Image.open(mask_path).convert("L")

        # Resize if target_size specified
        if self.target_size is not None:
            h, w = self.target_size
            image = image.resize((w, h), Image.Resampling.BILINEAR)
            if mask is not None:
                mask = mask.resize((w, h), Image.Resampling.NEAREST)

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        if mask is not None and self.mask_transform is not None:
            mask = self.mask_transform(mask)
        elif mask is not None:
            # Default: convert to tensor and normalize
            mask = torch.from_numpy(np.array(mask)).float().unsqueeze(0) / 255.0

        # Convert image to tensor if not already
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0

        # Map label to integer
        label_map = {"benign": 0, "malignant": 1, "normal": -1}
        label_int = label_map.get(row["normalized_label"], -1)

        # Build result
        result = {
            "image": image,
            "label": label_int,
            "sample_id": row["sample_id"],
            "metadata": {
                "dataset": row["dataset"],
                "raw_label": row["raw_label"],
                "normalized_label": row["normalized_label"],
                "included_in_primary_task": row["included_in_primary_task"],
                "patient_id": row.get("patient_id", ""),
            },
        }

        if mask is not None:
            result["mask"] = mask

        return result


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    """Load a manifest Parquet file.

    Args:
        manifest_path: Path to the manifest Parquet file.

    Returns:
        DataFrame with manifest data.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_parquet(manifest_path)
    logger.info(f"Loaded manifest from {manifest_path}: {len(df)} samples")
    return df


def filter_manifest(
    df: pd.DataFrame,
    include_primary_task_only: bool = False,
    datasets: Optional[list[str]] = None,
    labels: Optional[list[str]] = None,
    exclude_flags: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Filter manifest DataFrame based on criteria.

    Args:
        df: Input manifest DataFrame.
        include_primary_task_only: If True, only include primary task samples.
        datasets: Filter to specific datasets (e.g., ["busi"]).
        labels: Filter to specific labels (e.g., ["benign", "malignant"]).
        exclude_flags: Exclude samples with any of these quality flags.

    Returns:
        Filtered DataFrame.
    """
    filtered = df.copy()

    if include_primary_task_only:
        filtered = filtered[filtered["included_in_primary_task"] == True]

    if datasets is not None:
        filtered = filtered[filtered["dataset"].isin(datasets)]

    if labels is not None:
        filtered = filtered[filtered["normalized_label"].isin(labels)]

    if exclude_flags is not None:
        # Exclude samples that have any of the specified flags
        mask = pd.Series([True] * len(filtered), index=filtered.index)
        for flag in exclude_flags:
            has_flag = filtered["quality_flags"].apply(lambda x: flag in x if isinstance(x, list) else False)
            mask = mask & ~has_flag
        filtered = filtered[mask]

    logger.info(
        f"Filtered manifest: {len(df)} -> {len(filtered)} samples "
        f"(primary_task_only={include_primary_task_only}, "
        f"datasets={datasets}, labels={labels})"
    )
    return filtered


def get_label_distribution(df: pd.DataFrame) -> dict:
    """Get label distribution from manifest DataFrame.

    Returns:
        Dictionary with label counts.
    """
    return df["normalized_label"].value_counts().to_dict()


def get_dataset_distribution(df: pd.DataFrame) -> dict:
    """Get dataset distribution from manifest DataFrame.

    Returns:
        Dictionary with dataset counts.
    """
    return df["dataset"].value_counts().to_dict()
