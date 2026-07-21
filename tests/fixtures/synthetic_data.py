"""Synthetic data fixtures for smoke tests.

Produces small in-memory datasets with controlled properties.
No real images are required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SyntheticUltrasoundDataset(Dataset):
    """Small deterministic synthetic dataset for pipeline smoke tests.

    Generates fixed random images and masks so that outputs are reproducible.
    """

    def __init__(
        self,
        num_samples: int = 32,
        image_size: tuple[int, int] = (224, 224),
        num_classes: int = 2,
        seed: int = 42,
        transform=None,
        mask_transform=None,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.transform = transform
        self.mask_transform = mask_transform

        rng = np.random.default_rng(seed)
        self.images = rng.uniform(0, 1, size=(num_samples, 3, image_size[0], image_size[1]))
        self.masks = rng.integers(0, 2, size=(num_samples, 1, image_size[0], image_size[1]))
        self.labels = rng.integers(0, num_classes, size=num_samples)
        self.sample_ids = [f"synthetic_{i:04d}" for i in range(num_samples)]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        image = torch.from_numpy(self.images[idx]).float()
        mask = torch.from_numpy(self.masks[idx]).float()
        label = int(self.labels[idx])

        if self.transform is not None:
            image = self.transform(image)
        if self.mask_transform is not None and mask is not None:
            mask = self.mask_transform(mask)

        return {
            "image": image,
            "mask": mask,
            "label": label,
            "sample_id": self.sample_ids[idx],
        }


def create_synthetic_manifest(
    num_samples: int = 16,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a small synthetic manifest DataFrame for split testing."""
    rng = np.random.default_rng(seed)
    data = {
        "sample_id": [f"synthetic_{i:04d}" for i in range(num_samples)],
        "dataset": ["busi"] * num_samples,
        "normalized_label": rng.choice(["benign", "malignant"], size=num_samples).tolist(),
        "has_mask": [True] * num_samples,
        "image_sha256": [f"sha_{i:064d}" for i in range(num_samples)],
        "group_id": [f"group_{i // 2}" for i in range(num_samples)],
    }
    return pd.DataFrame(data)


def create_synthetic_split(num_samples: int = 8) -> dict:
    """Create a minimal synthetic split dict for testing."""
    all_ids = [f"synthetic_{i:04d}" for i in range(num_samples)]
    train = all_ids[: num_samples // 2]
    val = all_ids[num_samples // 2 : 3 * num_samples // 4]
    test = all_ids[3 * num_samples // 4 :]
    return {
        "metadata": {
            "split_version": "v1",
            "split_name": "synthetic_smoke",
            "n_splits": 5,
            "seed": 42,
            "algorithm": "test",
        },
        "folds": {
            "fold_0": {
                "train": train,
                "validation": val,
                "test": test,
            },
        },
        "sample_assignments": {sid: "fold_0" for sid in test},
    }
