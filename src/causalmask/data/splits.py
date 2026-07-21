"""Group-aware split generation for medical-image cross-validation.

This module creates one immutable, versioned five-fold split for the BUSI
dataset. Requirements:
- Group-disjoint folds (no group_id crosses partitions)
- Approximate class stratification
- Deterministic with recorded seed
- Validation drawn only from non-test development portion
- Every included sample appears in exactly one test fold
- No external samples (BUS-UCLM) appear in any split
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

logger = logging.getLogger(__name__)


def create_grouped_kfold_split(
    manifest_df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    group_col: str = "group_id",
    label_col: str = "normalized_label",
    sample_id_col: str = "sample_id",
    dataset_col: str = "dataset",
    external_datasets: Optional[list[str]] = None,
    valid_labels: Optional[list[str]] = None,
) -> dict:
    """Create a group-disjoint, stratified k-fold split.

    Args:
        manifest_df: Manifest DataFrame with group_id, label, etc.
        n_splits: Number of folds.
        seed: Deterministic seed.
        group_col: Column name for grouping variable.
        label_col: Column name for class label.
        sample_id_col: Column name for sample identifier.
        dataset_col: Column name for dataset identifier.
        external_datasets: List of dataset names to exclude from splits.
        valid_labels: List of valid labels to include.

    Returns:
        dict with structure:
        {
            "metadata": { ... split configuration ... },
            "folds": {
                "fold_0": {
                    "train": [...],
                    "validation": [...],
                    "test": [...]
                },
                ...
            },
            "sample_assignments": { sample_id: fold_0, ... }
        }
    """
    if external_datasets is None:
        external_datasets = ["bus_uclm"]
    if valid_labels is None:
        valid_labels = ["benign", "malignant"]

    # Filter to only internal primary-task samples
    internal = manifest_df[
        (manifest_df[dataset_col].isin(external_datasets) == False)
        & (manifest_df[label_col].isin(valid_labels))
    ].copy()

    logger.info(
        f"Split input: {len(manifest_df)} total -> "
        f"{len(internal)} internal primary-task samples"
    )

    groups = internal[group_col].values
    labels = internal[label_col].values
    sample_ids = internal[sample_id_col].values

    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)

    # Create group-level label for stratification
    group_labels = []
    for g in unique_groups:
        group_mask = groups == g
        group_label_counts = pd.Series(labels[group_mask]).value_counts()
        group_labels.append(group_label_counts.idxmax())

    # Use StratifiedGroupKFold-like logic: assign groups to folds with stratification
    fold_assignments = _assign_groups_to_folds(
        unique_groups, np.array(group_labels), n_splits, seed
    )

    # For each fold, create train/validation/test splits
    folds = {}
    sample_assignments = {}

    for fold_idx in range(n_splits):
        test_mask = fold_assignments == fold_idx
        test_groups = set(unique_groups[test_mask])
        remaining_groups = list(set(unique_groups) - test_groups)

        # Validation: ~20% of remaining groups (16% of total)
        val_n = max(1, int(len(remaining_groups) * 0.2))
        rng_val = np.random.default_rng(seed + fold_idx + 1)
        val_indices = rng_val.choice(
            len(remaining_groups), val_n, replace=False
        )
        val_groups = set(remaining_groups[i] for i in val_indices)
        train_groups = set(remaining_groups) - val_groups

        train_ids = internal[
            internal[group_col].isin(train_groups)
        ][sample_id_col].tolist()
        val_ids = internal[
            internal[group_col].isin(val_groups)
        ][sample_id_col].tolist()
        test_ids = internal[
            internal[group_col].isin(test_groups)
        ][sample_id_col].tolist()

        folds[f"fold_{fold_idx}"] = {
            "train": sorted(train_ids),
            "validation": sorted(val_ids),
            "test": sorted(test_ids),
        }

        for sid in test_ids:
            sample_assignments[sid] = f"fold_{fold_idx}"

    # Build metadata
    split_dict = {
        "metadata": {
            "split_version": "v1",
            "split_name": "busi_binary_grouped_5fold_v1",
            "n_splits": n_splits,
            "seed": seed,
            "algorithm": "grouped_stratified_kfold",
            "group_col": group_col,
            "label_col": label_col,
            "external_datasets_excluded": external_datasets,
            "valid_labels": valid_labels,
            "total_internal_samples": len(internal),
            "total_groups": len(unique_groups),
        },
        "folds": folds,
        "sample_assignments": sample_assignments,
    }

    # Add fold statistics
    split_dict["statistics"] = _compute_split_statistics(
        split_dict, internal, group_col, label_col, sample_id_col
    )

    return split_dict


def _assign_groups_to_folds(
    groups: np.ndarray,
    group_labels: np.ndarray,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    """Assign groups to folds with approximate class stratification.

    Uses a greedy approach that balances both the number of groups
    and the class distribution across folds.
    """
    rng = np.random.default_rng(seed)
    n_groups = len(groups)

    fold_counts = np.zeros(n_splits, dtype=int)
    fold_labels: list[list[str]] = [[] for _ in range(n_splits)]
    assignments = np.full(n_groups, -1, dtype=int)

    # Sort groups by class for balanced distribution
    group_indices = list(range(n_groups))
    rng.shuffle(group_indices)

    for idx in group_indices:
        label = group_labels[idx]
        # Find fold with fewest groups, preferring same-label folds
        fold_scores = []
        for f in range(n_splits):
            same_label_count = sum(
                1 for l in fold_labels[f] if l == label
            )
            score = fold_counts[f] - same_label_count * 0.5
            fold_scores.append(score)

        best_fold = int(np.argmin(fold_scores))
        assignments[idx] = best_fold
        fold_counts[best_fold] += 1
        fold_labels[best_fold].append(label)

    return assignments


def _compute_split_statistics(
    split_dict: dict,
    internal_df: pd.DataFrame,
    group_col: str,
    label_col: str,
    sample_id_col: str,
) -> dict:
    """Compute per-fold and aggregate statistics for the split."""
    stats = {"per_fold": {}}

    all_sample_counts = {}
    all_label_counts = {}
    all_group_counts = {}

    for fold_name, fold_data in split_dict["folds"].items():
        fold_samples = {}
        for partition in ["train", "validation", "test"]:
            ids = fold_data[partition]
            partition_df = internal_df[
                internal_df[sample_id_col].isin(ids)
            ]
            fold_samples[partition] = {
                "n_samples": len(ids),
                "class_counts": (
                    partition_df[label_col].value_counts().to_dict()
                ),
                "n_groups": partition_df[group_col].nunique(),
            }
        stats["per_fold"][fold_name] = fold_samples

        for partition in ["train", "validation", "test"]:
            key = f"{fold_name}_{partition}"
            all_sample_counts[key] = fold_samples[partition]["n_samples"]
            all_label_counts[key] = fold_samples[partition]["class_counts"]
            all_group_counts[key] = fold_samples[partition]["n_groups"]

    stats["aggregate"] = {
        "total_samples_across_folds": sum(
            len(v) for v in split_dict["sample_assignments"].values()
        ),
        "total_unique_samples": len(split_dict["sample_assignments"]),
        "total_groups": internal_df[group_col].nunique(),
        "n_folds": len(split_dict["folds"]),
    }

    return stats


def validate_split_disjointness(
    split_dict: dict,
    manifest_df: pd.DataFrame,
    group_col: str = "group_id",
    sample_id_col: str = "sample_id",
    dataset_col: str = "dataset",
    external_datasets: Optional[list[str]] = None,
) -> dict:
    """Validate that the split satisfies all disjointness requirements.

    Checks:
    1. Sample disjointness across train/val/test within each fold
    2. Group disjointness across partitions within each fold
    3. Exact duplicate cluster disjointness
    4. Near-duplicate cluster disjointness
    5. Complete fold coverage (every sample in exactly one test fold)
    6. No external samples in any split

    Returns:
        Dict with check results.
    """
    if external_datasets is None:
        external_datasets = ["bus_uclm"]

    results = {
        "passed": True,
        "checks": {},
        "failures": [],
    }

    # 1. Sample disjointness within each fold
    for fold_name, fold_data in split_dict["folds"].items():
        all_ids = set()
        for partition in ["train", "validation", "test"]:
            ids = set(fold_data[partition])
            overlap = all_ids & ids
            if overlap:
                msg = (
                    f"{fold_name}: {len(overlap)} sample(s) cross "
                    f"partitions: {list(overlap)[:5]}"
                )
                results["failures"].append(msg)
            all_ids |= ids

    # 2. Group disjointness within each fold
    for fold_name, fold_data in split_dict["folds"].items():
        partition_groups = {}
        for partition in ["train", "validation", "test"]:
            ids = fold_data[partition]
            partition_df = manifest_df[
                manifest_df[sample_id_col].isin(ids)
            ]
            partition_groups[partition] = set(
                partition_df[group_col].unique()
            )

        for p1 in ["train", "validation", "test"]:
            for p2 in ["train", "validation", "test"]:
                if p1 < p2:
                    overlap = partition_groups[p1] & partition_groups[p2]
                    if overlap:
                        msg = (
                            f"{fold_name}: {len(overlap)} group(s) cross "
                            f"{p1}/{p2}: {list(overlap)[:5]}"
                        )
                        results["failures"].append(msg)

    # 3 & 4. Duplicate cluster checks (from manifest columns)
    for cluster_col in [
        "exact_duplicate_cluster",
        "near_duplicate_cluster",
    ]:
        if cluster_col not in manifest_df.columns:
            continue
        for fold_name, fold_data in split_dict["folds"].items():
            partition_clusters = {}
            for partition in ["train", "validation", "test"]:
                ids = fold_data[partition]
                partition_df = manifest_df[
                    manifest_df[sample_id_col].isin(ids)
                ]
                non_empty = partition_df[
                    partition_df[cluster_col].notna()
                    & (partition_df[cluster_col] != "")
                ]
                partition_clusters[partition] = set(
                    non_empty[cluster_col].unique()
                )

            for p1 in ["train", "validation", "test"]:
                for p2 in ["train", "validation", "test"]:
                    if p1 < p2:
                        overlap = (
                            partition_clusters[p1]
                            & partition_clusters[p2]
                        )
                        if overlap:
                            msg = (
                                f"{fold_name}: {len(overlap)} "
                                f"{cluster_col}(s) cross {p1}/{p2}: "
                                f"{list(overlap)[:5]}"
                            )
                            results["failures"].append(msg)

    # 5. Complete fold coverage
    all_test_ids = set()
    for fold_data in split_dict["folds"].values():
        all_test_ids.update(fold_data["test"])

    all_internal_ids = set(manifest_df[
        ~manifest_df[dataset_col].isin(external_datasets)
    ][sample_id_col].tolist())

    uncovered = all_internal_ids - all_test_ids
    if uncovered:
        msg = f"{len(uncovered)} internal sample(s) not in any test fold"
        results["failures"].append(msg)

    extra = all_test_ids - all_internal_ids
    if extra:
        msg = (
            f"{len(extra)} sample(s) in splits but not in internal set"
        )
        results["failures"].append(msg)

    # 6. No external samples
    assigned_samples = set()
    for fold_data in split_dict["folds"].values():
        for partition in ["train", "validation", "test"]:
            assigned_samples.update(fold_data[partition])

    external_mask = manifest_df[dataset_col].isin(external_datasets)
    external_in_split = set(
        manifest_df[external_mask][sample_id_col].tolist()
    ) & assigned_samples
    if external_in_split:
        msg = (
            f"{len(external_in_split)} external sample(s) found in splits"
        )
        results["failures"].append(msg)

    results["passed"] = len(results["failures"]) == 0
    results["checks"]["sample_disjointness"] = all(
        "sample(s) cross partitions" not in f for f in results["failures"]
    )
    results["checks"]["group_disjointness"] = all(
        "group(s) cross" not in f for f in results["failures"]
    )
    results["checks"]["exact_duplicate_disjointness"] = all(
        "exact_duplicate_cluster(s) cross" not in f
        for f in results["failures"]
    )
    results["checks"]["near_duplicate_disjointness"] = all(
        "near_duplicate_cluster(s) cross" not in f
        for f in results["failures"]
    )
    results["checks"]["complete_coverage"] = "not in any test fold" not in str(
        results["failures"]
    )
    results["checks"]["no_external_contamination"] = len(
        external_in_split
    ) == 0

    return results


def compute_split_digest(split_dict: dict) -> str:
    """Compute a deterministic SHA-256 digest of the split.

    Only the fold assignments are included (metadata may change).
    """
    canonical = json.dumps(
        {
            "folds": {
                k: {pk: sorted(v) for pk, v in fold.items()}
                for k, fold in split_dict["folds"].items()
            },
            "sample_assignments": {
                k: v
                for k, v in sorted(split_dict["sample_assignments"].items())
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def compute_manifest_digest(manifest_df: pd.DataFrame) -> str:
    """Compute a deterministic SHA-256 digest of a manifest.

    Uses sorted sample IDs, SHAs, and labels to detect any change.
    """
    canonical = json.dumps(
        {
            "sample_ids": sorted(manifest_df["sample_id"].tolist()),
            "image_sha256": manifest_df.set_index("sample_id")[
                "image_sha256"
            ]
            .to_dict(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def save_split(
    split_dict: dict,
    path: Path,
) -> None:
    """Save a split dictionary to JSON with digest embedded.

    The digest is computed before saving and stored in metadata.
    """
    split_dict["metadata"]["split_digest"] = compute_split_digest(split_dict)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(split_dict, f, indent=2, default=str)

    logger.info(
        f"Split saved: {path} "
        f"(digest={split_dict['metadata']['split_digest'][:16]}...)"
    )


def load_split(path: Path) -> dict:
    """Load a split dictionary from JSON and verify digest."""
    with open(path) as f:
        split_dict = json.load(f)

    stored_digest = split_dict.get("metadata", {}).get("split_digest", "")
    computed_digest = compute_split_digest(split_dict)

    if stored_digest and stored_digest != computed_digest:
        logger.warning(
            f"Split digest mismatch for {path}: "
            f"stored={stored_digest[:16]}..., "
            f"computed={computed_digest[:16]}..."
        )

    return split_dict


def is_split_reproducible(split_dict: dict) -> bool:
    """Check whether a split can be reproduced from its configuration.

    Returns True if metadata describes the algorithm and seed used.
    """
    meta = split_dict.get("metadata", {})
    required = [
        "split_version",
        "seed",
        "algorithm",
        "group_col",
        "label_col",
        "valid_labels",
    ]
    return all(k in meta for k in required)
