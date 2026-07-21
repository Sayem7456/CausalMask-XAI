"""Unit tests for causalmask.data.splits module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causalmask.data.splits import (
    create_grouped_kfold_split,
    validate_split_disjointness,
    compute_split_digest,
    compute_manifest_digest,
    save_split,
    load_split,
    is_split_reproducible,
)


def _make_manifest(n_samples=100, n_groups=50, seed=42):
    rng = np.random.default_rng(seed)
    group_ids = [f"G{i}" for i in range(n_groups)]
    groups = rng.choice(group_ids, n_samples)
    labels = rng.choice(["benign", "malignant"], n_samples, p=[0.67, 0.33])
    return pd.DataFrame(
        {
            "sample_id": [f"S{i}" for i in range(n_samples)],
            "group_id": groups,
            "normalized_label": labels,
            "dataset": ["busi"] * n_samples,
            "image_sha256": [
                hashlib.sha256(f"img{i}".encode()).hexdigest()
                for i in range(n_samples)
            ],
        }
    )


import hashlib


class TestCreateGroupedKfoldSplit:
    def test_returns_correct_folds(self):
        df = _make_manifest(100, 50)
        split = create_grouped_kfold_split(df)
        assert len(split["folds"]) == 5
        for fold_name in [f"fold_{i}" for i in range(5)]:
            assert fold_name in split["folds"]
            for p in ["train", "validation", "test"]:
                assert p in split["folds"][fold_name]
                assert len(split["folds"][fold_name][p]) > 0

    def test_all_samples_assigned(self):
        df = _make_manifest(100, 50)
        split = create_grouped_kfold_split(df)
        all_in_test = []
        for fold_data in split["folds"].values():
            all_in_test.extend(fold_data["test"])
        assert len(all_in_test) == len(
            df[df["normalized_label"].isin(["benign", "malignant"])]
        )

    def test_no_external_samples(self):
        df = _make_manifest(100, 50)
        df.loc[0, "dataset"] = "bus_uclm"
        split = create_grouped_kfold_split(df)
        for fold_data in split["folds"].values():
            for p in ["train", "validation", "test"]:
                assert df.loc[0, "sample_id"] not in fold_data[p]

    def test_deterministic_reproducibility(self):
        df = _make_manifest(200, 80)
        s1 = create_grouped_kfold_split(df, seed=42)
        s2 = create_grouped_kfold_split(df, seed=42)
        assert compute_split_digest(s1) == compute_split_digest(s2)

    def test_different_seed_changes_split(self):
        df = _make_manifest(200, 80)
        s1 = create_grouped_kfold_split(df, seed=42)
        s2 = create_grouped_kfold_split(df, seed=99)
        assert compute_split_digest(s1) != compute_split_digest(s2)

    def test_group_disjoint(self):
        df = _make_manifest(200, 80)
        split = create_grouped_kfold_split(df)
        validation = validate_split_disjointness(split, df)
        assert validation["passed"], validation["failures"]


class TestValidateSplitDisjointness:
    def test_clean_split_passes(self):
        df = _make_manifest(200, 80)
        split = create_grouped_kfold_split(df)
        result = validate_split_disjointness(split, df)
        assert result["passed"]
        assert len(result["failures"]) == 0

    def test_contaminated_split_fails(self):
        df = _make_manifest(200, 80)
        split = create_grouped_kfold_split(df)
        # Invalidate: add same sample to two partitions
        sid = list(split["folds"]["fold_0"]["test"])[0]
        split["folds"]["fold_0"]["train"].append(sid)
        result = validate_split_disjointness(split, df)
        assert not result["passed"]

    def test_external_contamination_detected(self):
        df = _make_manifest(200, 80)
        df.loc[0, "dataset"] = "bus_uclm"
        split = create_grouped_kfold_split(df)
        # manually add an external sample
        split["folds"]["fold_0"]["test"].append(df.loc[0, "sample_id"])
        result = validate_split_disjointness(split, df)
        assert not result["passed"]


class TestSplitDigest:
    def test_digest_deterministic(self):
        split = {"folds": {"fold_0": {"train": ["a"], "test": ["b"]}}, "sample_assignments": {"a": "fold_0"}}
        d1 = compute_split_digest(split)
        d2 = compute_split_digest(split)
        assert d1 == d2

    def test_different_splits_different_digests(self):
        s1 = {"folds": {"fold_0": {"train": ["a"], "test": ["b"]}}, "sample_assignments": {"a": "fold_0"}}
        s2 = {"folds": {"fold_0": {"train": ["c"], "test": ["d"]}}, "sample_assignments": {"c": "fold_0"}}
        assert compute_split_digest(s1) != compute_split_digest(s2)

    def test_manifest_digest(self):
        df = pd.DataFrame(
            {
                "sample_id": ["a", "b"],
                "image_sha256": ["x" * 64, "y" * 64],
            }
        )
        d1 = compute_manifest_digest(df)
        d2 = compute_manifest_digest(df)
        assert d1 == d2

    def test_manifest_digest_changes_with_data(self):
        df1 = pd.DataFrame(
            {"sample_id": ["a"], "image_sha256": ["x" * 64]}
        )
        df2 = pd.DataFrame(
            {"sample_id": ["a"], "image_sha256": ["z" * 64]}
        )
        assert compute_manifest_digest(df1) != compute_manifest_digest(df2)


class TestSaveLoadSplit:
    def test_save_and_load(self, tmp_path):
        split = {
            "metadata": {"split_version": "v1", "seed": 42},
            "folds": {"fold_0": {"train": ["a"], "test": ["b"]}},
            "sample_assignments": {"a": "fold_0", "b": "fold_0"},
            "statistics": {},
        }
        path = tmp_path / "split.json"
        save_split(split, path)
        assert path.exists()
        loaded = load_split(path)
        assert loaded["metadata"]["split_version"] == "v1"

    def test_digest_embedded(self, tmp_path):
        split = {
            "metadata": {"split_version": "v1", "seed": 42},
            "folds": {"fold_0": {"train": ["a"], "test": ["b"]}},
            "sample_assignments": {"a": "fold_0", "b": "fold_0"},
            "statistics": {},
        }
        path = tmp_path / "split.json"
        save_split(split, path)
        loaded = load_split(path)
        assert "split_digest" in loaded["metadata"]


class TestIsSplitReproducible:
    def test_valid_config(self):
        split = {
            "metadata": {
                "split_version": "v1",
                "seed": 42,
                "algorithm": "grouped_stratified_kfold",
                "group_col": "group_id",
                "label_col": "normalized_label",
                "valid_labels": ["benign", "malignant"],
            }
        }
        assert is_split_reproducible(split)

    def test_missing_config(self):
        split = {"metadata": {"split_version": "v1"}}
        assert not is_split_reproducible(split)
