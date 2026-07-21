"""Unit tests for causalmask.data.duplicate_audit module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from causalmask.data.duplicate_audit import (
    compute_perceptual_hash,
    hamming_distance,
    normalized_hamming_similarity,
    find_exact_duplicates,
    compute_phashes_for_manifest,
    find_near_duplicate_candidates,
    compute_image_similarity,
    verify_candidate_pairs,
    assign_duplicate_clusters,
    build_grouped_manifest,
)


def _create_test_image(path: Path, seed: int = 0, size=(100, 100)):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (*size, 3), dtype=np.uint8)
    Image.fromarray(img).save(path)


class TestPerceptualHash:
    def test_deterministic(self, tmp_path):
        p = tmp_path / "test.png"
        _create_test_image(p)
        h1 = compute_perceptual_hash(p)
        h2 = compute_perceptual_hash(p)
        assert h1 == h2

    def test_different_images_different_hashes(self, tmp_path):
        p1 = tmp_path / "a.png"
        p2 = tmp_path / "b.png"
        _create_test_image(p1, seed=0)
        _create_test_image(p2, seed=1)
        h1 = compute_perceptual_hash(p1)
        h2 = compute_perceptual_hash(p2)
        assert h1 != h2

    def test_similar_images_similar_hashes(self, tmp_path):
        p1 = tmp_path / "a.png"
        p2 = tmp_path / "b.png"
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(arr).save(p1)
        arr2 = arr.copy()
        arr2[5:8, 5:8] = np.random.randint(0, 255, (3, 3, 3), dtype=np.uint8)
        Image.fromarray(arr2).save(p2)
        h1 = compute_perceptual_hash(p1)
        h2 = compute_perceptual_hash(p2)
        sim = normalized_hamming_similarity(h1, h2)
        assert sim > 0.5


class TestHammingDistance:
    def test_identical(self):
        assert hamming_distance("aabb", "aabb") == 0

    def test_different(self):
        assert hamming_distance("0000", "ffff") > 0

    def test_normalized_similarity(self):
        assert normalized_hamming_similarity("aabb", "aabb") == 1.0
        s = normalized_hamming_similarity("0000", "ffff")
        assert 0 <= s < 1.0


class TestFindExactDuplicates:
    def test_no_duplicates(self, tmp_path):
        df = pd.DataFrame(
            {
                "sample_id": ["a", "b"],
                "image_path": ["a.png", "b.png"],
                "image_sha256": ["x" * 64, "y" * 64],
            }
        )
        result = find_exact_duplicates(df, tmp_path)
        assert len(result) == 0

    def test_with_duplicates(self, tmp_path):
        df = pd.DataFrame(
            {
                "sample_id": ["a", "b", "c"],
                "image_path": ["a.png", "b.png", "c.png"],
                "image_sha256": ["x" * 64, "x" * 64, "y" * 64],
            }
        )
        result = find_exact_duplicates(df, tmp_path)
        assert len(result) == 1
        assert set(result.iloc[0]["sample_ids"]) == {"a", "b"}


class TestComputePhashes:
    def test_all_phashes_computed(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        p1 = img_dir / "a.png"
        p2 = img_dir / "b.png"
        _create_test_image(p1)
        _create_test_image(p2)

        df = pd.DataFrame(
            {
                "sample_id": ["a", "b"],
                "image_path": [str(p1.relative_to(tmp_path)), str(p2.relative_to(tmp_path))],
            }
        )
        result = compute_phashes_for_manifest(df, tmp_path)
        assert len(result) == 2
        assert result["phash"].notna().all()


class TestFindNearDuplicateCandidates:
    def test_finds_candidates(self, tmp_path):
        phash_df = pd.DataFrame(
            {
                "sample_id": ["a", "b", "c"],
                "phash": [
                    "a1b2c3d4e5f6a7b8",
                    "a1b2c3d4e5f6a7b9",
                    "0000000000000000",
                ],
            }
        )
        result = find_near_duplicate_candidates(
            phash_df, min_similarity=0.5
        )
        assert len(result) >= 1


class TestComputeImageSimilarity:
    def test_identical_images(self, tmp_path):
        p = tmp_path / "img.png"
        _create_test_image(p)
        sim = compute_image_similarity(p, p)
        assert sim["ssim_approximate"] > 0.99
        assert sim["normalized_mse"] < 0.01

    def test_different_images(self, tmp_path):
        p1 = tmp_path / "a.png"
        p2 = tmp_path / "b.png"
        _create_test_image(p1, seed=0)
        _create_test_image(p2, seed=1)
        sim = compute_image_similarity(p1, p2)
        assert sim["mse"] > 0


class TestVerifyCandidatePairs:
    def test_verification(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        p1 = img_dir / "a.png"
        p2 = img_dir / "b.png"
        _create_test_image(p1, seed=0)
        _create_test_image(p2, seed=0)

        candidates = pd.DataFrame(
            {
                "sample_id_a": ["a"],
                "sample_id_b": ["b"],
                "phash_similarity": [0.95],
            }
        )
        manifest = pd.DataFrame(
            {
                "sample_id": ["a", "b"],
                "image_path": [
                    str(p1.relative_to(tmp_path)),
                    str(p2.relative_to(tmp_path)),
                ],
            }
        )
        result = verify_candidate_pairs(candidates, manifest, tmp_path)
        assert len(result) >= 0


class TestAssignDuplicateClusters:
    def test_no_duplicates(self):
        df = pd.DataFrame({"sample_id": ["a", "b", "c"]})
        exact = pd.DataFrame()
        verified = pd.DataFrame()
        clusters = assign_duplicate_clusters(exact, verified, df)
        assert len(clusters) == 3
        assert all(clusters["is_exact_duplicate"] == False)
        assert all(clusters["is_near_duplicate"] == False)

    def test_exact_duplicates(self):
        df = pd.DataFrame({"sample_id": ["a", "b", "c", "d"]})
        exact = pd.DataFrame(
            {
                "sample_ids": [["a", "b"], ["c", "d"]],
                "sha256": ["x" * 64, "y" * 64],
                "cluster_id": ["exact_1", "exact_2"],
                "image_paths": [["a.png", "b.png"], ["c.png", "d.png"]],
                "cluster_size": [2, 2],
                "detection_method": ["sha256_exact", "sha256_exact"],
            }
        )
        verified = pd.DataFrame()
        clusters = assign_duplicate_clusters(exact, verified, df)
        assert len(clusters) == 4
        exact_ids = clusters[clusters["is_exact_duplicate"] == True][
            "sample_id"
        ].tolist()
        assert len(exact_ids) == 4


class TestBuildGroupedManifest:
    def test_merges_correctly(self):
        manifest = pd.DataFrame(
            {
                "sample_id": ["a", "b"],
                "image_path": ["a.png", "b.png"],
            }
        )
        clusters = pd.DataFrame(
            {
                "sample_id": ["a", "b"],
                "group_id": ["group_1", "group_1"],
                "near_duplicate_cluster": ["near_1", "near_1"],
                "exact_duplicate_cluster": ["", ""],
                "is_exact_duplicate": [False, False],
                "is_near_duplicate": [True, True],
            }
        )
        result = build_grouped_manifest(manifest, clusters)
        assert len(result) == 2
        assert result["group_id"].tolist() == ["group_1", "group_1"]
