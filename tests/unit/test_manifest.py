"""Unit tests for causalmask.data.manifest module."""

import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from causalmask.data.manifest import (
    compute_sha256,
    normalize_label,
    generate_sample_id,
    compute_mask_area_fraction,
    detect_quality_flags,
    discover_busi_files,
    discover_bus_uclm_files,
    create_sample_record,
    validate_manifest,
    SampleRecord,
)


class TestNormalizeLabel:
    """Tests for label normalization."""

    def test_benign_variations(self):
        assert normalize_label("benign") == "benign"
        assert normalize_label("Benign") == "benign"
        assert normalize_label("BENIGN") == "benign"

    def test_malignant_variations(self):
        assert normalize_label("malignant") == "malignant"
        assert normalize_label("Malignant") == "malignant"
        assert normalize_label("MALIGNANT") == "malignant"

    def test_normal_variations(self):
        assert normalize_label("normal") == "normal"
        assert normalize_label("Normal") == "normal"
        assert normalize_label("NORMAL") == "normal"

    def test_unknown_label(self):
        # Unknown labels are lowercased
        assert normalize_label("unknown") == "unknown"
        assert normalize_label("Suspicious") == "suspicious"


class TestGenerateSampleId:
    """Tests for stable sample ID generation."""

    def test_basic_generation(self):
        sid = generate_sample_id("busi", "benign", "image_001")
        assert sid == "busi_benign_image_001"

    def test_special_characters(self):
        sid = generate_sample_id("busi", "benign", "image-001 (1)")
        # Hyphens are allowed, parentheses and spaces are replaced
        assert sid == "busi_benign_image-001__1_"

    def test_deterministic(self):
        id1 = generate_sample_id("busi", "benign", "image_001")
        id2 = generate_sample_id("busi", "benign", "image_001")
        assert id1 == id2


class TestComputeMaskAreaFraction:
    """Tests for mask area fraction computation."""

    def test_empty_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        assert compute_mask_area_fraction(mask) == 0.0

    def test_full_mask(self):
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        assert compute_mask_area_fraction(mask) == 1.0

    def test_half_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[:50, :] = 255
        assert abs(compute_mask_area_fraction(mask) - 0.5) < 0.01

    def test_small_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[45:55, 45:55] = 255  # 10x10 = 100 pixels out of 10000
        assert abs(compute_mask_area_fraction(mask) - 0.01) < 0.001


class TestDetectQualityFlags:
    """Tests for quality flag detection."""

    def test_normal_image(self):
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        flags = detect_quality_flags(img, None, Path("test.png"))
        assert "very_small_image" not in flags
        assert "very_large_image" not in flags

    def test_very_small_image(self):
        img = np.random.randint(0, 255, (30, 30, 3), dtype=np.uint8)
        flags = detect_quality_flags(img, None, Path("test.png"))
        assert "very_small_image" in flags

    def test_extreme_aspect_ratio(self):
        img = np.random.randint(0, 255, (100, 1000, 3), dtype=np.uint8)
        flags = detect_quality_flags(img, None, Path("test.png"))
        assert "extreme_aspect_ratio" in flags

    def test_empty_mask_flag(self):
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        mask = np.zeros((200, 200), dtype=np.uint8)
        flags = detect_quality_flags(img, mask, Path("test.png"))
        assert "empty_mask" in flags

    def test_very_small_mask_flag(self):
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[95:105, 95:105] = 255  # Very small mask
        flags = detect_quality_flags(img, mask, Path("test.png"))
        assert "very_small_mask" in flags


class TestComputeSha256:
    """Tests for SHA-256 computation."""

    def test_deterministic(self):
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".txt") as f:
            f.write(b"test content")
            f.flush()
            hash1 = compute_sha256(Path(f.name))
            hash2 = compute_sha256(Path(f.name))
            assert hash1 == hash2

    def test_different_content(self):
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".txt") as f1, \
             tempfile.NamedTemporaryFile(mode="wb", suffix=".txt") as f2:
            f1.write(b"content 1")
            f1.flush()
            f2.write(b"content 2")
            f2.flush()
            hash1 = compute_sha256(Path(f1.name))
            hash2 = compute_sha256(Path(f2.name))
            assert hash1 != hash2


class TestCreateSampleRecord:
    """Tests for sample record creation."""

    def test_basic_record(self, tmp_path):
        # Create test image and mask
        img_dir = tmp_path / "benign"
        img_dir.mkdir()
        
        img_path = img_dir / "test_image.png"
        mask_path = img_dir / "test_image_mask.png"
        
        # Create simple test images
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(img_path)
        
        mask = Image.fromarray(np.random.randint(0, 255, (100, 100), dtype=np.uint8))
        mask.save(mask_path)
        
        sample_info = {
            "raw_label": "Benign",
            "normalized_label": "benign",
            "image_path": img_path,
            "mask_paths": [mask_path],
            "dataset": "busi",
        }
        
        record = create_sample_record(sample_info, tmp_path, patient_id_prefix="test_")
        
        assert record.dataset == "busi"
        assert record.normalized_label == "benign"
        assert record.included_in_primary_task is True
        assert record.has_mask is True
        assert record.image_width == 100
        assert record.image_height == 100
        assert record.channels == 3
        assert len(record.image_sha256) == 64
        assert len(record.mask_sha256) == 64

    def test_normal_image_excluded(self, tmp_path):
        img_dir = tmp_path / "normal"
        img_dir.mkdir()
        
        img_path = img_dir / "test_normal.png"
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(img_path)
        
        sample_info = {
            "raw_label": "Normal",
            "normalized_label": "normal",
            "image_path": img_path,
            "mask_paths": [],
            "dataset": "busi",
        }
        
        record = create_sample_record(sample_info, tmp_path)
        
        assert record.normalized_label == "normal"
        assert record.included_in_primary_task is False
        assert record.has_mask is False


class TestValidateManifest:
    """Tests for manifest validation."""

    def test_clean_manifest(self):
        records = [
            SampleRecord(
                sample_id="busi_benign_001",
                dataset="busi",
                image_path="data/raw/extracted/busi/benign/001.png",
                mask_path="data/raw/extracted/busi/benign/001_mask.png",
                raw_label="Benign",
                normalized_label="benign",
                included_in_primary_task=True,
                patient_id="p001",
                provisional_group_id="p001",
                image_width=100,
                image_height=100,
                channels=3,
                image_sha256="a" * 64,
                mask_sha256="b" * 64,
                mask_area_fraction=0.1,
                has_mask=True,
                quality_flags=[],
            ),
            SampleRecord(
                sample_id="busi_malignant_002",
                dataset="busi",
                image_path="data/raw/extracted/busi/malignant/002.png",
                mask_path="data/raw/extracted/busi/malignant/002_mask.png",
                raw_label="Malignant",
                normalized_label="malignant",
                included_in_primary_task=True,
                patient_id="p002",
                provisional_group_id="p002",
                image_width=100,
                image_height=100,
                channels=3,
                image_sha256="c" * 64,
                mask_sha256="d" * 64,
                mask_area_fraction=0.15,
                has_mask=True,
                quality_flags=[],
            ),
        ]
        
        summary = validate_manifest(records)
        
        assert summary["total_samples"] == 2
        assert summary["issue_counts"]["duplicate_sample_ids"] == 0
        assert summary["issue_counts"]["duplicate_image_paths"] == 0

    def test_duplicate_sample_ids(self):
        records = [
            SampleRecord(
                sample_id="busi_benign_001",
                dataset="busi",
                image_path="data/raw/extracted/busi/benign/001.png",
                mask_path="",
                raw_label="Benign",
                normalized_label="benign",
                included_in_primary_task=True,
                patient_id="p001",
                provisional_group_id="p001",
                image_width=100,
                image_height=100,
                channels=3,
                image_sha256="a" * 64,
                mask_sha256="",
                mask_area_fraction=0.0,
                has_mask=False,
                quality_flags=[],
            ),
            SampleRecord(
                sample_id="busi_benign_001",  # Duplicate
                dataset="busi",
                image_path="data/raw/extracted/busi/benign/002.png",
                mask_path="",
                raw_label="Benign",
                normalized_label="benign",
                included_in_primary_task=True,
                patient_id="p002",
                provisional_group_id="p002",
                image_width=100,
                image_height=100,
                channels=3,
                image_sha256="b" * 64,
                mask_sha256="",
                mask_area_fraction=0.0,
                has_mask=False,
                quality_flags=[],
            ),
        ]
        
        summary = validate_manifest(records)
        
        assert summary["issue_counts"]["duplicate_sample_ids"] == 1


class TestDiscoverBusiFiles:
    """Tests for BUSI file discovery."""

    def test_discover_basic_structure(self, tmp_path):
        # Create BUSI-like structure
        busi_dir = tmp_path / "Dataset_BUSI_with_GT"
        busi_dir.mkdir()
        
        benign_dir = busi_dir / "benign"
        benign_dir.mkdir()
        
        # Create test images
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(benign_dir / "benign_001.png")
        img.save(benign_dir / "benign_001_mask.png")
        
        samples = discover_busi_files(tmp_path)
        
        assert len(samples) == 1
        assert samples[0]["normalized_label"] == "benign"
        assert len(samples[0]["mask_paths"]) == 1
