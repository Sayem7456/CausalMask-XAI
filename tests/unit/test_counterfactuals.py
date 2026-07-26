"""Unit tests for counterfactual modules (Phase 6)."""

import numpy as np
import pytest

from causalmask.counterfactuals.masks import (
    compute_lesion_bbox,
    dilate_mask,
    lesion_plus_margin,
    lesion_plus_margin_feathered,
    MarginConfig,
)
from causalmask.counterfactuals.sufficient import (
    generate_lesion_sufficient,
    SufficientConfig,
)
from causalmask.counterfactuals.removal import (
    generate_lesion_removed,
    RemovalConfig,
    RemovalOperator,
)
from causalmask.counterfactuals.background_swap import (
    generate_background_swap,
    SwapConfig,
    _select_donor,
)
from causalmask.counterfactuals.controls import (
    generate_random_region_removal,
    generate_random_region_preservation,
    generate_shifted_mask_control,
    sham_mask_area,
    ControlsConfig,
)
from causalmask.counterfactuals.quality import (
    compute_quality_metrics,
    QualityMetrics,
    _compute_cache_key,
    _compute_config_digest,
    _js_divergence,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _image(h=128, w=128, c=3) -> np.ndarray:
    rng = np.random.default_rng(42)
    return (rng.uniform(0, 255, size=(h, w, c))).astype(np.uint8)


def _mask(h=128, w=128, size=20) -> np.ndarray:
    """Binary mask with a square lesion at centre."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    mask[cy - size:cy + size, cx - size:cx + size] = 1
    return mask


def _tiny_mask(h=128, w=128) -> np.ndarray:
    """Binary mask with a 2x2 pixel lesion."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[63:65, 63:65] = 1
    return mask


def _border_mask(h=128, w=128) -> np.ndarray:
    """Binary mask with a lesion touching the border."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[0:10, 50:70] = 1
    return mask


def _empty_mask(h=128, w=128) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------

class TestComputeLesionBbox:
    def test_centred_lesion(self):
        mask = _mask(128, 128, 20)
        x_min, y_min, x_max, y_max = compute_lesion_bbox(mask)
        assert x_min == 44
        assert y_min == 44
        assert x_max == 84
        assert y_max == 84

    def test_empty_mask_returns_full_bounds(self):
        mask = _empty_mask(64, 64)
        x_min, y_min, x_max, y_max = compute_lesion_bbox(mask)
        assert (x_min, y_min) == (0, 0)
        assert (x_max, y_max) == (64, 64)

    def test_border_touching(self):
        mask = _border_mask(128, 128)
        x_min, y_min, x_max, y_max = compute_lesion_bbox(mask)
        assert y_min == 0
        assert y_max == 10
        assert x_min == 50
        assert x_max == 70

    def test_3d_mask(self):
        mask_3d = np.zeros((128, 128, 3), dtype=np.uint8)
        mask_3d[50:70, 50:70, :] = 1
        x_min, y_min, x_max, y_max = compute_lesion_bbox(mask_3d)
        assert (x_min, y_min) == (50, 50)
        assert (x_max, y_max) == (70, 70)


class TestDilateMask:
    def test_dilate_enlarges(self):
        mask = _mask(128, 128, 10)
        orig = mask.sum()
        dilated = dilate_mask(mask, 3)
        assert dilated.sum() > orig

    def test_zero_kernel_is_identity(self):
        mask = _mask(128, 128, 10)
        dilated = dilate_mask(mask, 0)
        assert np.array_equal(mask, dilated)

    def test_tiny_lesion_survives(self):
        mask = _tiny_mask(128, 128)
        dilated = dilate_mask(mask, 2)
        assert dilated.sum() > 0


class TestLesionPlusMargin:
    def test_margin_zero_is_identity(self):
        mask = _mask(128, 128, 10)
        mplus = lesion_plus_margin(mask, MarginConfig(margin_ratio=0.0, min_kernel=0))
        assert np.array_equal(mask.astype(np.uint8), mplus.astype(np.uint8))

    def test_margin_increases_area(self):
        mask = _mask(128, 128, 10)
        m0 = lesion_plus_margin(mask, MarginConfig(margin_ratio=0.0)).sum()
        m5 = lesion_plus_margin(mask, MarginConfig(margin_ratio=0.05)).sum()
        m10 = lesion_plus_margin(mask, MarginConfig(margin_ratio=0.10)).sum()
        m20 = lesion_plus_margin(mask, MarginConfig(margin_ratio=0.20)).sum()
        assert m5 >= m0
        assert m10 >= m5
        assert m20 >= m10

    def test_clip_to_image_shape(self):
        mask = _mask(128, 128, 10)
        mplus = lesion_plus_margin(mask, MarginConfig(margin_ratio=0.5), (128, 128))
        assert mplus.shape == (128, 128)
        assert mplus.max() <= 1

    def test_tiny_lesion(self):
        mask = _tiny_mask(128, 128)
        mplus = lesion_plus_margin(mask, MarginConfig(margin_ratio=0.05))
        assert mplus.shape == (128, 128)
        assert mplus.sum() > 0

    def test_border_lesion(self):
        mask = _border_mask(128, 128)
        mplus = lesion_plus_margin(mask, MarginConfig(margin_ratio=0.10))
        assert mplus.shape == (128, 128)

    def test_feathered_output_range(self):
        mask = _mask(128, 128, 10)
        alpha = lesion_plus_margin_feathered(
            mask, MarginConfig(margin_ratio=0.05, feathered_blend_px=5)
        )
        assert alpha.max() <= 1.0
        assert alpha.min() >= 0.0


# ---------------------------------------------------------------------------
# sufficient
# ---------------------------------------------------------------------------

class TestGenerateLesionSufficient:
    def test_output_shape(self):
        img = _image(128, 128)
        mask = _mask(128, 128, 10)
        result, mplus = generate_lesion_sufficient(img, mask)
        assert result.shape == img.shape
        assert mplus.shape == (128, 128)

    def test_lesion_pixels_preserved(self):
        """Pixels inside M⁺ must remain within 1 intensity unit."""
        img = _image(128, 128)
        mask = _mask(128, 128, 10)
        result, mplus = generate_lesion_sufficient(
            img, mask,
            SufficientConfig(margin_config=MarginConfig(margin_ratio=0.0),
                             use_feathered_blend=False),
        )
        inside = mplus > 0
        diff = np.abs(img.astype(np.float32) - result.astype(np.float32))
        assert diff[inside].max() <= 1.0, f"max diff inside = {diff[inside].max()}"

    def test_exterior_changes(self):
        img = _image(128, 128)
        mask = _mask(128, 128, 10)
        result, mplus = generate_lesion_sufficient(img, mask)
        outside = ~ (mplus > 0)
        diff = np.abs(img.astype(np.float32) - result.astype(np.float32))
        mean_change = diff[outside].mean()
        assert mean_change > 0.0

    def test_output_is_finite(self):
        img = _image(128, 128)
        mask = _mask(128, 128, 10)
        result, _ = generate_lesion_sufficient(img, mask)
        assert np.isfinite(result).all()

    def test_output_in_uint8_range(self):
        img = _image(128, 128)
        mask = _mask(128, 128, 10)
        result, _ = generate_lesion_sufficient(img, mask)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_default_config(self):
        img = _image(128, 128)
        mask = _mask(128, 128, 10)
        result, mplus = generate_lesion_sufficient(img, mask)
        assert result is not None
        assert mplus is not None


# ---------------------------------------------------------------------------
# removal
# ---------------------------------------------------------------------------

class TestGenerateLesionRemoved:
    def test_output_shape_telea(self):
        img = _image(128, 128)
        mask = _mask(128, 128, 10)
        result, mplus = generate_lesion_removed(img, mask)
        assert result.shape == img.shape
        assert mplus.shape == (128, 128)

    def test_output_shape_ns(self):
        img = _image(128, 128)
        mask = _mask(128, 128, 10)
        config = RemovalConfig(operator=RemovalOperator.NAVIER_STOKES)
        result, mplus = generate_lesion_removed(img, mask, config)
        assert result.shape == img.shape

    def test_removed_region_changes(self):
        img = _image(128, 128)
        mask = _mask(128, 128, 10)
        result, mplus = generate_lesion_removed(
            img, mask,
            RemovalConfig(margin_config=MarginConfig(margin_ratio=0.0)),
        )
        inside = mplus > 0
        diff = np.abs(img.astype(np.float32) - result.astype(np.float32))
        assert diff[inside].mean() > 0.0

    def test_output_is_finite(self):
        img = _image(128, 128)
        mask = _mask(128, 128, 10)
        result, _ = generate_lesion_removed(img, mask)
        assert np.isfinite(result).all()

    def test_output_in_uint8_range(self):
        img = _image(128, 128)
        mask = _mask(128, 128, 10)
        result, _ = generate_lesion_removed(img, mask)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_tiny_lesion(self):
        img = _image(128, 128)
        mask = _tiny_mask(128, 128)
        result, mplus = generate_lesion_removed(img, mask)
        assert result.shape == img.shape

    def test_border_lesion(self):
        img = _image(128, 128)
        mask = _border_mask(128, 128)
        result, mplus = generate_lesion_removed(img, mask)
        assert result.shape == img.shape

    def test_both_operators(self):
        img = _image(128, 128)
        mask = _mask(128, 128, 10)
        for op in RemovalOperator:
            config = RemovalConfig(operator=op)
            result, _ = generate_lesion_removed(img, mask, config)
            assert np.isfinite(result).all()


# ---------------------------------------------------------------------------
# background_swap
# ---------------------------------------------------------------------------

class TestDonorSelection:
    def test_same_class(self):
        candidates = [
            {"sample_id": "s1", "normalized_label": "benign"},
            {"sample_id": "s2", "normalized_label": "benign"},
            {"sample_id": "s3", "normalized_label": "malignant"},
        ]
        rng = np.random.default_rng(42)
        config = SwapConfig(donor_class="same", seed=42)
        donor = _select_donor("s1", "benign", candidates, config, rng)
        assert donor is not None
        assert donor["sample_id"] == "s2"
        assert donor["normalized_label"] == "benign"

    def test_opposite_class(self):
        candidates = [
            {"sample_id": "s1", "normalized_label": "benign"},
            {"sample_id": "s2", "normalized_label": "benign"},
            {"sample_id": "s3", "normalized_label": "malignant"},
        ]
        rng = np.random.default_rng(42)
        config = SwapConfig(donor_class="opposite", seed=42)
        donor = _select_donor("s1", "benign", candidates, config, rng)
        assert donor is not None
        assert donor["normalized_label"] == "malignant"

    def test_no_self_donation(self):
        candidates = [
            {"sample_id": "s1", "normalized_label": "benign"},
        ]
        rng = np.random.default_rng(42)
        config = SwapConfig(donor_class="same", seed=42)
        donor = _select_donor("s1", "benign", candidates, config, rng)
        assert donor is None

    def test_no_matching_donor(self):
        candidates = [
            {"sample_id": "s2", "normalized_label": "malignant"},
        ]
        rng = np.random.default_rng(42)
        config = SwapConfig(donor_class="same", seed=42)
        donor = _select_donor("s1", "benign", candidates, config, rng)
        assert donor is None


class TestGenerateBackgroundSwap:
    def test_output_shape(self):
        img = _image(128, 128)
        mask = _mask(128, 128, 10)
        donor = _image(128, 128)
        result, mplus = generate_background_swap(img, mask, donor)
        assert result.shape == img.shape
        assert mplus.shape == (128, 128)

    def test_lesion_pixels_preserved(self):
        img = _image(128, 128)
        mask = _mask(128, 128, 10)
        donor = _image(128, 128)
        result, mplus = generate_background_swap(
            img, mask, donor,
            SwapConfig(margin_config=MarginConfig(margin_ratio=0.0),
                       use_feathered_blend=False),
        )
        inside = mplus > 0
        diff = np.abs(img.astype(np.float32) - result.astype(np.float32))
        assert diff[inside].max() <= 1.0, f"max diff inside lesion = {diff[inside].max()}"

    def test_exterior_changes(self):
        img = _image(128, 128)
        mask = _mask(128, 128, 10)
        donor = np.full_like(img, 128, dtype=np.uint8)
        result, mplus = generate_background_swap(img, mask, donor)
        outside = ~ (mplus > 0)
        diff = np.abs(img.astype(np.float32) - result.astype(np.float32))
        assert diff[outside].mean() > 0.0

    def test_output_in_uint8_range(self):
        img = _image(128, 128)
        mask = _mask(128, 128, 10)
        donor = _image(128, 128)
        result, _ = generate_background_swap(img, mask, donor)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_finite_output(self):
        img = _image(128, 128)
        mask = _mask(128, 128, 10)
        donor = _image(128, 128)
        result, _ = generate_background_swap(img, mask, donor)
        assert np.isfinite(result).all()


# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------

class TestShamMaskArea:
    def test_matches_foreground_pixels(self):
        mask = _mask(128, 128, 10)
        area = sham_mask_area(mask)
        expected = 20 * 20  # size=10 gives 20x20 square at centre
        assert area == expected

    def test_empty_mask(self):
        area = sham_mask_area(_empty_mask(128, 128))
        assert area == 0


class TestRandomRegionRemoval:
    def test_output_shape(self):
        img = _image(128, 128)
        mask = _mask(128, 128, 10)
        result, ctrl_mask, actual_area = generate_random_region_removal(
            img, mask, ControlsConfig(seed=42)
        )
        assert result.shape == img.shape
        assert ctrl_mask.shape == (128, 128)
        assert actual_area > 0

    def test_output_in_uint8_range(self):
        img = _image(128, 128)
        mask = _mask(128, 128, 10)
        result, _, _ = generate_random_region_removal(img, mask, ControlsConfig(seed=42))
        assert result.min() >= 0
        assert result.max() <= 255


class TestRandomRegionPreservation:
    def test_output_shape(self):
        img = _image(128, 128)
        mask = _mask(128, 128, 10)
        result, ctrl_mask, actual_area = generate_random_region_preservation(
            img, mask, ControlsConfig(seed=42)
        )
        assert result.shape == img.shape
        assert actual_area > 0

    def test_output_in_uint8_range(self):
        img = _image(128, 128)
        mask = _mask(128, 128, 10)
        result, _, _ = generate_random_region_preservation(
            img, mask, ControlsConfig(seed=42)
        )
        assert result.min() >= 0
        assert result.max() <= 255


class TestShiftedMaskControl:
    def test_output_shape(self):
        img = _image(128, 128)
        mask = _mask(128, 128, 10)
        result, shifted_mask, info = generate_shifted_mask_control(
            img, mask, ControlsConfig(seed=42)
        )
        assert result.shape == img.shape
        assert shifted_mask.shape == (128, 128)
        assert "overlap_iou" in info

    def test_output_in_uint8_range(self):
        img = _image(128, 128)
        mask = _mask(128, 128, 10)
        result, _, _ = generate_shifted_mask_control(
            img, mask, ControlsConfig(seed=42)
        )
        assert result.min() >= 0
        assert result.max() <= 255


# ---------------------------------------------------------------------------
# quality
# ---------------------------------------------------------------------------

class TestCacheKey:
    def test_deterministic(self):
        k1 = _compute_cache_key("s1", "md", "sd", "op", 0.05, "d1", 42, "cd")
        k2 = _compute_cache_key("s1", "md", "sd", "op", 0.05, "d1", 42, "cd")
        assert k1 == k2

    def test_different_input_yields_different_key(self):
        k1 = _compute_cache_key("s1", "md", "sd", "op", 0.05, "d1", 42, "cd")
        k2 = _compute_cache_key("s2", "md", "sd", "op", 0.05, "d1", 42, "cd")
        assert k1 != k2


class TestConfigDigest:
    def test_identical_configs(self):
        d1 = _compute_config_digest({"a": 1, "b": 2})
        d2 = _compute_config_digest({"b": 2, "a": 1})
        assert d1 == d2

    def test_different_configs(self):
        d1 = _compute_config_digest({"a": 1})
        d2 = _compute_config_digest({"a": 2})
        assert d1 != d2


class TestJSDivergence:
    def test_identical_distributions(self):
        p = np.array([0.5, 0.5])
        d = _js_divergence(p, p)
        assert d < 1e-10

    def test_different_distributions(self):
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        d = _js_divergence(p, q)
        assert d > 0.0


class TestComputeQualityMetrics:
    @pytest.fixture
    def img(self):
        return _image(128, 128)

    @pytest.fixture
    def mask(self):
        return _mask(128, 128, 10)

    def test_identical_no_change(self, img, mask):
        mplus = mask
        m = compute_quality_metrics(img, img, mplus, "s1", "sufficient", 0.05)
        assert m.operator_failed is False
        assert m.changed_pixel_fraction == 0.0
        assert m.lesion_preservation_error == 0.0
        assert m.output_is_finite is True

    def test_finite_output(self, img, mask):
        m = compute_quality_metrics(img, img, mask, "s1", "sufficient", 0.05)
        assert m.output_is_finite is True

    def test_intensity_range(self, img, mask):
        m = compute_quality_metrics(img, img, mask, "s1", "sufficient", 0.05)
        assert m.intensity_in_range is True

    def test_different_images_detected(self, img, mask):
        rng = np.random.default_rng(99)
        cf = (rng.uniform(0, 255, size=img.shape)).astype(np.uint8)
        m = compute_quality_metrics(img, cf, mask, "s1", "sufficient", 0.05)
        assert m.changed_pixel_fraction > 0.0

    def test_nan_detected(self, img, mask):
        bad = img.astype(np.float32).copy()
        bad[64, 64, 0] = np.nan
        m = compute_quality_metrics(img, bad, mask, "s1", "op", 0.05)
        assert m.operator_failed is True
        assert not m.output_is_finite

    def test_output_shape_recorded(self, img, mask):
        m = compute_quality_metrics(img, img, mask, "s1", "sufficient", 0.05)
        assert m.output_shape == (128, 128, 3)

    def test_sham_area_match_recorded(self, img, mask):
        m = compute_quality_metrics(img, img, mask, "s1", "sham_removal", 0.05,
                                     sham_area_match=0.98)
        assert m.sham_area_match == 0.98
