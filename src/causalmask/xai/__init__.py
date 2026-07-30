"""XAI subpackage — attribution methods, normalization, caching."""

from causalmask.xai.base import (
    AttributionInput,
    AttributionOutput,
    AttributionMetadata,
    resolve_target_layer,
    get_num_target_classes,
)
from causalmask.xai.normalization import (
    normalize_minmax_per_sample,
    normalize_percentile_per_sample,
    safe_normalize,
    AttributionCache,
    AttributionCacheEntry,
)
from causalmask.xai.gradcam import (
    GradCAM,
    GradCAMPlusPlus,
    build_gradcam,
    build_gradcam_plusplus,
)
from causalmask.xai.integrated_gradients import (
    IntegratedGradientsMethod,
    build_integrated_gradients,
)
from causalmask.xai.rise import (
    RISE,
    build_rise,
)

__all__ = [
    "AttributionInput",
    "AttributionOutput",
    "AttributionMetadata",
    "resolve_target_layer",
    "get_num_target_classes",
    "normalize_minmax_per_sample",
    "normalize_percentile_per_sample",
    "safe_normalize",
    "AttributionCache",
    "AttributionCacheEntry",
    "GradCAM",
    "GradCAMPlusPlus",
    "build_gradcam",
    "build_gradcam_plusplus",
    "IntegratedGradientsMethod",
    "build_integrated_gradients",
    "RISE",
    "build_rise",
]
