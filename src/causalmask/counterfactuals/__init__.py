"""Counterfactual interventions for causal auditing of medical-image classifiers.

This package produces lesion-conditioned counterfactual images
(lesion-sufficient, lesion-removed, background-swapped, sham controls)
and audits their quality — never claiming anatomical realism.
"""

from causalmask.counterfactuals.masks import (
    compute_lesion_bbox,
    dilate_mask,
    lesion_plus_margin,
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
    build_audit_grid,
    QualityMetrics,
    AuditConfig,
    save_quality_metrics,
    load_quality_metrics,
    generate_quality_report,
)
