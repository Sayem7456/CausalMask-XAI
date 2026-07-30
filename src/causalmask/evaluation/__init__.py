"""Evaluation subpackage — classification, calibration, faithfulness, CausalMask score, localization."""

from causalmask.evaluation.classification import (
    compute_classification_metrics,
    compute_youden_threshold,
    evaluate_from_predictions,
    load_predictions,
    save_metrics_json,
)
from causalmask.evaluation.calibration import (
    compute_ece,
    compute_calibration_metrics,
    evaluate_calibration_from_predictions,
    save_calibration_json,
)
from causalmask.evaluation.faithfulness import (
    raw_lesion_necessity,
    normalized_lesion_necessity,
    lesion_sufficiency,
    background_invariance,
    prediction_flip_rate,
    donor_stratified_invariance,
    lesion_vs_sham_difference,
    ensure_confidence_for_target,
    compute_per_sample_causal_metrics,
    insertion_auc,
    deletion_auc,
    compute_faithfulness_insertion_deletion,
)
from causalmask.evaluation.causalmask_score import (
    harmonic_mean,
    arithmetic_mean,
    geometric_mean,
    compute_causalmask_harmonic,
    compute_causalmask_arithmetic,
    compute_causalmask_geometric,
    compute_all_aggregations,
    compute_aggregation_sensitivity,
)
from causalmask.evaluation.localization import (
    attribution_mass_inside_mask,
    attribution_mass_inside_lesion,
    attribution_mass_inside_lesion_plus_margin,
    pointing_game_accuracy,
    soft_dice,
    saliency_iou,
    LocalizationResult,
    compute_localization_metrics,
    compute_localization_batch,
    DEFAULT_IOU_THRESHOLD,
)
