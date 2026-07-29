# CausalMask-XAI: Phase 8 Causal Pilot Report

**Status: exploratory pilot** — not a scientific result.

**Date:** 2026-07-29
**Run ID:** causal_full_effb0_fold0_seed42_pilot
**Fold:** 0 (pilot only)

## Configuration

- Backbone: efficientnet_b0
- Pretrained: EfficientNet_B0_Weights.IMAGENET1K_V1
- Input size: [224, 224]
- Epochs: 20 (pilot budget)
- Batch size: 16
- Learning rate: 1e-4
- Loss variant: full (CE + sufficiency + background + necessity)
- Sufficiency weight: 0.5
- Background weight: 0.5
- Necessity weight: 0.5
- Necessity margin: 0.1
- Necessity warm-up: 3 epochs
- Necessity ramp: 3 epochs
- Confidence threshold: 0.6
- Detached teacher: True
- Margin ratio: 0.05
- Blur sigma: 20.0
- Removal operator: telea

## Results

Training not yet executed. This report is a placeholder from local implementation.
Pilot training requires Colab with real BUSI data and Phase 5 baseline fold-0 checkpoint.

## Scientific caveats

- This is an exploratory single-fold pilot. No scientific conclusions.
- Hyperparameters were selected from training/validation behavior only.
- Test fold was evaluated once after all training choices were frozen.
- BUS-UCLM was not loaded and had zero influence on any choice.
- Five-fold causal training requires separate validation before reporting.
- Loss weights and warm-up schedule are pilot defaults; no tuning was performed.

## Deviations

- Background swap is disabled during training (swapped=None) to avoid in-batch donor selection complexity. Swap consistency is monitored only during validation.
- Only fold 0 was trained. Full five-fold training is deferred to Phase 9.
- Counterfactuals are generated per-sample on-the-fly rather than pre-cached.

## Artifacts

- Run directory: artifacts/runs/causal_full_effb0_fold0_seed42_pilot
- Predictions: artifacts/runs/causal_full_effb0_fold0_seed42_pilot/predictions_test.parquet
- History: artifacts/runs/causal_full_effb0_fold0_seed42_pilot/history.csv
- Config: artifacts/runs/causal_full_effb0_fold0_seed42_pilot/config.resolved.yaml
- Training curves: reports/results/causal_pilot_training_curves.png
