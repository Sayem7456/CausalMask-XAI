# Implementation Plan

## Phase 0 — Environment and Repository Audit [COMPLETED]

- Project structure established
- Reproducibility helpers created
- Environment notebook created

## Phase 1 — Dataset Audit and Split Preparation

- Download and audit BUSI and BUS-UCLM datasets
- Near-duplicate detection
- Patient-aware stratified splitting
- Manifest generation

## Phase 2 — Baseline Training

- EfficientNet-B0 and ResNet-18 baselines
- Five-fold stratified cross-validation
- Evaluation metrics: AUROC, balanced accuracy, sensitivity, specificity, F1, ECE

## Phase 3 — XAI Methods

- Grad-CAM, Grad-CAM++, Integrated Gradients, RISE
- Sanity checks (model randomization)
- Robustness testing

## Phase 4 — Counterfactual Generation

- Lesion-sufficient images
- Lesion-removed images
- Background-swapped images
- Mask dilation ablation

## Phase 5 — CausalMask Score

- Necessity, sufficiency, invariance, localization
- Harmonic mean combination
- Score analysis and visualization

## Phase 6 — Causal Regularization

- Training with combined loss
- Hyperparameter selection
- Comparison against baselines

## Phase 7 — External Validation

- BUS-UCLM frozen test set
- Cross-dataset generalization

## Phase 8 — Ablations and Statistics

- Loss component ablations
- Mask dilation experiments
- Bootstrap CIs and statistical tests

## Phase 9 — Final Report

- Figures, tables, paper preparation
