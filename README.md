# CausalMask-XAI

Can we distinguish a genuinely lesion-focused medical classifier from one that produces visually attractive explanations while relying on background texture, scanner artefacts, or other shortcuts?

This project implements a **causal auditing and training framework** for breast-ultrasound classification. It tests whether explanations correspond to lesion-dependent decisions using real lesion masks, counterfactual image generation, and a composite CausalMask Score.

## Current state

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Environment and repository audit | Done |
| 1 | Download and extract datasets | Done |
| 2 | Manifest and quality audit | Done |
| 3 | Duplicate audit and fixed group-disjoint 5-fold splits | Done |
| 4 | Baseline pipeline smoke test | Passed |
| 5 | Baseline EfficientNet-B0 five-fold CV | Done — validated |
| 6 | Counterfactual engine and quality audit | Done — validated |
| 7 | CausalMask metrics on frozen baseline models | Done — executed |
| 8 | Causal regularization one-fold pilot | Done — executed |
| 9 | XAI methods and faithfulness evaluation | Done — executed |
| 10 | Causal five-fold cross-validation | Done — validated |
| **11** | **Robustness, sanity checks, and ablations** | **Done — executed** |
| **11b** | **Ablation scientific evidence (five-fold OOF)** | **Done — executed** |
| 12 | External validation (BUS-UCLM) | Planned |

### Phase 7: CausalMask metrics on baseline

- 544 samples processed across 5 folds with causal component metrics
- 3 margins, 2 removal operators, 2 donor classes, 2 target definitions
- Sham controls, composite CausalMask score computed
- No model retrained — frozen baseline checkpoints only

### Phase 8: Causal regularization pilot

- One-fold pilot (fold 0) with full causal training objective
- CE + sufficiency + background + gated necessity ranking
- Config frozen as `reports/results/frozen_causal_configuration.yaml`
- Results labelled exploratory

### Phase 9: XAI methods and faithfulness

- 4 XAI methods: Grad-CAM, Grad-CAM++, Integrated Gradients (50 steps), RISE (1000 masks)
- 101 test samples per method on fold 0 (baseline + causal pilot)
- Localization and faithfulness metrics (insertion/deletion AUC, soft Dice, IoU)

### Phase 10: Causal five-fold cross-validation

- All 5 folds validated using frozen Phase 8 configuration
- Paired comparison against baseline (Wilcoxon p=0.0000)
- Denominator reconciliation: baseline=647, causal=647 (verified)
- Donor-leakage tests pass across all 5 folds

### Phase 11: Robustness, sanity checks, and ablations

- **Explanation robustness** evaluated on 5 diagnosis-preserving transforms
  - Horizontal flip, contrast, gamma, translation, speckle noise
  - Prediction stability reported alongside explanation stability
- **Sanity checks:** progressive parameter randomization (GradCAM), intensity/edge/center-prior baselines
- **Ablation matrix:** 22 entries with terminal states (4 validated, 11 implemented, 7 planned)
- Failed XAI methods disclosed (GradCAM++ autograd limitation)
- 185 unit tests pass locally; all pass in Colab
- Outputs: `xai_robustness_metrics.parquet`, `xai_sanity_metrics.parquet`, `xai_randomization_curves/`, `ablation_matrix.csv`

### Phase 11b: Ablation scientific evidence (five-fold OOF)

- **6 models** evaluated via OOF aggregation across 5 folds (647 samples each):
  Baseline CE, Full Causal, A02 (Necessity only), A03 (Sufficiency only),
  A04 (Background only), G01 (Gating disabled)
- **Classification metrics:** AUROC, Balanced Accuracy, F1, PR-AUC, ECE, Brier
- **Causal components:** Lesion necessity, sufficiency, background invariance
- **Paired statistical tests:** Wilcoxon signed-rank across 9 model pairs
- Fold-0 retained as exploratory; OOF evidence is primary scientific evidence
- Outputs: `ablation_scientific_summary.parquet`, `ablation_paired_tests.json`, `ablation_component_comparison.md`

**Key evidence across all phases:**

| Metric | Value |
|--------|-------|
| Split digest | `2a88e7ada1aff73e245d...` |
| Manifest digest | `6462d283b3fcfe6657ec...` |
| BUS-UCLM loaded during development | Never |
| Group disjointness | Verified across all folds |
| Donor-leakage tests | Passed |
| Baseline OOF AUROC | 0.7869 |
| Full Causal OOF AUROC | 0.7396 |
| Split seed | 42 |
| Backbone | EfficientNet-B0 |
| GPU used | Tesla T4 (Colab Free) |

## Datasets

- **BUSI** (primary): ~647 breast ultrasound images with lesion masks — benign and malignant.
- **BUS-UCLM** (external validation): frozen — never loaded during development.

## Repository structure

```
src/causalmask/
  data/
    manifest.py          — Dataset manifest creation and validation
    datasets.py          — PyTorch Dataset adapters for BUSI images
    transforms.py        — Paired image-mask transforms (train/eval)
    splits.py            — Split loading, digest computation, reproducibility
    duplicate_audit.py   — Exact and near-duplicate detection (SHA-256, pHash, SSIM)
  counterfactuals/
    masks.py             — Lesion-plus-margin masks (0%, 5%, 10%, 20%)
    sufficient.py        — Lesion-sufficient images (Gaussian-blurred exterior)
    removal.py           — Lesion-removed images (Telea, Navier-Stokes inpainting)
    background_swap.py   — Background-swapped images (partition-isolated donors)
    controls.py          — Sham controls (random region, shifted mask)
    quality.py           — Quality metrics, caching, Parquet export, audit grids
  models/
    factory.py           — Model creation (EfficientNet-B0, ResNet-18)
  training/
    engine.py            — Training loop with checkpointing, AMP, early stopping
    checkpointing.py     — Checkpoint save/load/resume, run status
    losses.py            — Causal losses (CE, sufficiency, background, necessity)
    schedules.py         — Loss-weight schedules (warm-up, ramp, gating)
    causal_trainer.py    — Causal trainer extending engine with counterfactual forwards
  evaluation/
    classification.py    — AUROC, balanced accuracy, sensitivity, specificity, F1, Youden threshold
    calibration.py       — ECE, MCE, Brier score
    causalmask_score.py  — Composite CausalMask score (harmonic mean)
    faithfulness.py      — Insertion/deletion AUC, prediction-flip metrics
    localization.py      — Attribution mass, pointing game, soft Dice, IoU
    robustness.py        — Explanation robustness (5 transforms, Spearman, SSIM, top-k)
    sanity.py            — Parameter randomization, intensity/edge/center baselines
  xai/
    base.py              — Common XAI interface
    gradcam.py           — Grad-CAM and Grad-CAM++
    integrated_gradients.py — Integrated Gradients
    rise.py              — RISE (Randomized Input Sampling for Explanation)
    normalization.py     — Min-max, percentile normalization
  statistics/
    bootstrap.py         — Group-level stratified bootstrap confidence intervals
  reproducibility.py     — Seed management, environment capture
notebooks/
  00_environment_and_repository_audit.ipynb
  01_download_and_extract_datasets.ipynb
  02_dataset_manifest_and_quality_audit.ipynb
  03_duplicate_audit_and_fixed_splits.ipynb
  04_baseline_pipeline_smoke_test.ipynb
  05_baseline_five_fold_cross_validation.ipynb
  06_counterfactual_engine_and_quality_audit.ipynb
  07_causalmask_metrics_on_baseline.ipynb
  08_causal_regularization_one_fold_pilot.ipynb
  09_xai_methods_and_faithfulness.ipynb
  10_causal_five_fold_cross_validation.ipynb
  11_robustness_sanity_and_ablations.ipynb
  11b_ablation_five_fold_scientific.ipynb
tests/
  unit/                  — Unit tests for all modules
```

## Key design decisions

- **Duplicate-group splitting**: Images are grouped by SHA-256 exact duplicates and pHash+SSIM near-duplicates before 5-fold split.
- **No external data leakage**: BUS-UCLM is never loaded during development or validation. It is frozen for external validation.
- **ImageNet normalization**: All training and inference uses ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). Robustness transforms handle the unnormalize→transform→renormalize cycle.
- **Partition-local donors**: Background-swap donors always come from the same active partition. No cross-partition leakage.
- **Frozen configuration**: Five-fold causal training uses the exact configuration frozen at the end of the Phase 8 pilot.
- **Conservative GPU settings**: Batch size 16, AMP enabled, gradient clipping 1.0 — suitable for single Colab T4 GPU.
- **Deterministic run IDs**: `make_fold_run_id` use (fold, seed) — no timestamps — so checkpoint resume works across Colab disconnections.
- **Google Drive sync**: All artifacts sync to Drive for persistence across Colab sessions.

## Setup

```bash
git clone https://github.com/Sayem7456/CausalMask-XAI.git
cd CausalMask-XAI
pip install -e .[dev]
```

To run on Colab, open any `notebooks/` notebook — the bootstrap cell handles cloning, Drive mount, and dependency installation automatically.

## Tests

```bash
pytest tests/
```

## What's next

| Phase | Milestone |
|-------|-----------|
| 12 | BUS-UCLM untouched external validation |
| 13 | Bootstrap confidence intervals, Holm correction, final statistical aggregation |
| 14 | Paper-ready reporting, figures, reproducibility audit |

## Reference

Proposal: [`CausalMask-XAI.md`](./CausalMask-XAI.md) — full research specification with methodology, evaluation metrics, ablation plan, and publication targets.
