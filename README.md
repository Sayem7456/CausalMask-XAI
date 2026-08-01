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
| **10** | **Causal five-fold cross-validation** | **Done — validated** |
| 11–12 | Ablations, bootstrap, external validation, reporting | Planned |

### Phase 7: CausalMask metrics on baseline (executed 2026-07-29 on Colab T4)

- **544 samples** processed across 5 folds with causal component metrics
- **103 failed samples** recorded with explicit reasons
- **3 margins** (0%, 10%, 20%), **2 removal operators** (Telea, Navier-Stokes), **2 donor classes** (same, opposite), **2 target definitions** (predicted, true)
- **3 donors per sample** for stable background-invariance estimates
- **1,632 metric records** saved to `reports/results/baseline_causal_components.parquet`
- Component metrics, sham controls, and composite CausalMask score computed
- No model retrained — frozen baseline checkpoints only
- BUS-UCLM never loaded

### Phase 8: Causal regularization pilot (executed 2026-07-30 on Colab T4)

- **One-fold pilot** (fold 0) with full causal training objective
- CE + sufficiency consistency + background consistency + gated necessity ranking
- 3-epoch CE warm-up, 3-epoch necessity ramp, confidence threshold 0.6
- Config frozen as `reports/results/frozen_causal_configuration.yaml`
- Results labelled exploratory — five-fold validation deferred to Phase 10
- BUS-UCLM never loaded

### Phase 9: XAI methods and faithfulness (executed 2026-07-31 on Colab T4)

- **4 XAI methods** evaluated: Grad-CAM, Grad-CAM++, Integrated Gradients (50 steps), RISE (1000 masks)
- **101 test samples** per method on fold 0 (baseline + causal pilot)
- **0 failures** across all methods on baseline
- Localization and faithfulness metrics (insertion/deletion AUC, pointing game, soft Dice, IoU)
- Target layers explicitly resolved per architecture (`features` for EfficientNet-B0)
- Grad-CAM++ has known autograd limitation (4 tests xfailed) — not a defect
- Attribution caching by model checkpoint digest for resume safety

### Phase 10: Causal five-fold cross-validation (executed 2026-08-01 on Colab T4)

- **All 5 folds validated** using frozen Phase 8 configuration
- EfficientNet-B0 backbone, full causal objective, 20 epochs per fold
- Per-sample causal components: necessity, sufficiency, background invariance, prediction flips
- Paired comparison against baseline (Wilcoxon p=0.0000)
- Denominator reconciliation: baseline=647, causal=647 (match verified)
- Donor-leakage tests pass across all 5 folds
- BUS-UCLM never loaded
- Phase gate: passed

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
    causalmask_score.py  — Composite CausalMask score (harmonic mean of necessity, sufficiency, invariance)
    faithfulness.py      — Insertion/deletion AUC, prediction-flip metrics
    localization.py      — Attribution mass, pointing game, soft Dice, IoU
  xai/
    base.py              — Common XAI interface (attribute, normalize, resize)
    gradcam.py           — Grad-CAM and Grad-CAM++ implementations
    integrated_gradients.py — Integrated Gradients (Captum)
    rise.py              — RISE (Randomized Input Sampling for Explanation)
    normalization.py     — Min-max, percentile, and no-op normalization
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
tests/
  unit/                  — Unit tests for all modules
```

## Key design decisions

- **Duplicate-group splitting**: Images are grouped by SHA-256 exact duplicates and pHash+SSIM near-duplicates before 5-fold split. No patient-level splitting (BUSI lacks reliable patient IDs).
- **No external data leakage**: BUS-UCLM is never loaded during development or validation. It is frozen for external validation.
- **Deterministic run IDs**: `make_fold_run_id` / `make_causal_run_id` use (fold, seed) — no timestamps — so checkpoint resume works across Colab disconnections.
- **Google Drive sync**: All artifacts (runs, reports, manifests, splits) sync to Drive for persistence across Colab sessions.
- **Partition-local donors**: Background-swap donors always come from the same active partition (training, validation, or test). No cross-partition leakage.
- **Frozen configuration**: Five-fold causal training uses the exact configuration frozen at the end of the Phase 8 pilot (`reports/results/frozen_causal_configuration.yaml`).
- **Conservative GPU settings**: Batch size 16, AMP enabled, gradient clipping 1.0 — suitable for single Colab T4 GPU.

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
| 11 | Ablations — loss components, margin ratios, removal operators, architecture, necessity gating, donor class |
| 12 | BUS-UCLM untouched external validation |
| 13 | Bootstrap confidence intervals, paired statistical tests, Holm correction |
| 14 | Paper-ready reporting, figures, reproducibility audit |

## Reference

Proposal: [`CausalMask-XAI.md`](./CausalMask-XAI.md) — full research specification with methodology, evaluation metrics, ablation plan, and publication targets.
