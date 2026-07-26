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
| **6** | **Counterfactual engine and quality audit** | **Executed — validated** |
| 7–12 | CausalMask metrics, regularization, XAI methods, ablations | Planned |

### Phase 6 results (executed 2026-07-26 on Colab T4)

- **647 BUSI primary-task samples** processed (benign + malignant)
- **14,572 quality metric records** across all margins × operators
- **4 margin ratios** (0%, 5%, 10%, 20%) relative to lesion bbox scale
- **3 intervention types**: lesion-sufficient, lesion-removed (Telea + Navier-Stokes), background-swapped (same + opposite class)
- **3 sham controls**: random-region removal, random-region preservation, shifted-mask
- **27 visual audit grids** stratified by operator and margin
- **103 failed samples** explicitly recorded with reasons
- **Donor isolation verified**: no cross-partition or self-donation
- BUS-UCLM never loaded or consulted
- 57 counterfactual unit tests pass (163 total)
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
  evaluation/
    classification.py    — AUROC, balanced accuracy, sensitivity, specificity, F1, Youden threshold
    calibration.py       — ECE, MCE, Brier score
  reproducibility.py     — Seed management, environment capture
notebooks/
  00_environment_and_repository_audit.ipynb
  01_download_and_extract_datasets.ipynb
  02_dataset_manifest_and_quality_audit.ipynb
  03_duplicate_audit_and_fixed_splits.ipynb
  04_baseline_pipeline_smoke_test.ipynb
  05_baseline_five_fold_cross_validation.ipynb
  06_counterfactual_engine_and_quality_audit.ipynb
tests/
  unit/                  — 163 unit tests (57 counterfactual)
```

## Key design decisions

- **Duplicate-group splitting**: Images are grouped by SHA-256 exact duplicates and pHash+SSIM near-duplicates before 5-fold split. No patient-level splitting (BUSI lacks reliable patient IDs).
- **No external data leakage**: BUS-UCLM is never loaded during development or validation. It is frozen for Phase 12 external validation.
- **Deterministic run IDs**: `make_fold_run_id` uses (fold, seed) — no timestamps — so checkpoint resume works across Colab disconnections.
- **Google Drive sync**: All artifacts (runs, reports, manifests, splits) sync to Drive for persistence across Colab sessions.
- **All tests pass**: 163 unit tests pass (57 counterfactual, 106 data/models/training/evaluation).

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
| 7 | CausalMask Score (necessity, sufficiency, background invariance, localization) |
| 8 | XAI methods — Grad-CAM, Grad-CAM++, Integrated Gradients, RISE |
| 9 | Causal regularization training |
| 10 | BUS-UCLM external validation |
| 11 | Ablations, bootstrap CIs, statistical tests |
| 12 | Paper-ready reporting and reproducibility audit |

## Reference

Proposal: [`CausalMask-XAI.md`](./CausalMask-XAI.md) — full research specification with methodology, evaluation metrics, ablation plan, and publication targets.
