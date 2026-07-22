# Implementation Plan

## Phase 0 — Environment and Repository Audit [COMPLETED]

- Project structure established
- Reproducibility helpers created
- Environment notebook created (`00_environment_and_repository_audit.ipynb`)

## Phase 1 — Dataset Download [COMPLETED]

- Download BUSI and BUS-UCLM archives
- Extract and organize raw files
- Create `data/raw/dataset_sources.json`
- Notebook: `01_download_and_extract_datasets.ipynb`

## Phase 2 — Dataset Manifest and Quality Audit [COMPLETED]

- Discover and pair images with masks
- Generate immutable manifests with SHA-256 checksums
- Validate quality flags, label mapping, and mask coverage
- Save manifests as Parquet
- Notebook: `02_dataset_manifest_and_quality_audit.ipynb`
- **Gate passed**

## Phase 3 — Duplicate Audit and Fixed Splits [COMPLETED]

- SHA-256 exact duplicate detection
- Perceptual hash near-duplicate detection
- SSIM verification of candidate pairs
- Group-aware five-fold stratified split
- Split digest and manifest digest recorded
- Notebook: `03_duplicate_audit_and_fixed_splits.ipynb`
- **Gate passed**

## Phase 4 — Baseline Pipeline Smoke Test [COMPLETED]

- Implemented:
  - `src/causalmask/reproducibility.py` (enhanced)
  - `src/causalmask/data/transforms.py` — paired image-mask transforms
  - `src/causalmask/models/factory.py` — EfficientNet-B0, ResNet-18
  - `src/causalmask/training/engine.py` — training loop with AMP, early stopping
  - `src/causalmask/training/checkpointing.py` — save/resume
  - `src/causalmask/evaluation/classification.py` — AUROC, accuracy, F1, etc.
  - `src/causalmask/evaluation/calibration.py` — ECE, MCE, Brier
- One-epoch smoke test completed with synthetic data
- Checkpoint save/resume verified
- Prediction export tested
- 90 unit tests pass
- All outputs labelled **smoke**
- No full cross-validation or external evaluation
- **Status:** smoke-tested

## Phase 5 — Baseline Five-Fold Cross-Validation [IMPLEMENTED]

- EfficientNet-B0 baseline on five fixed BUSI folds
- Per-fold training, validation checkpoint selection, Youden threshold from validation
- Out-of-fold aggregate metrics (AUROC, balanced accuracy, sensitivity, specificity, F1, precision, ECE, Brier)
- Checkpoint resume after disconnection
- Never overwrites completed runs; records failed/interrupted runs
- Notebook: `notebooks/05_baseline_five_fold_cross_validation.ipynb`
- **Status:** implemented, runnable (blocked locally — requires Colab Drive with BUSI data)
- **Gate:** structurally complete; real execution requires BUSI via Google Drive

## Phase 6 — CausalMask Score [PLANNED]

- Necessity, sufficiency, invariance, localization
- Harmonic mean combination
- Score analysis and visualization

## Phase 7 — Causal Regularization [PLANNED]

- Training with combined loss
- Hyperparameter selection
- Comparison against baselines

## Phase 8 — External Validation [PLANNED]

- BUS-UCLM frozen test set
- Cross-dataset generalization

## Phase 9 — Ablations and Statistics [PLANNED]

- Loss component ablations
- Mask dilation experiments
- Bootstrap CIs and statistical tests

## Phase 10 — Final Report [PLANNED]

- Figures, tables, paper preparation
