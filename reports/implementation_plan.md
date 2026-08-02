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

## Phase 6 — Counterfactual Engine and Quality Audit [IMPLEMENTED]

- **Lesion-plus-margin masks** at 0%, 5%, 10%, 20% relative to bounding-box scale
- **Lesion-sufficient images** — lesion+margin preserved, Gaussian-blurred exterior
- **Lesion-removed images** — OpenCV Telea and Navier-Stokes inpainting (described as interventions)
- **Background swaps** — partition-isolated donors, same/opposite class, no self-donation
- **Sham controls** — random-region removal, random-region preservation, shifted-mask
- **Quality metrics** — changed-pixel fraction, preservation error, boundary gradient, SSIM, histogram divergence, failure rate
- **Deterministic caching** — keyed by sample ID, manifest digest, split digest, operator, margin, donor ID, seed, config digest
- **Visual audit grids** — stratified by class, lesion size, margin, operator, quality flags
- **57 unit tests** pass
- All outputs saved under versioned artifact directories
- No causal model training or performance claims
- Notebook: `notebooks/06_counterfactual_engine_and_quality_audit.ipynb`
- **Status:** implemented, runnable (blocked locally — requires Colab Drive with BUSI data)
- **Gate:** structurally complete; real execution requires BUSI via Google Drive

### Module summary
- `src/causalmask/counterfactuals/__init__.py`
- `src/causalmask/counterfactuals/masks.py`
- `src/causalmask/counterfactuals/sufficient.py`
- `src/causalmask/counterfactuals/removal.py`
- `src/causalmask/counterfactuals/background_swap.py`
- `src/causalmask/counterfactuals/controls.py`
- `src/causalmask/counterfactuals/quality.py`

## Phase 7 — CausalMask Metrics on Frozen Baseline Models [IMPLEMENTED]

- **Component metrics** — raw lesion necessity, normalized lesion necessity, lesion sufficiency, background invariance, prediction-flip rate, donor-stratified invariance, lesion-vs-sham difference
- **Per-sample computation** — `compute_per_sample_causal_metrics` with separate predicted-class and true-class target columns
- **CausalMask composite score** — harmonic (primary), arithmetic, and geometric aggregation with equal preregistered weights
- **Aggregation sensitivity** — Spearman correlations between harmonic, arithmetic, and geometric composites
- **Group-aware bootstrap CIs** — resampling at the group level
- **Distribution analysis** — benign vs malignant, correct vs incorrect, lesion vs sham, margin sensitivity, operator sensitivity
- **Synthetic metric tests** — 48 tests proving expected behavior and valid numeric ranges
- Notebook: `notebooks/07_causalmask_metrics_on_baseline.ipynb`
- **Status:** implemented, runnable (blocked locally — requires Colab Drive with baseline checkpoints and BUSI data)
- **Gate:** structurally complete; real execution requires Colab + Drive

### Module summary
- `src/causalmask/statistics/__init__.py`
- `src/causalmask/statistics/bootstrap.py`
- `src/causalmask/evaluation/faithfulness.py`
- `src/causalmask/evaluation/causalmask_score.py`
- `tests/unit/test_causalmask_metrics.py`
- `tests/unit/test_bootstrap.py`

### Phase 7 gate criteria
- Component metrics pass synthetic tests — YES (48 tests)
- Sham controls are included — YES
- Intervention-operator sensitivity is reported — YES
- The composite score remains secondary — YES
- No model was retrained — YES
- No external data were used — YES

## Phase 8 — Causal Regularization [PLANNED]

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

## Phase 11 — Robustness, Sanity Checks, and Ablations [IMPLEMENTED]

**Part A: Explanation Robustness**
- Horizontal flip (with geometric inverse)
- Mild contrast adjustment (factor 1.1)
- Mild gamma correction (gamma 1.1)
- Small translation (5 pixels, with geometric inverse)
- Mild speckle noise (std 0.02)
- Prediction stability, probability change, Spearman rho, SSIM, top-k overlap
- Reports prediction change alongside explanation change

**Part B: Sanity Checks**
- Progressive model-parameter randomization (6 fractions: 0.0–1.0)
- Intensity baseline
- Edge baseline (Sobel)
- Center-prior baseline (Gaussian)
- Randomization-degradation curves via GradCAM
- Label-randomization control: not feasible in this phase (training cost)
- Failed XAI methods disclosed: GradCAM++ higher-order autograd limitation

**Part C: Required Ablations**
- Ablation matrix: 22 entries covering loss components, margins, operators, gating, aggregation, architecture, controls
- 4 validated, 11 implemented, 7 planned
- Sham controls and operator sensitivity reported
- BUS-UCLM remains frozen

**Notebook:** `notebooks/11_robustness_sanity_and_ablations.ipynb`
**Status:** implemented (structurally complete; real execution requires Colab with prior-phase checkpoints)
**Gate:** structurally complete; real execution requires Colab + Drive
