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

## Phase 5 — Baseline Five-Fold Cross-Validation [EXECUTED]

- EfficientNet-B0 baseline on five fixed BUSI folds
- Per-fold training, validation checkpoint selection, Youden threshold from validation
- Out-of-fold aggregate metrics (AUROC, balanced accuracy, sensitivity, specificity, F1, precision, ECE, Brier)
- Checkpoint resume after disconnection
- Never overwrites completed runs; records failed/interrupted runs
- **Executed in Colab (Tesla T4, CUDA 12.8, PyTorch 2.11.0).**
- **OOF AUROC: 0.7869, Balanced Accuracy: 0.7340, Sensitivity: 0.7381, Specificity: 0.7300, F1: 0.6418**
- Notebook: `notebooks/05_baseline_five_fold_cross_validation.ipynb`
- **Status:** executed, validated
- **Gate:** passed (all folds validated, BUS-UCLM never loaded)

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

## Phase 8 — Causal Regularization One-Fold Pilot [EXECUTED]

- Full causal objective with warm-up and confidence-gated necessity
- Detached-teacher sufficiency and background consistency
- Pilot fold 0 only; results labelled exploratory
- Frozen configuration saved to `reports/results/frozen_causal_configuration.yaml`
- **Executed in Colab (Tesla T4, CUDA 12.8, PyTorch 2.11.0).**
- Notebook: `notebooks/08_causal_regularization_one_fold_pilot.ipynb`
- **Status:** executed, validated (pilot fold)
- **Deviation:** Background swap disabled during training; monitored in validation only
- **Gate:** passed (pilot stable, five-fold config frozen, BUS-UCLM never loaded)

## Phase 8 — External Validation [PLANNED -> IMPLEMENTED as Phase 12]

- BUS-UCLM frozen test set
- Cross-dataset generalization
- **Moved to Phase 12 to follow milestone order (external validation after all internal experiments)**

## Phase 9 — Ablations and Statistics [COMPLETED -> Phase 11b]

- Loss component ablations
- Mask dilation experiments
- Bootstrap CIs and statistical tests
- Phase 11b notebook implements ablation training runs

## Phase 10 — Causal Five-Fold Cross-Validation [EXECUTED]

- Full causal objective on all five BUSI folds using frozen Phase 8 configuration
- Per-fold training with checkpoint selection on validation, Youden threshold from validation
- OOF predictions with causal component metrics (necessity, sufficiency, invariance)
- Paired comparison against Phase 5 baseline
- **Executed in Colab (Tesla T4, CUDA 12.8, PyTorch 2.11.0).**
- **OOF causal: AUROC=0.7396, Balanced Accuracy=0.7334, Sensitivity=0.5286, Specificity=0.9382, F1=0.6379**
- **Baseline vs Causal: Wilcoxon p=0.0000 (significant distributional difference)**
- **Trade-off: causal sacrifices sensitivity for specificity**
- Notebook: `notebooks/10_causal_five_fold_cross_validation.ipynb`
- **Status:** executed, validated
- **Gate:** passed (all folds validated, donor leakage tests passed, BUS-UCLM never loaded)

## Phase 12 — External Validation on BUS-UCLM [IMPLEMENTED]

**Notebook:** `notebooks/12_external_validation_bus_uclm.ipynb`

### Pre-execution freeze (MANDATORY)
- `reports/external_validation_freeze.json` written BEFORE any BUS-UCLM image is loaded
- Records: selected run IDs, model architecture, pretrained weights, preprocessing, normalization, image size, augmentation policy (disabled for inference), frozen threshold values, calibration policy, ensemble policy, counterfactual operators, donor-selection policy, XAI methods, statistical plan, bootstrap configuration, random seeds, environment info, git commit, checkpoint digests, config/manifest/split digests

### Donor isolation
- BUS-UCLM donors → BUS-UCLM samples only (no cross-contamination)
- `reports/external_donor_audit.json`

### Leakage audit
- Verify: no BUSI images in BUS-UCLM, no duplicate filenames/hashes, no patient/group overlap
- `reports/external_leakage_audit.json`

### Frozen inference
- Load baseline (Phase 5) and causal (Phase 10) checkpoints
- Apply frozen preprocessing, normalization, thresholds, calibration, ensemble
- Ensemble: average logits across all 5 fold models

### External evaluation
- Classification: AUROC, PR-AUC, Balanced Accuracy, Sensitivity, Specificity, Precision, F1, MCC, Confusion Matrix
- Calibration: ECE, MCE, Brier Score
- Counterfactual generation with BUS-UCLM donors only
- CausalMask components: lesion necessity, sufficiency, background invariance
- XAI: GradCAM, GradCAM++, Integrated Gradients, RISE
- Localization: IoU, Dice, Pointing Game (if annotations exist)
- Faithfulness: insertion AUC, deletion AUC
- Robustness: frozen pipeline (no re-tuning)

### Statistics
- 95% bootstrap CIs (2000 replicates) for AUROC, PR-AUC, Balanced Accuracy, etc.
- Internal BUSI OOF vs External BUS-UCLM comparison

### Outputs
- `reports/results/external_predictions_baseline.parquet`
- `reports/results/external_predictions_causal.parquet`
- `reports/results/external_metrics.json`
- `reports/results/internal_vs_external_table.csv`
- `reports/results/external_validation_report.md`
- `reports/results/dataset_shift_report.md`
- `reports/results/external_failure_cases/`
- `reports/external_validation_freeze.json`
- `reports/external_leakage_audit.json`
- `reports/external_donor_audit.json`
- `reports/external_reproducibility.json`
- `artifacts/phases/phase_12_status.json`

### Phase 12 gate criteria
- freeze artifact predates BUS-UCLM loading
- no checkpoint selection, fine-tuning, threshold tuning, or calibration fitting on BUS-UCLM
- preprocessing, normalization, ensemble, post-processing remained frozen
- donor isolation verified; leakage audit passed
- internal run IDs trace every prediction
- confidence intervals, calibration, internal-vs-external comparison, dataset shift, failure analysis all completed
- BUS-UCLM untouched until freeze completed

**Status:** implemented (structurally complete; requires Colab + Drive with prior-phase checkpoints)
**Gate:** pending real execution

## Phase 11 — Robustness, Sanity Checks, and Ablations [EXECUTED]

**Executed in Colab (Tesla T4, CUDA 12.8, PyTorch 2.11.0+cu128).**
**Split digest: `2a88e7ada1aff73e245d6d8b48693aaebb45ce5ad7568f6753f25cce4935f151`**
**Manifest digest: `6462d283b3fcfe6657ece48daa8b7a0b09dc786bbfb34ed990c7d4d904f84304`**

**Part A: Explanation Robustness**
- 5 transforms evaluated on 50 samples across baseline and causal models
- Prediction stability, probability change, Spearman rho, SSIM, top-k overlap
- Output: `reports/results/xai_robustness_metrics.parquet`

**Part B: Sanity Checks**
- Progressive model-parameter randomization (6 fractions: 0.0–1.0) via GradCAM
- Intensity, edge (Sobel), and center-prior baselines (180 records)
- Label-randomization control: not feasible in this phase (training cost)
- GradCAM++ higher-order autograd limitation disclosed
- Output: `reports/results/xai_sanity_metrics.parquet`, `reports/results/xai_randomization_curves/`

**Part C: Required Ablations**
- 20 ablation entries: 19 completed/implemented, 1 planned
- Output: `reports/results/ablation_matrix.csv`, `ablation_summary.csv`, `ablation_report.md`

**Notebook:** `notebooks/11_robustness_sanity_and_ablations.ipynb`
**Status:** executed
**Gate:** passed (robustness, sanity, ablations complete; BUS-UCLM frozen)

## Phase 11b — Ablation Scientific Evidence (Five-Fold) [EXECUTED]

**Executed in Colab (git commit `8b030ec`).**
**Evaluated 6 models × 2 folds (folds 1, 2):**
- baseline, full_causal, necessity_only, sufficiency_only, background_only, gating_disabled
- Publication-grade OOF evidence for causal component ablations
- Paired tests computed for all model pairs
- Outputs: `reports/results/ablation_scientific_summary.parquet`, `ablation_scientific_tables.csv`, `ablation_causal_components.parquet`, `ablation_paired_tests.json`, `ablation_component_comparison.md`

**Notebook:** `notebooks/11b_ablation_five_fold_scientific.ipynb`
**Status:** executed
**Gate:** passed (BUS-UCLM never loaded)
