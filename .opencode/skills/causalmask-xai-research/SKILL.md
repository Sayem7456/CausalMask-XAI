---
name: causalmask-xai-research
description: Implement, verify, and document the CausalMask-XAI breast-ultrasound research project end to end. Use for dataset auditing, leakage-safe cross-validation, lesion-conditioned counterfactual generation, causal regularization, medical XAI evaluation, external validation, statistical testing, ablations, reproducibility, and conference-grade reporting based on the root CausalMask-XAI.md specification.
license: MIT
compatibility: opencode
metadata:
  audience: medical-ai-researchers
  project: causalmask-xai
  framework: pytorch
  scope: end-to-end-research-implementation
  rigor: conference-grade
  source-of-truth: CausalMask-XAI.md
---

# CausalMask-XAI Research Implementation Skill

## Mission

Build a complete, reproducible, scientifically defensible implementation of the research proposal in `CausalMask-XAI.md`.

The implementation must answer this question:

> Does a breast-ultrasound classifier genuinely depend on the lesion and clinically relevant perilesional context, or does it exploit background texture, acquisition artefacts, annotations, or other shortcuts?

The project is not complete when code runs. It is complete only when:

1. the data pipeline is audited and leakage-safe;
2. baselines and the proposed method are implemented fairly;
3. counterfactual interventions are controlled and reproducible;
4. classification, calibration, localization, faithfulness, robustness, and sanity metrics are computed correctly;
5. internal and external evaluations are separated;
6. statistical uncertainty is reported;
7. every reported result is traceable to an immutable run artifact; and
8. limitations and failed hypotheses are documented honestly.

## When to use this skill

Use this skill for any task involving:

- reading or operationalizing `CausalMask-XAI.md`;
- creating or modifying the research repository;
- preparing BUSI or BUS-UCLM data;
- duplicate detection or split generation;
- training EfficientNet-B0, ResNet-18, or defined comparison models;
- lesion-sufficient, lesion-removed, or background-swapped images;
- CausalMask metrics or causal regularization;
- Grad-CAM, Grad-CAM++, Integrated Gradients, or RISE;
- explanation localization, faithfulness, robustness, or sanity checks;
- calibration, bootstrap confidence intervals, hypothesis tests, or ablations;
- external validation;
- generating experiment tables, figures, reports, or paper-ready evidence;
- auditing whether a claimed result is actually supported by artifacts.

Do not use this skill for unrelated medical-imaging projects unless the user explicitly asks to adapt it.

## Source-of-truth precedence

Use the following precedence when instructions conflict:

1. the user's current explicit instruction;
2. `CausalMask-XAI.md` in the project root;
3. this skill;
4. checked-in project configuration and documentation;
5. conservative implementation defaults.

Never silently override the proposal. Record any necessary deviation in:

- the run configuration;
- `reports/deviations.md`; and
- the final experiment report.

If `CausalMask-XAI.md` is missing, stop research-specific implementation and report the missing source file. You may still inspect the repository and prepare a non-destructive inventory.

## Non-negotiable scientific rules

### No fabricated evidence

Never invent:

- dataset counts;
- patient identifiers;
- metric values;
- confidence intervals;
- p-values;
- training completion;
- GPU usage;
- checkpoint quality;
- external-validation performance;
- visual findings;
- successful commands; or
- file paths that do not exist.

Label all outputs using one of these states:

- **planned**: specified but not implemented;
- **implemented**: code exists and static/unit checks pass;
- **runnable**: a smoke run completes;
- **executed**: the intended experiment completed;
- **validated**: outputs passed integrity checks and are suitable for reporting;
- **failed**: execution or validation failed;
- **blocked**: required data, hardware, dependency, or decision is unavailable.

Synthetic fixtures and toy runs must be visibly marked `synthetic`, `smoke`, or `debug`. Never present them as scientific results.

### No data leakage

Never allow:

- the same patient or inferred duplicate group in train and test;
- external BUS-UCLM images to influence training, early stopping, threshold selection, preprocessing choices, loss weights, model selection, or ablation selection;
- test-fold labels to select hyperparameters;
- preprocessing fitted on the full dataset;
- donor backgrounds from held-out data to enter training;
- normal images to be silently mixed into the benign class;
- augmentation before split assignment;
- checkpoint selection using test-fold metrics;
- threshold tuning on the external dataset; or
- repeated manual inspection of external results followed by method changes without declaring exploratory contamination.

### No cherry-picking

Report all preregistered folds, seeds, methods, and primary metrics. Failed runs must remain in the experiment registry with a reason. Do not select a “best fold” as the headline result.

### Fair comparison

All compared training methods must use, unless an ablation explicitly changes them:

- identical split files;
- identical input resolution;
- identical backbone initialization;
- identical augmentation;
- identical optimizer family;
- identical epoch budget and early-stopping rule;
- equivalent hyperparameter-search effort;
- identical evaluation code;
- identical classification threshold policy; and
- the same external test protocol.

## Required project architecture

Adapt to an existing repository when appropriate, but preserve these responsibilities:

```text
.
├── CausalMask-XAI.md
├── pyproject.toml
├── README.md
├── configs/
│   ├── data/
│   │   ├── busi.yaml
│   │   └── bus_uclm.yaml
│   ├── model/
│   │   ├── efficientnet_b0.yaml
│   │   └── resnet18.yaml
│   ├── experiment/
│   │   ├── baseline_ce.yaml
│   │   ├── attention_supervised.yaml
│   │   ├── causal_necessity.yaml
│   │   ├── causal_sufficiency.yaml
│   │   ├── causal_background.yaml
│   │   └── causal_full.yaml
│   └── evaluation/
│       └── default.yaml
├── data/
│   ├── raw/                  # immutable and gitignored
│   ├── manifests/
│   ├── splits/
│   ├── cache/
│   └── README.md
├── src/
│   └── causalmask/
│       ├── cli.py
│       ├── config.py
│       ├── constants.py
│       ├── reproducibility.py
│       ├── data/
│       │   ├── manifest.py
│       │   ├── datasets.py
│       │   ├── transforms.py
│       │   ├── duplicate_audit.py
│       │   └── splits.py
│       ├── models/
│       │   ├── factory.py
│       │   └── attention_supervision.py
│       ├── counterfactuals/
│       │   ├── masks.py
│       │   ├── sufficient.py
│       │   ├── removal.py
│       │   ├── background_swap.py
│       │   ├── controls.py
│       │   └── quality.py
│       ├── training/
│       │   ├── engine.py
│       │   ├── losses.py
│       │   ├── schedules.py
│       │   └── checkpointing.py
│       ├── xai/
│       │   ├── base.py
│       │   ├── gradcam.py
│       │   ├── integrated_gradients.py
│       │   ├── rise.py
│       │   └── normalization.py
│       ├── evaluation/
│       │   ├── classification.py
│       │   ├── calibration.py
│       │   ├── localization.py
│       │   ├── faithfulness.py
│       │   ├── robustness.py
│       │   ├── sanity.py
│       │   └── causalmask_score.py
│       ├── statistics/
│       │   ├── bootstrap.py
│       │   ├── paired_tests.py
│       │   └── multiplicity.py
│       └── reporting/
│           ├── tables.py
│           ├── figures.py
│           └── model_cards.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── scripts/
├── artifacts/               # generated and gitignored except schemas
├── reports/
│   ├── experiment_registry.csv
│   ├── deviations.md
│   ├── data_audit.md
│   └── results/
└── .opencode/
    ├── skills/causalmask-xai-research/SKILL.md
    └── agents/causalmask-researcher.md
```

Avoid creating empty architecture for appearance. Add modules when their responsibility is implemented.

## Configuration principles

Use typed, serializable configuration. Prefer the repository's existing configuration system. Otherwise use a minimal YAML plus dataclass or Pydantic design rather than introducing a large framework solely for configuration.

Every experiment configuration must explicitly contain:

- dataset and manifest versions;
- split file and split digest;
- task definition;
- included classes;
- input resolution;
- preprocessing;
- augmentation;
- model and pretrained-weight identifier;
- optimizer and scheduler;
- batch size and gradient accumulation;
- epoch budget;
- early-stopping metric and patience;
- random seed;
- mixed-precision setting;
- counterfactual parameters;
- loss terms, margins, weights, warm-up, and ramp schedule;
- attribution parameters;
- classification threshold policy;
- bootstrap unit and replicate count;
- output directory; and
- hardware/device selection.

Resolved configuration must be copied into each run directory before training begins.

## Environment and dependency policy

Prefer a modern, maintainable Python stack:

- Python 3.11 unless the repository already pins a compatible version;
- PyTorch and torchvision;
- `timm` for lightweight backbones when needed;
- Albumentations or torchvision v2 for paired image-mask transforms;
- OpenCV and scikit-image for counterfactual operations;
- scikit-learn, SciPy, and statsmodels for metrics and statistics;
- Captum for Integrated Gradients;
- a maintained Grad-CAM implementation or a small tested local implementation;
- pandas and Parquet support for prediction-level artifacts;
- PyYAML or the existing config library;
- pytest and Ruff;
- optional static typing with mypy or pyright.

Do not add two libraries for the same responsibility without justification. Pin direct dependencies. Record exact installed versions in every executed run.

## Dataset protocol

### Main task

The primary task is binary classification:

- benign;
- malignant.

Normal images are excluded from the primary binary experiment unless a separately named secondary experiment is added. Never relabel normal as benign.

### Primary and external datasets

- **BUSI** is the internal development dataset.
- **BUS-UCLM** is external validation only.

Do not assume counts or identifier quality from documentation. Inspect the actual files and build a manifest from them.

### Immutable raw data

Treat `data/raw/` as immutable. Never resize, rename, overwrite, or delete raw files during routine processing. Derived files belong in `data/manifests/`, `data/cache/`, or run artifacts.

### Manifest contract

Create a tabular manifest with at least:

- `sample_id`;
- `dataset`;
- `image_path`;
- `mask_path`;
- `label`;
- `patient_id` when reliable;
- `group_id`;
- `width`;
- `height`;
- `channels`;
- `image_sha256`;
- `mask_sha256`;
- `near_duplicate_cluster`;
- `mask_area_fraction`;
- `has_mask`;
- `quality_flags`; and
- `manifest_version`.

Validate:

- image readability;
- mask readability;
- image-mask shape compatibility;
- binary mask semantics;
- non-empty lesion masks for included abnormal cases;
- class labels;
- path uniqueness;
- exact duplicates;
- near duplicates;
- suspicious overlays, rulers, text, or borders;
- extreme aspect ratios; and
- missing metadata.

Write the audit to `reports/data_audit.md` and machine-readable JSON.

### Patient and duplicate grouping

Use patient-level grouping when trustworthy patient IDs exist. When identifiers are absent or unreliable:

1. compute exact hashes;
2. compute perceptual hashes;
3. identify high-similarity candidates;
4. review or algorithmically verify candidate pairs;
5. assign a stable `near_duplicate_cluster`; and
6. use the duplicate cluster as a split group.

Never claim a patient-level split when only image-level or duplicate-group splitting was possible. State the limitation explicitly.

### Split protocol

Create one versioned, immutable five-fold split file for BUSI.

Requirements:

- group-disjoint folds;
- approximate class stratification;
- no fold regenerated per experiment;
- a training-only validation partition or nested group-aware validation;
- test fold untouched until checkpoint selection is complete;
- split seed and algorithm recorded;
- split integrity tests;
- digest stored with each run.

The external dataset receives no development split unless needed only for grouped uncertainty estimation. It is never used for model choice.

## Preprocessing and augmentation

### Preprocessing

- Preserve ultrasound texture.
- Convert channel format consistently.
- Resize with an interpolation appropriate for images and nearest-neighbour for masks.
- Normalize using training-only statistics or fixed pretrained normalization.
- Store the exact policy in configuration.
- Avoid cropping away lesion margins.
- Detect and document padding, black borders, annotations, and scanner overlays.

### Training augmentation

Use conservative paired transforms:

- horizontal flip when anatomically acceptable;
- small rotation;
- small translation or scale;
- mild gamma or contrast change;
- mild Gaussian or speckle noise.

Masks must undergo exactly the corresponding geometric transform. Do not apply interpolation that creates fractional class labels without deliberate soft-mask handling.

### Evaluation transforms

Evaluation is deterministic. Robustness transformations are a separate protocol and must not alter the base test pipeline.

## Baseline models

Implement:

1. **standard cross-entropy classifier**;
2. **attention-supervised classifier**;
3. **proposed causal-regularized classifier**;
4. optional **lesion-crop-only classifier**.

Primary backbone:

- EfficientNet-B0.

Architecture-generalization backbone:

- ResNet-18.

Use pretrained weights unless the proposal is deliberately testing training from scratch. Record the precise weight identifier.

### Attention-supervised baseline

Do not describe a method as “Grad-CAM supervision” unless the training objective truly differentiates through Grad-CAM correctly.

Prefer a transparent activation-localization baseline:

- derive a non-negative spatial class activation or channel-aggregated feature map;
- resize it to mask resolution;
- normalize it safely;
- optimize Dice, BCE, or mass-outside-mask loss against the lesion-plus-margin region;
- retain classification loss.

Document exactly what map is supervised.

## Mask and margin construction

Let `M` be the binary lesion mask and `M+` the lesion-plus-margin mask.

Implement configurable margin ratios:

- 0%;
- 5%;
- 10%;
- 20%.

Define the ratio relative to a stable lesion scale, such as the maximum lesion bounding-box dimension, rather than an undocumented fixed pixel count. Record the final kernel size for each sample.

Requirements:

- binary output;
- clipping to image bounds;
- no empty masks;
- deterministic morphology;
- unit tests on border-touching and tiny lesions;
- optional feathered alpha mask only for image blending, never for ground-truth localization metrics.

## Counterfactual interventions

Counterfactuals are interventions, not cosmetic augmentations. Preserve the intended causal variable while minimizing unintended distribution shifts.

### Lesion-sufficient image

Preserve pixels inside `M+`. Replace or suppress the exterior.

Primary variant:

- Gaussian-blurred exterior with configurable kernel or sigma.

Optional controlled variant:

- texture-preserving exterior replacement.

Measure and record:

- foreground preservation error;
- boundary discontinuity;
- intensity-distribution shift;
- output range;
- mask coverage.

### Lesion-removed image

Remove `M+` and fill it without a generative foundation model.

Allowed primary methods:

- OpenCV Telea inpainting;
- OpenCV Navier-Stokes inpainting;
- deterministic local patch or texture sampling.

Do not claim the filled region is anatomically realistic. It is an intervention used to estimate dependence.

Use more than one removal operator in a robustness ablation where feasible. A conclusion that only appears with one inpainting method must be reported as operator-sensitive.

### Background-swapped image

Preserve the original lesion and `M+`. Replace only the exterior with a donor image.

Implementation rules:

- donor comes from the same active partition;
- training donors come only from the training partition;
- validation donors come only from validation;
- test donors come only from the internal test fold;
- external donors come only from the external dataset;
- no sample may donate to itself;
- resize donor deterministically;
- apply intensity or histogram alignment outside `M+` when configured;
- feather the blend boundary without changing the preserved lesion;
- record donor `sample_id`, donor label, alignment method, and random seed;
- evaluate same-class and opposite-class donors separately;
- use multiple donors per source image when estimating invariance.

Labels may be used to choose an opposite-class donor for an audit, but must never be passed to the classifier or used to tune the model on held-out data.

### Negative and sham controls

Implement matched controls so the lesion intervention is not confused with generic image corruption:

- same-area random-region removal outside the lesion;
- same-area random-region preservation;
- optional spatially shifted mask that avoids lesion overlap;
- boundary-only perturbation control.

Control regions must be deterministic from a recorded seed and obey image bounds.

### Counterfactual quality audit

Before model conclusions, report intervention quality:

- foreground pixel preservation;
- changed-pixel fraction;
- boundary-gradient discrepancy;
- SSIM outside or inside the protected region as appropriate;
- intensity histogram divergence;
- failure flags;
- visual grids for a stratified sample.

Exclude failed generated images only by a preregistered, model-independent rule. Report exclusion counts.

## Causal decision targets

For explanation faithfulness, define the target as the original model decision:

\[
\hat{y}(x)=\arg\max_c p_c(x)
\]

Use \(p_{\hat{y}}(x)\) for decision-faithfulness metrics.

Also report true-class versions on correctly classified samples. Never mix the two definitions in one table without explicit labels.

Store:

- true label;
- original predicted label;
- target class used by each XAI method;
- original confidence;
- counterfactual confidence;
- correctness.

## CausalMask component metrics

Report raw components individually before any composite score.

### Raw lesion necessity

\[
N_{\mathrm{raw}}
=
p_{\hat{y}}(x)-p_{\hat{y}}(x_{\mathrm{removed}})
\]

Positive values indicate reduced confidence after lesion removal.

### Normalized lesion necessity

\[
N
=
\operatorname{clip}
\left(
\frac{p_{\hat{y}}(x)-p_{\hat{y}}(x_{\mathrm{removed}})}
{\max(p_{\hat{y}}(x),\epsilon)},
0,
1
\right)
\]

### Lesion sufficiency

\[
S
=
\operatorname{clip}
\left(
1-
\left|
p_{\hat{y}}(x)-p_{\hat{y}}(x_{\mathrm{sufficient}})
\right|,
0,
1
\right)
\]

### Background invariance

For donor set \(D(x)\):

\[
B
=
\operatorname{clip}
\left(
1-
\frac{1}{|D(x)|}
\sum_{j \in D(x)}
\left|
p_{\hat{y}}(x)-p_{\hat{y}}(x_{\mathrm{swap},j})
\right|,
0,
1
\right)
\]

Also report prediction-flip rate and same-class versus opposite-class donor results.

### Explanation localization

For non-negative normalized attribution \(A(x)\):

\[
L
=
\frac{\sum A(x)\odot M^+}
{\sum A(x)+\epsilon}
\]

### Composite score

The composite CausalMask score is secondary. Compute it only after individual components are validated.

Default harmonic mean:

\[
\operatorname{CausalMask}
=
\frac{4}
{\frac{1}{N+\epsilon}
+\frac{1}{S+\epsilon}
+\frac{1}{B+\epsilon}
+\frac{1}{L+\epsilon}}
\]

Rules:

- use preregistered equal weights unless the proposal specifies otherwise;
- report the four components beside the composite;
- include bootstrap confidence intervals;
- conduct sensitivity analysis for arithmetic, geometric, and harmonic aggregation;
- never use the external dataset to choose score weights;
- do not hide poor components behind a high aggregate.

## Causal training objective

Base objective:

\[
\mathcal{L}
=
\mathcal{L}_{CE}
+
\lambda_s\mathcal{L}_{sufficiency}
+
\lambda_n\mathcal{L}_{necessity}
+
\lambda_b\mathcal{L}_{background}
\]

### Sufficiency consistency

Use a detached original prediction as the teacher target to reduce trivial co-adaptation:

\[
\mathcal{L}_{sufficiency}
=
D\left(
\operatorname{stopgrad}(p(x)),
p(x_{\mathrm{sufficient}})
\right)
\]

Use KL divergence, Jensen-Shannon divergence, or a documented alternative.

### Background consistency

\[
\mathcal{L}_{background}
=
D\left(
\operatorname{stopgrad}(p(x)),
p(x_{\mathrm{swap}})
\right)
\]

Training donors must come from the current training partition only.

### Necessity ranking

For target class \(y\):

\[
\mathcal{L}_{necessity}
=
\max
\left(
0,
p_y(x_{\mathrm{removed}})
-
p_y(x)
+
m
\right)
\]

Default safety policy:

- warm up using cross entropy only;
- enable necessity loss after the warm-up;
- apply it only when the original sample is correctly classified and exceeds a configured confidence threshold;
- ramp its weight gradually;
- record eligible-sample fraction;
- ablate the gating rule.

Do not tune the confidence threshold, margin, or loss weights on external results.

### Stability safeguards

Monitor:

- class collapse;
- prediction entropy;
- original-image AUROC;
- counterfactual loss magnitude;
- gradient norm;
- eligible-sample fraction;
- divergence between original and sufficient predictions;
- swap consistency;
- removed-image confidence;
- NaNs and infinities.

If a causal loss improves its own metric while substantially degrading diagnostic performance, report the trade-off rather than redefining success.

### Memory-conscious execution

Four simultaneous forward passes may exceed memory. Allowed strategies:

- alternate causal terms by step;
- gradient accumulation;
- reduced training batch size;
- cached deterministic counterfactuals where scientifically valid;
- automatic mixed precision on CUDA;
- sequential forward passes retaining only required tensors.

Never silently change image resolution, backbone, or scientific protocol after an out-of-memory error. Change configuration, assign a new run ID, and record the deviation.

## Explainability methods

Implement and evaluate:

- Grad-CAM;
- Grad-CAM++;
- Integrated Gradients;
- RISE.

Each method must expose a common interface:

```python
attribute(
    model,
    inputs,
    target_classes,
    *,
    device,
    config,
) -> Tensor  # [B, 1, H, W], finite and non-negative after normalization
```

### XAI normalization

Store both raw attribution where possible and a standardized visualization/evaluation map.

Rules:

- handle NaNs and zero maps;
- resize to image resolution consistently;
- use a declared normalization method;
- do not normalize across the full test dataset;
- do not apply a visually chosen threshold for quantitative metrics;
- use identical processing across compared models.

### Target layers

Resolve and test target layers per architecture. Never silently use a layer producing an invalid or excessively low-resolution map.

### Integrated Gradients

Record:

- baseline type;
- integration steps;
- internal batch size;
- convergence delta where available.

Use multiple plausible baselines as a sensitivity analysis if resources permit.

### RISE

Record:

- number of masks;
- mask grid size;
- Bernoulli probability;
- interpolation;
- seed.

Use enough masks for stable estimates or explicitly label low-mask runs as approximate.

## Evaluation protocol

### Classification

Report:

- AUROC;
- balanced accuracy;
- sensitivity;
- specificity;
- F1;
- precision;
- negative predictive value where useful;
- confusion matrix;
- calibration error;
- Brier score.

Threshold policy must be fixed from validation data, for example Youden's J or a clinically motivated sensitivity target. Also report threshold-free metrics.

### Localization

Report:

- attribution mass inside lesion;
- attribution mass inside lesion-plus-margin;
- pointing-game accuracy;
- soft Dice;
- saliency-mask IoU;
- optional energy-based pointing game.

Thresholded saliency metrics require a fixed, declared thresholding rule and sensitivity analysis.

### Faithfulness

Report:

- insertion AUC;
- deletion AUC;
- lesion necessity;
- lesion sufficiency;
- background invariance;
- prediction-flip rate;
- sham-control effects;
- composite CausalMask score.

Insertion/deletion evaluation must declare:

- perturbation baseline;
- number of steps;
- ordering;
- blur or replacement operator;
- target class.

Use paired evaluation because methods are applied to the same images.

### Robustness

Use diagnosis-preserving transformations:

- horizontal flip;
- mild contrast;
- mild gamma;
- small translation;
- mild speckle noise.

For geometric transforms:

1. transform the input;
2. compute attribution;
3. invert the geometric transform on the attribution;
4. compare in the original coordinate system.

Report:

- Spearman rank correlation;
- SSIM;
- top-k attribution overlap;
- localization change;
- prediction stability.

Do not interpret explanation instability when the model prediction itself changes without reporting both.

### Sanity checks

Implement:

- progressive model-parameter randomization;
- label-randomization training for a limited control where feasible;
- attribution comparison against untrained or randomized models;
- simple edge or intensity baselines.

A saliency method that remains nearly unchanged after randomization must not be described as model-sensitive.

### Calibration

Fit any temperature scaling on internal validation predictions only. Apply the frozen calibrator to the internal test fold and external dataset. Report uncalibrated and calibrated values separately.

## Cross-validation and external validation

### Internal BUSI evaluation

Run five fixed group-aware folds.

For every fold:

- train only on the training partition;
- select checkpoint and threshold only on validation;
- evaluate once on the held-out test fold;
- save per-sample predictions;
- save counterfactual outputs or deterministic regeneration metadata;
- save attribution metadata;
- validate artifact completeness.

Aggregate out-of-fold predictions for overall internal metrics. Preserve fold-wise metrics to show variability.

### External BUS-UCLM evaluation

After all method and hyperparameter decisions are frozen:

- train or use the five internal fold models according to the preregistered protocol;
- apply them without fine-tuning;
- use the frozen preprocessing and threshold;
- report per-model and ensemble results when appropriate;
- estimate uncertainty with patient/group-level resampling when possible;
- document domain differences;
- make no post-hoc method changes while still calling the result untouched external validation.

If external results already influenced development, label the dataset as an exploratory secondary test rather than pristine external validation.

## Statistical analysis

### Resampling unit

Bootstrap at the patient or reliable group level. Use image-level bootstrap only when no grouping information exists, and state that it may underestimate uncertainty.

### Confidence intervals

Use stratified or group bootstrap with a fixed seed and enough replicates for stable intervals, normally at least 1,000 for development and 2,000 or more for final reporting when computationally practical.

Report:

- point estimate;
- 95% confidence interval;
- number of valid bootstrap replicates;
- resampling unit;
- random seed.

### Paired comparisons

For per-image or per-group XAI metrics:

- use paired Wilcoxon signed-rank tests when assumptions support it;
- report paired bootstrap confidence intervals for the mean or median difference;
- report an effect size;
- correct the planned family of comparisons using Holm's procedure.

For AUROC differences, use paired bootstrap or a validated DeLong implementation.

Do not treat fold means as five independent samples for strong inferential claims.

### Missing and failed samples

Record why any metric is missing:

- empty saliency;
- counterfactual failure;
- unreadable mask;
- numerical failure;
- unsupported layer;
- prediction unavailable.

Never silently drop samples. Report denominator per metric.

## Essential ablations

At minimum:

1. cross entropy only;
2. necessity loss only;
3. sufficiency loss only;
4. background-invariance loss only;
5. all causal losses;
6. margin ratios 0%, 5%, 10%, and 20%;
7. blurred versus swapped background;
8. at least two lesion-removal operators when feasible;
9. ground-truth mask versus optional predicted mask;
10. EfficientNet-B0 versus ResNet-18;
11. internal versus external evaluation;
12. same-class versus opposite-class donors;
13. causal intervention versus same-area sham control;
14. necessity gating on versus off;
15. composite-score aggregation sensitivity.

Do not start optional predicted-mask experiments until the ground-truth-mask pipeline is validated.

## Reproducibility contract

### Seed control

Set and record seeds for:

- Python;
- NumPy;
- PyTorch CPU;
- PyTorch CUDA;
- data-loader workers;
- split generation;
- donor sampling;
- RISE masks;
- bootstrap;
- sham controls.

Use deterministic algorithms where feasible. When a CUDA operation is nondeterministic, record it rather than claiming exact reproducibility.

### Run identifier

Use a stable unique run ID, for example:

```text
20260720-153012_causal-full_effb0_fold2_seed42_ab12cd3
```

Include:

- timestamp;
- experiment name;
- backbone;
- fold;
- seed;
- short git commit or `nogit`.

### Run artifact contract

Each scientific run must contain:

```text
artifacts/runs/<run_id>/
├── status.json
├── config.resolved.yaml
├── command.txt
├── stdout.log
├── stderr.log
├── environment.json
├── git_state.json
├── data_manifest_digest.json
├── split_digest.json
├── checkpoints/
├── predictions.parquet
├── metrics.json
├── history.csv
├── counterfactual_index.parquet
├── attribution_index.parquet
├── exclusions.csv
├── figures/
└── validation_report.json
```

`status.json` must include:

- state;
- start and end timestamps;
- exit code;
- hostname;
- device;
- run ID;
- config digest;
- split digest;
- git state;
- completed stages;
- failed checks;
- notes.

A run is **validated** only when a validator confirms required artifacts, finite metrics, sample counts, split integrity, and configuration consistency.

### Experiment registry

Maintain `reports/experiment_registry.csv` with:

- run ID;
- date;
- hypothesis;
- experiment;
- model;
- fold;
- seed;
- split digest;
- state;
- primary metrics;
- artifact path;
- parent run;
- notes.

Append; do not rewrite history to hide failed runs.

## Testing requirements

### Unit tests

Test at least:

- manifest validation;
- image-mask pairing;
- label mapping;
- exact and near-duplicate grouping;
- group-disjoint splits;
- deterministic transforms;
- mask dilation;
- border-touching lesions;
- sufficient-image foreground preservation;
- removed-image changed region;
- swap-image foreground preservation;
- donor partition rules;
- sham-control area matching;
- counterfactual output range;
- CausalMask metric bounds;
- expected metric behavior on synthetic examples;
- causal-loss gradient flow;
- necessity gating;
- XAI output shape, finiteness, and target handling;
- geometric robustness inverse transforms;
- bootstrap grouping;
- Holm correction;
- run-manifest validation.

### Integration tests

Implement:

- a tiny synthetic dataset smoke train;
- one real-data mini-batch when data is available;
- one-epoch baseline training;
- one-epoch causal training;
- checkpoint save and resume;
- prediction export;
- one attribution method end to end;
- counterfactual generation end to end;
- metric and report generation.

### Scientific acceptance tests

Before full experiments, demonstrate:

- train/validation/test groups are disjoint;
- raw data are unchanged;
- the same split file is reused;
- external samples never appear in training;
- protected lesion pixels remain unchanged in swap and sufficient interventions within tolerance;
- removal changes only the intended region plus declared blending boundary;
- sham masks match lesion area within tolerance;
- causal losses behave in the intended direction on controlled tensors;
- explanation randomization checks detect at least the expected sensitivity for model-dependent methods;
- run artifacts can regenerate a reported table row.

## Command-line contract

Expose coherent commands. Exact naming may follow an existing CLI, but support equivalent functionality:

```bash
python -m causalmask.cli audit-data \
  --config configs/data/busi.yaml

python -m causalmask.cli make-splits \
  --config configs/data/busi.yaml \
  --output data/splits/busi_v1.json

python -m causalmask.cli train \
  --config configs/experiment/baseline_ce.yaml \
  --fold 0 \
  --seed 42

python -m causalmask.cli train \
  --config configs/experiment/causal_full.yaml \
  --fold 0 \
  --seed 42

python -m causalmask.cli evaluate \
  --run-id <run_id> \
  --partition internal-test

python -m causalmask.cli evaluate-xai \
  --run-id <run_id> \
  --methods gradcam gradcampp integrated-gradients rise

python -m causalmask.cli evaluate-counterfactuals \
  --run-id <run_id>

python -m causalmask.cli external-evaluate \
  --run-id <run_id> \
  --dataset bus-uclm

python -m causalmask.cli statistics \
  --registry reports/experiment_registry.csv

python -m causalmask.cli report \
  --registry reports/experiment_registry.csv \
  --output reports/results/
```

All commands must:

- support `--help`;
- fail with non-zero exit status;
- log resolved configuration;
- avoid overwriting a completed run;
- print the artifact directory;
- validate inputs before expensive work.

## Hardware strategy

Choose device in this order when available:

1. CUDA;
2. Apple MPS for compatible development or smoke runs;
3. CPU for tests and small audits.

For CUDA:

- use automatic mixed precision when configured;
- record GPU name, CUDA version, and peak allocated memory;
- begin with conservative batch size;
- use gradient accumulation rather than changing the scientific protocol;
- release references between XAI batches;
- process RISE and bootstrap in chunks;
- cache deterministic counterfactual metadata or images when storage permits.

For MPS:

- do not assume every attribution or interpolation operator is supported;
- permit controlled CPU fallback per operation;
- record fallback events.

Full five-fold XAI evaluation can be expensive even on a small dataset. Make computations resumable and cache by:

- model checkpoint digest;
- sample ID;
- target class;
- attribution method configuration;
- counterfactual configuration.

## Implementation sequence

### Phase 0: repository and proposal audit

- Read `CausalMask-XAI.md`.
- Inventory existing code.
- Identify reusable modules.
- Create `reports/implementation_plan.md`.
- Create or update the experiment registry.
- Record assumptions and blockers.

### Phase 1: data integrity

- Implement dataset adapters.
- Build manifests.
- Audit masks and labels.
- Detect duplicates.
- create fixed group-safe folds.
- Write integrity tests and data report.

Do not train full models before Phase 1 passes.

### Phase 2: baseline pipeline

- Implement transforms and dataloaders.
- Implement EfficientNet-B0 and ResNet-18.
- Implement training, checkpointing, prediction export, and core metrics.
- Complete a synthetic smoke run and one real-data fold smoke run.

### Phase 3: counterfactual engine

- Implement margins.
- Implement sufficient, removal, swap, and sham interventions.
- Add quality metrics, caching, visual audits, and tests.

### Phase 4: proposed losses

- Implement individual causal losses.
- Add warm-up, ramping, gating, and monitoring.
- Run controlled loss tests.
- Run one-fold pilot before five-fold experiments.

### Phase 5: XAI evaluation

- Implement common attribution interface.
- Add localization and faithfulness metrics.
- Add robustness transformations and sanity checks.
- Validate target layers and normalization.

### Phase 6: preregistered experiments

- Freeze split and primary protocol.
- Run baselines and proposed method across all folds.
- Run required ablations.
- Preserve all runs and failures.

### Phase 7: external validation

- Freeze method and thresholds.
- Run untouched BUS-UCLM evaluation.
- Do not tune from these results.

### Phase 8: statistics and reporting

- Aggregate out-of-fold predictions.
- Run group-aware bootstrap and paired tests.
- Apply multiplicity correction.
- Generate tables, plots, qualitative grids, model cards, limitations, and reproducibility report.

## Reporting contract

Generate paper-ready outputs without exaggeration.

### Required tables

- dataset and split characteristics;
- internal diagnostic performance;
- external diagnostic performance;
- calibration;
- localization;
- causal faithfulness;
- robustness;
- sanity checks;
- loss ablations;
- margin ablations;
- counterfactual operator sensitivity;
- computational cost.

### Required figures

- method overview;
- counterfactual construction examples;
- intervention quality audit;
- qualitative XAI comparison;
- necessity-sufficiency-invariance distributions;
- internal versus external performance;
- failure cases;
- calibration curves;
- sanity-test degradation curves.

### Interpretation rules

Distinguish:

- association from causation;
- intervention-based evidence from proof of clinical reasoning;
- lesion dependence from medically correct reasoning;
- localization from faithfulness;
- internal validation from external generalization;
- statistical significance from practical importance.

Do not claim:

- clinical deployment readiness;
- causal identification of disease mechanisms;
- radiologist-equivalent reasoning;
- robustness to all scanners or hospitals;
- generalization beyond evaluated datasets.

## Completion gates

### Gate A: implementation complete

- required modules exist;
- unit tests pass;
- lint passes;
- smoke pipeline completes;
- CLI help works;
- no known split leakage.

### Gate B: experiment complete

- all planned folds and seeds have terminal states;
- predictions and metrics exist;
- failures are documented;
- external protocol remained frozen.

### Gate C: evidence validated

- artifact validator passes;
- sample denominators reconcile;
- metrics are finite;
- confidence intervals are reproducible;
- table values trace to run IDs;
- plots trace to prediction-level artifacts;
- no debug or synthetic run enters final tables.

### Gate D: publication package ready

- code and config documented;
- data acquisition instructions documented without redistributing restricted data;
- results report includes limitations and negative findings;
- reproducibility checklist complete;
- claims match evidence.

## Required response style for the implementing agent

During implementation, report:

1. **Current milestone**
2. **Files changed**
3. **Commands actually run**
4. **Observed evidence**
5. **Scientific risks or deviations**
6. **Next concrete action**

Never say “works,” “completed,” “improved,” or “validated” without naming the command or artifact supporting the claim.
