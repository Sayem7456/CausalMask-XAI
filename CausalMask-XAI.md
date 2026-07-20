# CausalMask-XAI: Does a Breast-Ultrasound Classifier Actually Use the Lesion?

## Core research question

Can we distinguish a genuinely lesion-focused medical classifier from one that produces visually attractive explanations while relying on background texture, scanner artefacts, annotations, or other shortcuts?

## Research gap

Most medical-imaging XAI studies generate Grad-CAM-style heatmaps and judge them visually. However, visually plausible saliency maps are not necessarily faithful to the model: some attribution methods can remain similar even when model parameters or labels are randomized. ([arxiv.org](https://arxiv.org/abs/1810.03292))

Breast-ultrasound XAI has progressed toward concept bottlenecks, lesion-aware explanations, and combined attribution–uncertainty frameworks. For example, a MICCAI 2024 study used BI-RADS concepts for interpretable lesion classification, while another framework combined attribution with uncertainty quantification. ([papers.miccai.org](https://papers.miccai.org/miccai-2024/449-Paper4008.html)) However, a focused framework that tests **lesion necessity, lesion sufficiency, and invariance to background replacement** using real lesion masks appears comparatively underexplored.

The proposed novelty is therefore not another saliency method. It is a **causal auditing and training framework for determining whether explanations correspond to lesion-dependent decisions**.

## Proposed hypothesis

> A trustworthy breast-ultrasound classifier should maintain its prediction when irrelevant background tissue changes, preserve much of its prediction from the lesion and immediate margin alone, and lose confidence when the lesion is removed.

A model trained to satisfy these properties should produce more faithful explanations and generalize better to images from another hospital or ultrasound device.

## Public, small datasets

### Primary dataset: BUSI

- 780 ultrasound images from 600 women.
- Normal, benign, and malignant classes.
- Lesion masks are available for abnormal images.
- Approximately 647 lesion-containing benign or malignant images can be used for the main binary experiment. ([pmc.ncbi.nlm.nih.gov](https://pmc.ncbi.nlm.nih.gov/articles/PMC6906728/?utm_source=chatgpt.com))

### External validation dataset: BUS-UCLM

- 683 images from 38 patients.
- 419 normal, 174 benign, and 90 malignant images.
- Includes lesion annotations.
- Use its 264 benign and malignant images only as an external test set. ([nature.com](https://www.nature.com/articles/s41597-025-04562-3?utm_source=chatgpt.com))

The combined experiment remains small enough for a single GPU, while external validation makes the work considerably stronger.

## Proposed methodology

### A. Baseline classifier

Use a lightweight pretrained backbone:

- EfficientNet-B0 as the main model.
- ResNet-18 as an architecture-generalization baseline.
- Binary classification: benign versus malignant.
- Five-fold stratified cross-validation on BUSI.
- Test the final fold models directly on BUS-UCLM without retraining.

Apply moderate ultrasound-compatible augmentation:

- Horizontal flipping.
- Small rotations and translations.
- Gamma and contrast adjustment.
- Mild Gaussian or speckle noise.
- No aggressive cropping that removes parts of the lesion.

### B. Generate three mask-conditioned counterfactual images

For each image \(x\) and lesion mask \(M\), construct:

#### 1. Lesion-sufficient image

Keep the lesion and a small dilated perilesional margin, while blurring or replacing the remaining background.

\[
x_{	ext{sufficient}} =
x \odot M^{+} +
B(x)\odot(1-M^{+})
\]

Here, \(M^{+}\) is the dilated lesion mask and \(B(x)\) is a blurred background.

The model should retain much of its original class confidence.

#### 2. Lesion-removed image

Remove the lesion region and fill it using neighbouring ultrasound texture through inpainting, patch sampling, or diffusion-free OpenCV methods.

\[
x_{	ext{removed}} =
x\odot(1-M^{+})+
I(x,M^{+})\odot M^{+}
\]

The model’s original confidence should fall substantially.

#### 3. Background-swapped image

Preserve the lesion and margin but replace the external background with tissue from another image.

\[
x_{	ext{swap}} =
x\odot M^{+}+
x_j\odot(1-M^{+})
\]

The prediction should remain stable even when \(x_j\) comes from the opposite class. This directly tests whether the classifier is exploiting class-correlated background signals.

## C. Introduce the **CausalMask Score**

Create a per-image explanation-reliability score with four components.

### 1. Lesion necessity

\[
N = p_y(x)-p_y(x_{	ext{removed}})
\]

A larger confidence drop indicates that the lesion was necessary for the decision.

### 2. Lesion sufficiency

\[
S = 1-\left|p_y(x)-p_y(x_{	ext{sufficient}})ight|
\]

A small confidence change means the lesion and its margin contain sufficient diagnostic evidence.

### 3. Background invariance

\[
B = 1-\left|p_y(x)-p_y(x_{	ext{swap}})ight|
\]

A high value indicates low dependence on irrelevant background tissue.

### 4. Explanation localization

Measure how much attribution lies inside the lesion and perilesional margin:

\[
L =
rac{\sum A(x)\odot M^{+}}
{\sum A(x)+\epsilon}
\]

where \(A(x)\) can be Grad-CAM++, Integrated Gradients, or RISE.

Combine them using a harmonic or weighted mean:

\[
	ext{CausalMask}(x)
=
H(N,S,B,L)
\]

The harmonic mean is useful because a model cannot obtain a strong score by performing well on only one property.

## D. Causal explanation regularization

Train a second version of the classifier with a lightweight regularizer:

\[
\mathcal{L}
=
\mathcal{L}_{CE}
+\lambda_1\mathcal{L}_{sufficiency}
+\lambda_2\mathcal{L}_{necessity}
+\lambda_3\mathcal{L}_{background}
\]

Where:

\[
\mathcal{L}_{sufficiency}
=
D_{KL}\left(
p(x)\parallel p(x_{	ext{sufficient}})
ight)
\]

\[
\mathcal{L}_{background}
=
D_{KL}\left(
p(x)\parallel p(x_{	ext{swap}})
ight)
\]

\[
\mathcal{L}_{necessity}
=
\max\left(
0,\,
p_y(x_{	ext{removed}})
-p_y(x)+m
ight)
\]

The method does not require a new large network. It only adds counterfactual samples and consistency losses to an ordinary classifier.

A safer variant would apply the necessity loss only to correctly classified, high-confidence training examples after several warm-up epochs.

## Experimental comparison

Compare:

1. Standard cross-entropy classifier.
2. Attention-supervised classifier that concentrates Grad-CAM within the lesion.
3. Proposed causal-regularized classifier.
4. Optional lesion-crop-only classifier.

Evaluate explanations from:

- Grad-CAM.
- Grad-CAM++.
- Integrated Gradients.
- RISE.

This separates improvements caused by the classifier from improvements caused by a particular XAI algorithm.

## Evaluation metrics

### Diagnostic performance

- AUROC.
- Balanced accuracy.
- Sensitivity.
- Specificity.
- F1-score.
- Expected calibration error.

### Explanation localization

- Attribution mass inside the lesion.
- Pointing-game accuracy.
- Saliency–mask soft Dice.
- Saliency–mask IoU.

### Faithfulness

- Insertion AUC.
- Deletion AUC.
- Lesion necessity.
- Lesion sufficiency.
- Background invariance.
- Overall CausalMask score.

### Explanation robustness

Measure saliency-map similarity after transformations that should not change the diagnosis:

- Horizontal flip.
- Mild contrast adjustment.
- Gamma correction.
- Small translation.
- Mild speckle noise.

Use structural similarity, rank correlation, and top-attribution overlap.

### Sanity testing

Randomize model layers progressively and verify that explanations change. This prevents reporting attractive but model-insensitive saliency maps. ([arxiv.org](https://arxiv.org/abs/1810.03292))

### Statistical analysis

- Patient-level splitting wherever identifiers permit it.
- Bootstrap 95% confidence intervals.
- Paired Wilcoxon tests across test images.
- Holm correction for multiple XAI comparisons.

Also perform near-duplicate detection before splitting to reduce leakage risk.

## Essential ablation studies

- Necessity loss only.
- Sufficiency loss only.
- Background-invariance loss only.
- All three losses.
- Exact lesion mask versus mask dilated by 5%, 10%, and 20%.
- Blurred background versus swapped background.
- Ground-truth mask versus optional predicted mask.
- EfficientNet-B0 versus ResNet-18.
- BUSI internal testing versus BUS-UCLM external testing.

The mask-dilation experiment is particularly important because malignant diagnosis may depend on lesion margins, posterior shadowing, and nearby tissue rather than only pixels strictly inside the annotation.

## Expected contribution

A successful study would contribute:

1. **A mask-conditioned causal audit** for medical-image classifiers.
2. **The CausalMask score**, measuring necessity, sufficiency, localization, and background invariance.
3. **A lightweight causal regularization strategy** that discourages background shortcuts.
4. **External validation** across two small public breast-ultrasound datasets.
5. Evidence that classification accuracy and visual localization alone are insufficient for evaluating medical XAI.

The strongest expected result is not necessarily a large AUROC improvement. A publishable outcome would be showing that two similarly accurate classifiers have substantially different causal reliability, and that the proposed regularizer improves external generalization or explanation faithfulness.

## Twelve-week execution plan

| Weeks | Work |
|---|---|
| 1–2 | Literature review, dataset audit, duplicate detection, patient-aware split preparation |
| 3–4 | EfficientNet-B0 and ResNet-18 baselines with five-fold cross-validation |
| 5 | Grad-CAM++, Integrated Gradients and RISE implementation |
| 6 | Lesion-only, lesion-removed and background-swapped generator |
| 7 | CausalMask metric and faithfulness evaluation |
| 8–9 | Causal regularization training and hyperparameter selection |
| 10 | BUS-UCLM external validation and robustness experiments |
| 11 | Ablations, bootstrap confidence intervals and statistical tests |
| 12 | Figures, paper writing, code cleanup and reproducibility documentation |

## Computational feasibility

A single 12–24 GB GPU is sufficient.

- Image resolution: 224×224 or 256×256.
- Batch size: approximately 16–32.
- Backbones: EfficientNet-B0 and ResNet-18.
- Five-fold training can be completed sequentially.
- Counterfactuals can be generated offline and cached.
- No large language model, foundation-model training, or costly 3D processing is required.

## Publication positioning

**Most realistic primary target:** IEEE International Symposium on Biomedical Imaging.

**Other suitable targets:** IEEE CBMS, IEEE BHI, ICPR medical-imaging tracks, or a strong MICCAI workshop on trustworthy, human-centred, or explainable medical AI.

A full MICCAI submission would be more ambitious and would require especially rigorous external validation, leakage control, statistical testing, and a clear demonstration that CausalMask reveals failures missed by conventional XAI metrics.
