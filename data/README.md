# Data

## Datasets

### BUSI (Primary)
- 780 ultrasound images from 600 women
- Normal, benign, malignant classes
- Lesion masks available for abnormal images

### BUS-UCLM (External Validation)
- 683 images from 38 patients
- 419 normal, 174 benign, 90 malignant
- Frozen as external test set

## Directory Structure

- `data/raw/archives/` — Downloaded zip/tar files (gitignored)
- `data/raw/extracted/` — Extracted images (gitignored)
- `data/manifests/` — CSV manifests with image paths, labels, patient IDs
- `data/splits/` — Train/val/test split definitions (JSON/CSV)
- `data/cache/` — Preprocessed tensors and counterfactuals (gitignored)

## Rules

- Never overwrite raw data or archives
- Never split after augmentation
- Never allow patient ID to cross partitions
- Record all data transformations in manifests
