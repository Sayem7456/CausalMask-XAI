# Data Audit Report — Phase 2

**Generated:** 2026-07-21T06:13:47.461182+00:00
**Project root:** `/content/CausalMask-XAI`

---

## Dataset Detection Status

- **busi:** extracted
  - Path: `/content/CausalMask-XAI/data/raw/extracted/busi`
- **bus_uclm:** extracted
  - Path: `/content/CausalMask-XAI/data/raw/extracted/bus_uclm`

---

## Discovered Samples

### BUSI

- Total image samples: 780
  - benign: 437
  - malignant: 210
  - normal: 133

### BUS-UCLM

- Total image samples: 593
  - benign: 174
  - normal: 419

---

## Manifest Summary

- **BUSI records:** 780
- **BUS-UCLM records:** 593
- **Total records:** 1373
- **Primary task samples:** 821
- **Excluded normal:** 552

### Validation Results

- duplicate_sample_ids: 0
- duplicate_image_paths: 0
- labels_not_recognized: 0
- missing_masks_for_primary: 0

### Integrity Checks

- [PASS] all_images_readable
- [PASS] primary_abnormal_have_masks
- [PASS] paths_are_unique
- [PASS] labels_recognized
- [PASS] raw_checksums_unchanged

---

## Quality Flags

- **BUSI flag distribution:** {'empty_mask': 133, 'very_small_mask': 66}
- **BUS-UCLM flag distribution:** {'empty_mask': 593}

---

## Generated Artifacts

- `data/manifests/busi_manifest_v1.parquet`
- `data/manifests/bus_uclm_manifest_v1.parquet`
- `data/manifests/busi_manifest_summary_v1.json`
- `data/manifests/bus_uclm_manifest_summary_v1.json`
- `reports/data_audit.md` (this file)
- `reports/results/data_audit_grids/` (PNG grids)
- `artifacts/phases/phase_02_status.json`

---

## Rules Verification

- [ ] No hard-coded sample counts (all from actual files)
- [ ] Normal images retained with `included_in_primary_task = false`
- [ ] No normal images mapped to benign
- [ ] Problems flagged rather than silently deleted
- [ ] No train/validation/test splits created
- [ ] No models trained
- [ ] Raw files unchanged
- [ ] BUS-UCLM treated as frozen external validation

---

## Phase Gate Status

**Phase 2 gate: PASSED**
