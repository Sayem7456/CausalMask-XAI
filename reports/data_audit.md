# Data Audit Report — Phase 2

**Generated:** 2026-07-22T07:06:01.724932+00:00
**Project root:** `/home/eatl/CausalMask-XAI`

---

## Dataset Detection Status

- **busi:** missing
- **bus_uclm:** missing

---

## Discovered Samples

---

## Manifest Summary

- **BUSI records:** 0
- **BUS-UCLM records:** 0
- **Total records:** 0
- **Primary task samples:** N/A
- **Excluded normal:** N/A

### Validation Results

- (no validation run)

### Integrity Checks

- (no integrity checks run)

---

## Quality Flags


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

**Phase 2 gate: INCOMPLETE — see checks above**
