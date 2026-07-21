"""Dataset discovery and manifest generation from extracted raw directories.

Auto-discovers the actual folder structure under data/raw/extracted/,
pairs images with masks, normalizes labels, and writes manifests
without renaming, moving, or flattening any raw files.

Usage:
    python -m src.causalmask.data.discover [--project-root PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .manifest import (
    discover_busi_files,
    discover_bus_uclm_files,
    build_manifest,
    validate_manifest,
    save_manifest_parquet,
    save_manifest_summary,
    records_to_dicts,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def resolve_project_root() -> Path:
    """Resolve project root by walking upward from cwd."""
    cwd = Path.cwd()
    for candidate in [cwd] + list(cwd.parents):
        if (candidate / "CausalMask-XAI.md").exists():
            return candidate.resolve()
    raise RuntimeError(
        "Cannot resolve project root. Run from within the repo "
        "or pass --project-root."
    )


def load_dataset_sources(project_root: Path) -> dict:
    """Load dataset_sources.json if it exists."""
    sources_path = project_root / "data" / "raw" / "dataset_sources.json"
    if sources_path.exists():
        with open(sources_path) as f:
            return json.load(f)
    return {}


def find_extracted_dirs(project_root: Path) -> dict[str, Path]:
    """Auto-discover extraction directories under data/raw/extracted/.

    Does NOT assume directory names. Inspects actual contents to identify
    each dataset by its internal structure.
    """
    extracted_root = project_root / "data" / "raw" / "extracted"
    if not extracted_root.exists():
        raise FileNotFoundError(
            f"Extraction directory not found: {extracted_root}\n"
            "Run phase 01 notebook first."
        )

    discovered: dict[str, Path] = {}

    for child in sorted(extracted_root.iterdir()):
        if not child.is_dir():
            continue

        # Check for BUSI signature: Dataset_BUSI_with_GT/<label>/*.png
        busi_marker = child / "Dataset_BUSI_with_GT"
        if busi_marker.exists():
            discovered["busi"] = child
            logger.info(f"Identified BUSI: {child}")
            continue

        # Check for BUS-UCLM signatures
        uclm_markers = [
            child / "bus_uclm_separated",
            child / "BUS-UCLM Breast ultrasound lesion segmentation dataset",
        ]
        for marker in uclm_markers:
            if marker.exists():
                discovered["bus_uclm"] = child
                logger.info(f"Identified BUS-UCLM: {child}")
                break

        if "bus_uclm" not in discovered:
            # Fallback: check if any subdir has label dirs
            for sub in child.iterdir():
                if sub.is_dir() and sub.name.lower() in ("benign", "malignant", "normal"):
                    # Heuristic: if we already have busi, this is likely bus_uclm
                    if "busi" in discovered and "bus_uclm" not in discovered:
                        discovered["bus_uclm"] = child
                        logger.info(f"Identified BUS-UCLM (heuristic): {child}")
                    break

    return discovered


def discover_and_build(
    project_root: Path,
    output_dir: Path | None = None,
) -> dict:
    """Run full discovery and manifest build pipeline.

    Returns a summary dict with results for both datasets.
    """
    extracted_dirs = find_extracted_dirs(project_root)
    if not extracted_dirs:
        raise RuntimeError("No datasets found under data/raw/extracted/")

    sources_meta = load_dataset_sources(project_root)
    sources_list = {s["name"].lower().replace("-", "_"): s for s in sources_meta.get("sources", [])}

    output_dir = output_dir or project_root / "data" / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # --- BUSI ---
    if "busi" in extracted_dirs:
        busi_extracted = extracted_dirs["busi"]
        logger.info(f"Discovering BUSI files in {busi_extracted} ...")
        busi_samples = discover_busi_files(busi_extracted, dataset_name="busi")
        logger.info(f"Found {len(busi_samples)} BUSI image candidates")

        if busi_samples:
            busi_records = build_manifest(
                busi_samples, project_root, "busi", patient_id_prefix=""
            )
            busi_validation = validate_manifest(busi_records)

            save_manifest_parquet(busi_records, output_dir / "busi.parquet")
            save_manifest_summary(
                busi_validation, busi_records, "busi", output_dir / "busi_summary.json"
            )

            results["busi"] = {
                "extracted_dir": str(busi_extracted),
                "total_discovered": len(busi_samples),
                "manifest_rows": len(busi_records),
                "validation": busi_validation["issue_counts"],
            }
            logger.info(
                f"BUSI: {len(busi_records)} records, "
                f"issues: {busi_validation['issue_counts']}"
            )
        else:
            results["busi"] = {"error": "No files discovered"}
            logger.warning("BUSI discovery returned 0 files")

    # --- BUS-UCLM ---
    if "bus_uclm" in extracted_dirs:
        uclm_extracted = extracted_dirs["bus_uclm"]
        logger.info(f"Discovering BUS-UCLM files in {uclm_extracted} ...")
        uclm_samples = discover_bus_uclm_files(uclm_extracted, dataset_name="bus_uclm")
        logger.info(f"Found {len(uclm_samples)} BUS-UCLM image candidates")

        if uclm_samples:
            uclm_records = build_manifest(
                uclm_samples, project_root, "bus_uclm", patient_id_prefix=""
            )
            uclm_validation = validate_manifest(uclm_records)

            save_manifest_parquet(uclm_records, output_dir / "bus_uclm.parquet")
            save_manifest_summary(
                uclm_validation, uclm_records, "bus_uclm",
                output_dir / "bus_uclm_summary.json",
            )

            results["bus_uclm"] = {
                "extracted_dir": str(uclm_extracted),
                "total_discovered": len(uclm_samples),
                "manifest_rows": len(uclm_records),
                "validation": uclm_validation["issue_counts"],
            }
            logger.info(
                f"BUS-UCLM: {len(uclm_records)} records, "
                f"issues: {uclm_validation['issue_counts']}"
            )
        else:
            results["bus_uclm"] = {"error": "No files discovered"}
            logger.warning("BUS-UCLM discovery returned 0 files")

    # Save combined manifest manifest
    all_records = []
    if "busi" in extracted_dirs and "busi" in results and "manifest_rows" in results["busi"]:
        all_records.extend(
            records_to_dicts(
                build_manifest(
                    discover_busi_files(extracted_dirs["busi"], "busi"),
                    project_root,
                    "busi",
                )
            )
        )
    if "bus_uclm" in extracted_dirs and "bus_uclm" in results and "manifest_rows" in results["bus_uclm"]:
        all_records.extend(
            records_to_dicts(
                build_manifest(
                    discover_bus_uclm_files(extracted_dirs["bus_uclm"], "bus_uclm"),
                    project_root,
                    "bus_uclm",
                )
            )
        )

    if all_records:
        import pandas as pd
        combined_path = output_dir / "combined.parquet"
        try:
            pd.DataFrame(all_records).to_parquet(combined_path, index=False)
            logger.info(f"Saved combined manifest: {combined_path} ({len(all_records)} rows)")
            results["combined"] = {"total_rows": len(all_records), "path": str(combined_path)}
        except ImportError:
            csv_path = combined_path.with_suffix(".csv")
            pd.DataFrame(all_records).to_csv(csv_path, index=False)
            logger.warning(f"pyarrow not available; saved CSV: {csv_path} ({len(all_records)} rows)")
            results["combined"] = {"total_rows": len(all_records), "path": str(csv_path)}

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover datasets and generate manifests from extracted raw data."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root directory (auto-detected if omitted)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for manifests (default: data/manifests)",
    )
    args = parser.parse_args()

    project_root = args.project_root or resolve_project_root()
    logger.info(f"Project root: {project_root}")

    try:
        results = discover_and_build(project_root, args.output_dir)
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(str(e))
        sys.exit(1)

    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
