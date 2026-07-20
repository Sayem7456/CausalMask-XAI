"""Smoke tests for Phase 0."""

import sys
from pathlib import Path


def test_project_root_resolves():
    root = Path(__file__).resolve().parent.parent
    assert (root / "CausalMask-XAI.md").exists()


def test_causalmask_importable():
    src_dir = str(Path(__file__).resolve().parent.parent / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    import causalmask
    assert causalmask.__version__ == "0.1.0"


def test_reproducibility_module():
    src_dir = str(Path(__file__).resolve().parent.parent / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from causalmask.reproducibility import set_global_seed, seed_worker, configure_reproducibility
    info = configure_reproducibility(42)
    assert info["seed"] == 42
    assert info["torch_deterministic_algorithms"] is True
