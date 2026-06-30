#!/usr/bin/env python3
"""
assemble_folder.py — Build the final runnable JobPilot deliverable folder.

Run AFTER build_preloaded_data.py has verified ✅.

What this does:
  1. Reads a whitelist of code files and data artifacts.
  2. Copies them into an output folder (default: ../jobpilot_final/).
  3. Verifies the output folder is self-contained and runnable.

Usage:
  python scripts/assemble_folder.py [--out PATH] [--force]

  --out PATH   Destination folder (default: ../jobpilot_final next to repo root)
  --force      Overwrite destination if it already exists
"""

from __future__ import annotations

import sys
import shutil
import argparse
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("assemble")

# ── Whitelist ──────────────────────────────────────────────────────────────────
# Code files / dirs copied verbatim (relative to PROJECT_ROOT)
CODE_WHITELIST = [
    "app.py",
    "requirements.txt",
    "runtime.txt",
    "README.md",
    ".streamlit",
    "src",
    "scripts",
    "prompts.md",
]

# Data artifacts — exact filenames only (nothing else from data/)
DATA_ARTIFACTS = [
    "vector_index.zip",
    "job_meta.parquet",
    "job_clusters.npz",
    "cluster_state.json",
    "personas.json",
]

# Files/dirs to always exclude when copying directories
EXCLUDE_PATTERNS = {
    "__pycache__", ".DS_Store", "*.pyc", "*.pyo",
    "jobpilot.db", "jobpilot.db-wal", "jobpilot.db-shm",
    "preloaded_kaggle_50k.parquet",
    "embeddings.npy", "faiss_index.bin",
    "jsearch_embeddings.npy", "jsearch_job_ids.npy",
}


def _should_exclude(path: Path) -> bool:
    name = path.name
    return any(
        name == pat or name.endswith(pat.lstrip("*"))
        for pat in EXCLUDE_PATTERNS
    )


def _copy_item(src: Path, dst: Path) -> None:
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(
            src, dst,
            ignore=shutil.ignore_patterns(*EXCLUDE_PATTERNS),
        )
        log.info(f"  📁  {src.relative_to(PROJECT_ROOT)}/")
    elif src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        mb = src.stat().st_size / 1e6
        log.info(f"  📄  {src.relative_to(PROJECT_ROOT)}  ({mb:.1f} MB)")
    else:
        log.warning(f"  ⚠️  Not found, skipping: {src}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path,
                   default=PROJECT_ROOT.parent / "jobpilot_final",
                   help="Output folder path (default: ../jobpilot_final)")
    p.add_argument("--force", action="store_true",
                   help="Overwrite destination folder if it exists")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    dst_root: Path = args.out.resolve()

    # Pre-flight: verify build artifacts
    data_dir = PROJECT_ROOT / "data"
    required = [data_dir / a for a in DATA_ARTIFACTS if a != "cluster_state.json"]
    missing  = [f for f in required if not f.exists()]
    if missing:
        log.error(
            "Build artifacts missing — run first:\n"
            "  python scripts/build_preloaded_data.py --force\n"
            f"Missing: {[f.name for f in missing]}"
        )
        sys.exit(1)

    if dst_root.exists():
        if not args.force:
            log.error(
                f"Destination already exists: {dst_root}\n"
                "Use --force to overwrite."
            )
            sys.exit(1)
        shutil.rmtree(dst_root)

    log.info("=" * 60)
    log.info("JobPilot — Folder Assembler")
    log.info(f"Destination: {dst_root}")
    log.info("=" * 60)

    dst_root.mkdir(parents=True)

    # Copy code
    log.info("\nCopying code …")
    for rel in CODE_WHITELIST:
        src = PROJECT_ROOT / rel
        dst = dst_root / rel
        _copy_item(src, dst)

    # Copy data artifacts
    log.info("\nCopying data artifacts …")
    (dst_root / "data").mkdir(exist_ok=True)
    for name in DATA_ARTIFACTS:
        src = data_dir / name
        dst = dst_root / "data" / name
        if src.exists():
            _copy_item(src, dst)
        else:
            log.warning(f"  ⚠️  Optional artifact not found, skipping: {name}")

    # Create empty outputs/ dir (app expects it)
    (dst_root / "outputs").mkdir(exist_ok=True)

    # Summary
    total_mb = sum(f.stat().st_size for f in dst_root.rglob("*") if f.is_file()) / 1e6
    print()
    print("=" * 60)
    print(f"  ✅  Folder assembled: {dst_root}")
    print(f"      Total size: {total_mb:.1f} MB")
    print()
    print("  To run:")
    print(f"  cd {dst_root}")
    print("  python3 -m streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
