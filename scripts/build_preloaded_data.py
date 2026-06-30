#!/usr/bin/env python3
"""
build_preloaded_data.py — Offline build script for JobPilot shipped artifacts.

RUN ONCE on the author's machine before assembling the deliverable folder.
Never runs at app startup.

What this produces (into data/):
  vector_index.zip  — int8-quantized vectors + scale.json + job_ids  (~16 MB)
  job_meta.parquet  — display/ranking metadata with full descriptions (~10 MB)
  job_clusters.npz  — K-Means labels + centroids (~1 MB)

What it reads (on the author's machine, never shipped):
  data/preloaded_kaggle_50k.parquet — 79 MB raw corpus, stays local only

Usage:
  python scripts/build_preloaded_data.py [--source PATH] [--force]

  --source  PATH  Path to source parquet (default: data/preloaded_kaggle_50k.parquet)
  --force         Rebuild even if artifacts already exist
  --kaggle-only   Alias for running only the Kaggle build (legacy flag, same as default)
"""

from __future__ import annotations

import sys
import json
import time
import zipfile
import argparse
import logging
import tempfile
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build")

MAX_ARTIFACT_MB = 45.0

META_COLUMNS = [
    "job_id", "title", "company", "location", "city", "country",
    "remote", "seniority", "employment_type", "salary_min", "salary_max",
    "salary_midpoint", "description", "skills_extracted", "experience_required",
    "visa_possible", "date_posted", "recency_score", "source", "url",
    "job_text_clean",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path,
                   default=PROJECT_ROOT / "data" / "preloaded_kaggle_50k.parquet",
                   help="Source parquet (default: data/preloaded_kaggle_50k.parquet)")
    p.add_argument("--force", action="store_true",
                   help="Rebuild all artifacts even if they already exist")
    p.add_argument("--kaggle-only", action="store_true",
                   help="Legacy alias — same as default (builds all artifacts)")
    return p.parse_args()


def _load_source(source_path: Path) -> "pd.DataFrame":
    import pandas as pd
    if not source_path.exists():
        log.error(
            f"Source parquet not found: {source_path}\n"
            "Place preloaded_kaggle_50k.parquet in data/ (gitignored, never shipped)."
        )
        sys.exit(1)
    log.info(f"Loading source: {source_path.name}  ({source_path.stat().st_size / 1e6:.1f} MB)")
    df = pd.read_parquet(source_path)
    log.info(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
    if "job_id" not in df.columns:
        log.error("Source parquet missing 'job_id' column"); sys.exit(1)
    df = df.drop_duplicates(subset=["job_id"]).reset_index(drop=True)
    log.info(f"  After dedup: {len(df):,} rows")
    return df


def _build_job_meta(df: "pd.DataFrame", data_dir: Path, force: bool) -> "pd.DataFrame":
    """Write job_meta.parquet with display/ranking columns + full descriptions."""
    import pandas as pd
    meta_path = data_dir / "job_meta.parquet"

    if not force and meta_path.exists():
        log.info(f"job_meta.parquet exists ({meta_path.stat().st_size/1e6:.1f} MB) — skipping")
        return pd.read_parquet(meta_path)

    log.info("Building job_meta.parquet …")
    df = df.copy()

    if "salary_midpoint" not in df.columns:
        s_min = df["salary_min"].fillna(0).astype(float)
        s_max = df["salary_max"].fillna(0).astype(float)
        df["salary_midpoint"] = ((s_min + s_max) / 2).where((s_min > 0) | (s_max > 0), 0.0)

    if "job_text_clean" not in df.columns:
        log.info("  Building job_text_clean from fields …")
        from src.embeddings import build_job_text
        df["job_text_clean"] = [build_job_text(row) for _, row in df.iterrows()]

    present  = [c for c in META_COLUMNS if c in df.columns]
    missing  = set(META_COLUMNS) - set(present)
    if missing:
        log.warning(f"  Columns absent from source (will be missing): {missing}")

    meta_df = df[present].copy()

    def _norm_skills(v):
        if v is None: return []
        if hasattr(v, "tolist"): return v.tolist()
        if isinstance(v, (list, tuple)): return list(v)
        if isinstance(v, str):
            import ast
            try: return ast.literal_eval(v)
            except Exception: return []
        return []

    meta_df["skills_extracted"] = meta_df["skills_extracted"].apply(_norm_skills)

    tmp = meta_path.with_suffix(".tmp.parquet")
    meta_df.to_parquet(tmp, index=False, compression="snappy")
    tmp.replace(meta_path)

    log.info(f"  job_meta.parquet: {len(meta_df):,} rows, {meta_path.stat().st_size/1e6:.1f} MB")
    return meta_df


def _build_vector_index(df: "pd.DataFrame", data_dir: Path, force: bool) -> "np.ndarray":
    """Embed corpus, quantize int8, write vector_index.zip. Returns float32 embeddings."""
    import numpy as np
    from src.embeddings import embed, quantize_int8

    zip_path = data_dir / "vector_index.zip"

    if not force and zip_path.exists():
        log.info(f"vector_index.zip exists ({zip_path.stat().st_size/1e6:.1f} MB) — skipping")
        import json as _json
        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open("scale.json") as f:
                meta = _json.load(f)
            with zf.open("embeddings_int8.npy") as f:
                q = np.load(f)
        scale = float(meta["scale"])
        v = q.astype(np.float32) * scale
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (v / norms).astype(np.float32)

    log.info(f"Embedding {len(df):,} jobs …")
    t0 = time.time()
    texts      = df["job_text_clean"].fillna("").tolist()
    embeddings = embed(texts, batch_size=256, show_progress=True)
    log.info(f"  Embeddings: {embeddings.shape}  in {time.time()-t0:.1f}s")

    log.info("  Quantizing to int8 …")
    q, scale = quantize_int8(embeddings)
    job_ids  = df["job_id"].tolist()

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        np.save(tmp_dir / "embeddings_int8.npy", q)
        (tmp_dir / "scale.json").write_text(json.dumps({
            "scale": scale,
            "dim":   int(embeddings.shape[1]),
            "count": int(embeddings.shape[0]),
        }, indent=2))
        np.save(tmp_dir / "job_ids.npy", np.array(job_ids))

        tmp_zip = zip_path.with_suffix(".tmp.zip")
        with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            zf.write(tmp_dir / "embeddings_int8.npy", "embeddings_int8.npy")
            zf.write(tmp_dir / "scale.json",          "scale.json")
            zf.write(tmp_dir / "job_ids.npy",         "job_ids.npy")
        tmp_zip.replace(zip_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    raw_mb  = embeddings.nbytes / 1e6
    zip_mb  = zip_path.stat().st_size / 1e6
    log.info(f"  vector_index.zip: {zip_mb:.1f} MB  ({raw_mb/zip_mb:.1f}× smaller than raw float32)")
    return embeddings


def _build_clusters(embeddings: "np.ndarray", job_ids: list, data_dir: Path, force: bool) -> None:
    """K-Means → job_clusters.npz + cluster_state.json (atomic write)."""
    import numpy as np
    from datetime import datetime, timezone

    clusters_path = data_dir / "job_clusters.npz"
    state_path    = data_dir / "cluster_state.json"

    if not force and clusters_path.exists():
        log.info("job_clusters.npz exists — skipping"); return

    N_CLUSTERS = 30
    log.info(f"K-Means: {len(embeddings):,} jobs → {N_CLUSTERS} families …")
    t0 = time.time()

    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42,
                         batch_size=min(2048, len(embeddings)), n_init=5, max_iter=150)
    labels = km.fit_predict(embeddings).astype(np.int32)
    sizes  = np.bincount(labels)
    log.info(f"  Done {time.time()-t0:.1f}s — sizes min={sizes.min()} max={sizes.max()} mean={sizes.mean():.0f}")

    tmp = clusters_path.with_suffix(".tmp.npz")
    np.savez(tmp, labels=labels, centers=km.cluster_centers_.astype("float32"),
             n_clusters=np.int32(N_CLUSTERS), job_ids=np.array(job_ids))
    tmp.replace(clusters_path)

    state_path.write_text(json.dumps({
        "last_build_ts":   datetime.now(timezone.utc).isoformat(),
        "n_at_last_build": len(job_ids),
    }, indent=2))
    log.info(f"  job_clusters.npz: {clusters_path.stat().st_size/1e6:.2f} MB")


def _verify(data_dir: Path) -> bool:
    """Assert shipped artifacts exist and are within size limits."""
    SHIPPED = {
        "vector_index.zip": data_dir / "vector_index.zip",
        "job_meta.parquet": data_dir / "job_meta.parquet",
        "job_clusters.npz": data_dir / "job_clusters.npz",
        "personas.json":    data_dir / "personas.json",
    }
    FORBIDDEN_NAMES = [
        "preloaded_kaggle_50k.parquet", "jobpilot.db",
        "embeddings.npy", "faiss_index.bin",
        "jsearch_embeddings.npy", "jsearch_job_ids.npy",
    ]

    print("\n" + "=" * 60)
    print("  Build Verification")
    print("=" * 60)

    all_ok = True
    for name, path in SHIPPED.items():
        if not path.exists():
            print(f"  ❌ MISSING : {name}"); all_ok = False
        else:
            mb = path.stat().st_size / 1e6
            ok = mb < MAX_ARTIFACT_MB
            print(f"  {'✅' if ok else '❌ TOO LARGE'}  {name}  {mb:.1f} MB")
            if not ok: all_ok = False

    print()
    forbidden_present = [f for f in (data_dir / n for n in FORBIDDEN_NAMES) if f.exists()]
    if forbidden_present:
        print("  ⚠️  Present locally (gitignored — excluded from shipped folder):")
        for f in forbidden_present:
            print(f"       {f.name}  {f.stat().st_size/1e6:.1f} MB")
    else:
        print("  ✅  No forbidden files present in data/")

    print()
    if all_ok:
        print("  ✅  safe to assemble folder")
        print("      → python scripts/assemble_folder.py")
    else:
        print("  ❌  fix issues above before assembling")
    print("=" * 60 + "\n")
    return all_ok


def main() -> None:
    args     = _parse_args()
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    log.info("=" * 60)
    log.info("JobPilot — Offline Artifact Builder")
    log.info(f"Source : {args.source}")
    log.info(f"Force  : {args.force}")
    log.info("=" * 60 + "\n")

    t_start = time.time()

    df      = _load_source(args.source)
    meta_df = _build_job_meta(df, data_dir, args.force)
    embeds  = _build_vector_index(meta_df, data_dir, args.force)
    _build_clusters(embeds, meta_df["job_id"].tolist(), data_dir, args.force)

    log.info(f"\nTotal build time: {time.time()-t_start:.1f}s")
    _verify(data_dir)


if __name__ == "__main__":
    main()
