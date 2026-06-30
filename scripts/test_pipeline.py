#!/usr/bin/env python3
"""
test_pipeline.py — Standalone pipeline smoke-test for JobPilot.

Runs the full matching pipeline without the Streamlit UI, so you can
validate changes locally in seconds without redeploying.

Usage:
    # Run against 1,000 rows of the Kaggle corpus (default)
    python scripts/test_pipeline.py

    # Use the pre-built FAISS index fast-path (tests the same code path
    # that Streamlit Cloud will use after you push data/ to GitHub)
    python scripts/test_pipeline.py --prebuilt

    # Use a larger corpus slice for a more thorough test
    python scripts/test_pipeline.py --rows 5000

    # Test a specific persona (must match a key in data/personas.json)
    python scripts/test_pipeline.py --persona "Data Scientist"

Exit codes:
    0 — pipeline completed, top-5 results printed
    1 — pipeline error (exception traceback printed)
"""

import sys
import time
import argparse
import logging
from pathlib import Path

# ── Project root on sys.path ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_pipeline")

# ─────────────────────────────────────────────────────────────────────────────
# Default test persona — matches the "Data Analyst" demo persona in personas.json
# Override with --persona flag or edit here for quick one-off tests.
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_PROFILE = {
    "name":              "Test User",
    "job_title":         "Data Analyst",
    "target_role":       "Data Analyst",
    "skills":            ["python", "sql", "tableau", "pandas", "statistics"],
    "years_experience":  3,
    "preferred_location":"San Francisco, CA",
    "remote_preference": "hybrid",
    "salary_min":        90_000,
    "salary_max":        130_000,
    "education":         "Masters",
    "visa_required":     False,
    "seniority":         "mid",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Smoke-test the JobPilot matching pipeline without Streamlit.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--rows", type=int, default=1_000,
        help="Number of corpus rows to use (default: 1000). "
             "Ignored when --prebuilt is set (uses the full pre-built index).",
    )
    p.add_argument(
        "--prebuilt", action="store_true",
        help="Test the fast-path: load data/faiss_index.bin instead of rebuilding. "
             "Requires running build_preloaded_data.py first.",
    )
    p.add_argument(
        "--persona", type=str, default="",
        help="Persona name to load from data/personas.json "
             "(e.g. 'Data Scientist'). Falls back to the built-in test profile.",
    )
    p.add_argument(
        "--top-k", type=int, default=5,
        help="Number of top matches to display (default: 5).",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show DEBUG-level log output.",
    )
    return p.parse_args()


def _load_persona(name: str) -> dict:
    """Load a named persona from data/personas.json."""
    import json
    path = PROJECT_ROOT / "data" / "personas.json"
    if not path.exists():
        log.warning(f"personas.json not found at {path} — using default profile")
        return DEFAULT_PROFILE

    personas = json.loads(path.read_text())
    # personas.json can be a list or a dict keyed by name
    if isinstance(personas, list):
        matches = [p for p in personas if p.get("name", "").lower() == name.lower()]
    else:
        matches = [v for k, v in personas.items() if k.lower() == name.lower()]

    if not matches:
        available = (
            [p.get("name", "?") for p in personas]
            if isinstance(personas, list)
            else list(personas.keys())
        )
        log.warning(
            f"Persona '{name}' not found in personas.json. "
            f"Available: {available}. Using default profile."
        )
        return DEFAULT_PROFILE

    log.info(f"Loaded persona: {matches[0].get('name', name)}")
    return matches[0]


def _load_corpus(n_rows: int) -> "pd.DataFrame":
    """Load the Kaggle pre-built parquet or fall back to CLEAN_PARQUET."""
    import pandas as pd
    from src.utils import DATA_DIR, CLEAN_PARQUET

    kaggle_preloaded = DATA_DIR / "preloaded_kaggle_50k.parquet"
    if kaggle_preloaded.exists():
        df = pd.read_parquet(kaggle_preloaded)
        log.info(f"Corpus: {kaggle_preloaded.name}  ({len(df):,} total rows)")
    elif CLEAN_PARQUET.exists():
        df = pd.read_parquet(CLEAN_PARQUET)
        log.info(f"Corpus: {CLEAN_PARQUET.name}  ({len(df):,} total rows)")
    else:
        raise FileNotFoundError(
            "No corpus file found. Run one of:\n"
            "  python scripts/build_preloaded_data.py --kaggle-only\n"
            "  streamlit run app.py  (uploads a CSV via the UI)"
        )

    if n_rows < len(df):
        df = df.sample(n_rows, random_state=42).reset_index(drop=True)
        log.info(f"Sampled to {n_rows:,} rows for this test run")

    return df


def _run_fast_path(profile: dict, top_k: int) -> list[dict]:
    """
    Fast-path: load the pre-built FAISS index and run the full scoring stack.
    This is the exact code path used by the live app when index files exist.
    """
    import numpy as np
    from src.embeddings import (
        load_prebuilt_index, build_job_clusters, get_cluster_labels,
        embed_and_score_live_jobs, tfidf_retrieve,
    )
    from src.utils import DATA_DIR, TOP_K_JOBS, RETRIEVAL_K, DEFAULT_WEIGHTS
    import pandas as pd

    log.info("Fast-path: loading pre-built FAISS index…")
    t0 = time.time()
    index, embeddings, job_ids = load_prebuilt_index()
    log.info(f"  Index loaded in {time.time()-t0:.1f}s  ({index.ntotal:,} vectors)")

    # Load corpus aligned with the index
    kaggle_preloaded = DATA_DIR / "preloaded_kaggle_50k.parquet"
    if kaggle_preloaded.exists():
        corpus_df = pd.read_parquet(kaggle_preloaded)
    else:
        from src.utils import CLEAN_PARQUET
        corpus_df = pd.read_parquet(CLEAN_PARQUET)

    # Cluster labels
    cluster_labels = get_cluster_labels(job_ids)
    if cluster_labels is None:
        log.info("  Building clusters (first time)…")
        cluster_labels = build_job_clusters(embeddings, job_ids)

    return _score_and_rank(
        profile, index, embeddings, job_ids, corpus_df,
        cluster_labels, top_k,
    )


def _run_slow_path(profile: dict, corpus_df: "pd.DataFrame", top_k: int) -> list[dict]:
    """
    Slow-path: full dedup + embed from scratch.
    Used when --prebuilt is NOT set (tests the fallback rebuild code path).
    """
    from src.dedupe import full_deduplication
    from src.embeddings import (
        load_or_build_index, build_job_clusters, get_cluster_labels,
    )

    log.info("Slow-path: deduplicating corpus…")
    t_dedup = time.time()
    corpus_df, dedup_stats = full_deduplication(corpus_df)
    log.info(
        f"  Dedup done in {time.time()-t_dedup:.1f}s — "
        f"{dedup_stats['after_minhash']:,} rows remaining"
    )

    log.info("Building / loading FAISS index…")
    t_embed = time.time()

    if "embeddable" in corpus_df.columns:
        embeddable = corpus_df[corpus_df["embeddable"] == True].copy()
    else:
        embeddable = corpus_df.copy()

    index, embeddings, job_ids = load_or_build_index(embeddable)
    log.info(f"  Embedding done in {time.time()-t_embed:.1f}s")

    cluster_labels = get_cluster_labels(job_ids)
    if cluster_labels is None:
        cluster_labels = build_job_clusters(embeddings, job_ids)

    return _score_and_rank(
        profile, index, embeddings, job_ids, corpus_df,
        cluster_labels, top_k,
    )


def _score_and_rank(
    profile: dict,
    index,
    embeddings: "np.ndarray",
    job_ids: list,
    corpus_df: "pd.DataFrame",
    cluster_labels: "np.ndarray | None",
    top_k: int,
) -> list[dict]:
    """Shared scoring + ranking used by both paths."""
    import numpy as np
    from src.embeddings import embed_and_score_live_jobs, tfidf_retrieve
    from src.utils import DEFAULT_WEIGHTS, RETRIEVAL_K

    # Build a query text from the profile
    query_parts = [
        profile.get("target_role", profile.get("job_title", "")),
        " ".join(profile.get("skills", [])),
        profile.get("preferred_location", ""),
    ]
    query_text = " ".join(p for p in query_parts if p).strip()
    if not query_text:
        query_text = "data analyst python sql"

    log.info(f"Query text: '{query_text[:80]}…'" if len(query_text) > 80 else f"Query text: '{query_text}'")

    k = min(RETRIEVAL_K, len(job_ids))

    # FAISS ANN retrieval
    t_score = time.time()
    results = embed_and_score_live_jobs(
        query_text  = query_text,
        profile     = profile,
        corpus_df   = corpus_df,
        index       = index,
        job_ids     = job_ids,
        embeddings  = embeddings,
        cluster_labels = cluster_labels,
        weights     = DEFAULT_WEIGHTS,
        top_k       = top_k,
        retrieval_k = k,
    )
    log.info(f"  Scoring done in {time.time()-t_score:.1f}s  →  {len(results)} results")
    return results


def main() -> None:
    args = _parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    log.info("=" * 60)
    log.info("JobPilot Pipeline Smoke-Test")
    log.info("=" * 60)
    log.info(f"Mode    : {'⚡ fast-path (pre-built index)' if args.prebuilt else '🐢 slow-path (rebuild)'}")
    log.info(f"Rows    : {args.rows:,}  (ignored in fast-path mode)")
    log.info(f"Top-K   : {args.top_k}")
    log.info("")

    # ── Load profile ─────────────────────────────────────────────────────────
    if args.persona:
        profile = _load_persona(args.persona)
    else:
        profile = DEFAULT_PROFILE
        log.info("Using built-in test profile (Data Analyst, San Francisco)")

    # ── Run pipeline ─────────────────────────────────────────────────────────
    t_start = time.time()

    try:
        if args.prebuilt:
            from src.embeddings import prebuilt_index_exists
            if not prebuilt_index_exists():
                log.error(
                    "Pre-built index files not found in data/.\n"
                    "Run first:  python scripts/build_preloaded_data.py"
                )
                sys.exit(1)
            results = _run_fast_path(profile, args.top_k)
        else:
            corpus_df = _load_corpus(args.rows)
            results   = _run_slow_path(profile, corpus_df, args.top_k)

    except Exception as exc:
        log.exception(f"Pipeline failed: {exc}")
        sys.exit(1)

    elapsed = time.time() - t_start

    # ── Print results ─────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  Top {args.top_k} matches  (total time: {elapsed:.1f}s)")
    print("=" * 60)

    if not results:
        print("  ⚠️  No results returned — check corpus and profile.")
        sys.exit(1)

    for i, job in enumerate(results[:args.top_k], 1):
        score    = job.get("final_score", job.get("score", 0.0))
        title    = job.get("title",    "Unknown Title")
        company  = job.get("company",  "Unknown Company")
        location = job.get("location", "Unknown Location")
        salary   = job.get("salary_midpoint", 0)
        sal_str  = f"${salary:,.0f}/yr" if salary > 0 else "n/a"
        print(f"\n  #{i}  [{score:.3f}]  {title}")
        print(f"       {company}  ·  {location}  ·  {sal_str}")

    print()
    print(f"  ✅  Pipeline completed in {elapsed:.1f}s")
    if elapsed < 90:
        print(f"  ✅  Under 90s target  ({elapsed:.1f}s < 90s)")
    else:
        print(f"  ⚠️  Over 90s target  ({elapsed:.1f}s — consider --prebuilt flag)")
    print("=" * 60)


if __name__ == "__main__":
    main()
