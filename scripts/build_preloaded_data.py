#!/usr/bin/env python3
"""
build_preloaded_data.py — One-time builder for JobPilot pre-loaded datasets.

Run this once from the project root.  It auto-reads your credentials from
the same places the app does — no extra setup required:

  • JSearch key  →  .streamlit/secrets.toml  OR  .env  OR  env var JSEARCH_API_KEY
  • Kaggle creds →  ~/.kaggle/kaggle.json    OR  env vars KAGGLE_USERNAME + KAGGLE_KEY

Output files (saved to data/ automatically):
  data/preloaded_kaggle_50k.parquet   — Kaggle TechMap snapshot (training corpus)
  data/preloaded_jsearch_50k.parquet  — JSearch job snapshot   (match pool)

Usage:
    python scripts/build_preloaded_data.py

The app auto-detects both files on startup and skips the upload step entirely.
Re-run with --force to refresh the JSearch snapshot with newer postings.
"""

import sys
import os
import argparse
import logging
import hashlib
from pathlib import Path

# ── Locate project root regardless of where the script is called from ─────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR    = PROJECT_ROOT / "data"
TARGET_SIZE = 50_000
DATA_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# AUTO-CREDENTIAL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _load_jsearch_key() -> str:
    """
    Find the JSearch API key from any of the standard locations:
      1. JSEARCH_API_KEY environment variable (already set)
      2. .env file in the project root (loaded via python-dotenv)
      3. .streamlit/secrets.toml in the project root (same file the app uses)
    Returns the key string, or "" if not found.
    """
    # 1. Already in environment
    key = os.environ.get("JSEARCH_API_KEY", "")
    if key:
        log.info("JSearch key loaded from environment variable.")
        return key

    # 2. .env file
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
            key = os.environ.get("JSEARCH_API_KEY", "")
            if key:
                log.info(f"JSearch key loaded from {env_file}.")
                return key
        except ImportError:
            # Parse manually — no dotenv needed
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("JSEARCH_API_KEY"):
                    key = line.split("=", 1)[-1].strip().strip('"').strip("'")
                    if key:
                        log.info(f"JSearch key loaded from {env_file} (manual parse).")
                        return key

    # 3. .streamlit/secrets.toml
    secrets_file = PROJECT_ROOT / ".streamlit" / "secrets.toml"
    if secrets_file.exists():
        try:
            import tomllib  # Python 3.11+
        except ImportError:
            try:
                import tomli as tomllib  # pip install tomli
            except ImportError:
                tomllib = None

        if tomllib:
            try:
                with open(secrets_file, "rb") as f:
                    secrets = tomllib.load(f)
                key = secrets.get("JSEARCH_API_KEY", "")
                if key:
                    log.info(f"JSearch key loaded from {secrets_file}.")
                    return key
            except Exception as e:
                log.warning(f"Could not parse secrets.toml: {e}")
        else:
            # Fallback: grep for the key manually
            for line in secrets_file.read_text().splitlines():
                if "JSEARCH_API_KEY" in line:
                    key = line.split("=", 1)[-1].strip().strip('"').strip("'")
                    if key:
                        log.info(f"JSearch key loaded from {secrets_file} (manual parse).")
                        return key

    return ""


def _check_kaggle_creds() -> bool:
    """
    Return True if Kaggle credentials are available in any standard location.
    kagglehub handles the actual auth — this just provides a helpful message.
    """
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        log.info(f"Kaggle credentials found at {kaggle_json}.")
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        log.info("Kaggle credentials loaded from environment variables.")
        return True
    log.warning(
        "Kaggle credentials not found.\n"
        "  Option A: Download kaggle.json from kaggle.com → Account → API\n"
        f"            and place it at {kaggle_json}\n"
        "  Option B: Set KAGGLE_USERNAME and KAGGLE_KEY environment variables."
    )
    return False


# ══════════════════════════════════════════════════════════════════════════════
# SHARED SCHEMA UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

REQUIRED_COLS = [
    "job_id", "title", "company", "location", "description",
    "employment_type", "seniority", "salary_min", "salary_max",
    "salary_midpoint", "remote", "skills", "experience_required",
    "education_required", "date_posted", "source", "visa_possible",
]


def _fill_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure every required column exists with a sensible default."""
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = {
                "employment_type": "Full-time",
                "seniority":       "mid",
                "salary_min":      0.0,
                "salary_max":      0.0,
                "salary_midpoint": 0.0,
                "remote":          False,
                "experience_required": 0,
                "education_required":  "Not specified",
                "source":          "unknown",
                "visa_possible":   False,
            }.get(col, "")

    # job_id fallback
    mask = df["job_id"].isna() | (df["job_id"].astype(str).str.strip() == "")
    if mask.any():
        df.loc[mask, "job_id"] = (
            df.loc[mask, "title"].astype(str) + df.loc[mask, "company"].astype(str)
        ).apply(lambda x: hashlib.md5(x.encode()).hexdigest()[:16])

    # Skills must be lists
    df["skills"] = df["skills"].apply(
        lambda x: x if isinstance(x, list)
        else [s.strip() for s in str(x).split(",") if s.strip()]
        if x and not isinstance(x, float) else []
    )

    for col in ["salary_min", "salary_max", "salary_midpoint"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    has_mid = df["salary_midpoint"] > 0
    df.loc[~has_mid, "salary_midpoint"] = (
        df.loc[~has_mid, "salary_min"] + df.loc[~has_mid, "salary_max"]
    ) / 2

    df["remote"]        = df["remote"].astype(bool)
    df["visa_possible"] = df["visa_possible"].astype(bool)
    df["date_posted"]   = pd.to_datetime(df["date_posted"], errors="coerce").fillna(
        pd.Timestamp.now()
    )

    return df[REQUIRED_COLS].reset_index(drop=True)


def _infer_seniority(title: str) -> str:
    t = str(title).lower()
    if any(k in t for k in ["intern", "entry", "junior", "jr."]):
        return "junior"
    if any(k in t for k in ["senior", "sr.", "lead", "principal",
                              "director", "vp ", "head of", "chief"]):
        return "senior"
    return "mid"


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 1 — KAGGLE TECHMAP (training corpus)
# ══════════════════════════════════════════════════════════════════════════════

def build_kaggle_20k(out_path: Path) -> pd.DataFrame:
    """
    Download Kaggle TechMap, clean with the app's own pipeline, sample 20k rows.
    These jobs train the embedding model — they are NOT shown as match results.
    """
    log.info("━━━  Building Kaggle training corpus  ━━━")

    if not _check_kaggle_creds():
        raise RuntimeError("Kaggle credentials missing — see instructions above.")

    try:
        import kagglehub, glob
    except ImportError:
        raise ImportError("Run: pip install kagglehub")

    log.info("Downloading from Kaggle (cached after first run)…")
    path      = kagglehub.dataset_download("techmap/international-job-postings-september-2021")
    csv_files = sorted(glob.glob(f"{path}/**/*.csv", recursive=True))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs found under {path}")

    log.info(f"Found {len(csv_files)} CSV file(s)")
    chunks, per_file = [], max(TARGET_SIZE // max(len(csv_files), 1) + 1_000, 5_000)
    for f in csv_files:
        try:
            chunks.append(pd.read_csv(f, nrows=per_file, low_memory=False))
            if sum(len(c) for c in chunks) >= TARGET_SIZE * 1.2:
                break
        except Exception as e:
            log.warning(f"  Skipping {f}: {e}")

    raw = pd.concat(chunks, ignore_index=True)
    log.info(f"Raw rows: {len(raw):,}")

    try:
        from src.clean import clean_jobs
        df = clean_jobs(raw, save=False)
        log.info(f"After app clean pipeline: {len(df):,} rows")
    except Exception as e:
        log.warning(f"src.clean unavailable ({e}) — using fallback normaliser")
        df = _kaggle_fallback(raw)

    if len(df) > TARGET_SIZE:
        df = df.sample(TARGET_SIZE, random_state=42).reset_index(drop=True)

    df["source"] = "kaggle"
    df = _fill_schema(df)
    df.to_parquet(out_path, index=False)
    log.info(f"✅  {len(df):,} rows  →  {out_path.name}  "
             f"({out_path.stat().st_size / 1e6:.1f} MB)")
    return df


def _kaggle_fallback(raw: pd.DataFrame) -> pd.DataFrame:
    rename = {"jobTitle": "title", "companyName": "company",
               "jobLocation": "location", "jobDescription": "description",
               "employmentType": "employment_type",
               "salaryMin": "salary_min", "salaryMax": "salary_max",
               "datePosted": "date_posted"}
    df = raw.rename(columns={k: v for k, v in rename.items() if k in raw.columns}).copy()
    if "job_id" not in df.columns:
        df["job_id"] = (df.get("title", "").astype(str) +
                        df.get("company", "").astype(str)).apply(
            lambda x: hashlib.md5(x.encode()).hexdigest()[:16] + "_kg")
    df["seniority"] = df.get("title", "").apply(_infer_seniority)
    df["remote"]    = df.get("location", "").str.lower().str.contains("remote", na=False)
    df["source"]    = "kaggle"
    df["skills"]    = [[] for _ in range(len(df))]
    df = df[df.get("description", pd.Series([""] * len(df))).fillna("").str.len() > 50]
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 2 — JSEARCH SNAPSHOT (match pool)
# ══════════════════════════════════════════════════════════════════════════════

# Broad employment-signal queries — intentionally NOT job-title specific.
# The snapshot should contain diverse job types so that embedding similarity
# (not keyword matching) does all the relevance work at query time.
# Users' actual job searches (e.g. "data scientist") are used only as part of
# the candidate's profile embedding — NOT to filter which jobs appear here.
DEFAULT_QUERIES = [
    "full time",
    "remote",
    "entry level",
    "senior",
    "manager",
    "analyst",
    "engineer",
    "developer",
    "specialist",
    "associate",
]


def build_jsearch_20k(out_path: Path, queries: list) -> pd.DataFrame:
    """
    Fetch real US job postings from JSearch, clean them, save as a static snapshot.
    country="us" is hardcoded — the pool covers only US-based roles so results are
    relevant for US job seekers. These jobs are ranked and shown to users at query time.
    Not live — reflects build date.
    """
    log.info("━━━  Building JSearch match-pool snapshot (USA only)  ━━━")

    key = _load_jsearch_key()
    if not key:
        raise RuntimeError(
            "JSearch API key not found.\n"
            "  Add it to any of these (same files the app uses):\n"
            f"    {PROJECT_ROOT / '.env'}                    →  JSEARCH_API_KEY=xxx\n"
            f"    {PROJECT_ROOT / '.streamlit' / 'secrets.toml'}  →  JSEARCH_API_KEY = \"xxx\""
        )

    # Inject the key so src.ingest picks it up
    os.environ["JSEARCH_API_KEY"] = key

    try:
        from src.ingest import fetch_multiple_queries
        from src.clean import clean_jobs
    except ImportError as e:
        raise ImportError(f"Cannot import app modules: {e}. Run from the project root.")

    # Hard cap: 5 pages per query × 10 results/page × 15 queries = ~750 jobs max.
    # This keeps API usage well within free-tier limits (~75 requests total).
    # JSearch does NOT paginate a giant DB — each request returns fresh results.
    # 10 pages × 10 queries × 10 results/page = ~1,000 raw jobs from JSearch.
    # JSearch caps results per query at ~100 unique, so going beyond 10 pages
    # yields diminishing returns. 100 total API calls stays within free-tier limits.
    PAGES_PER_QUERY = 10
    est_jobs = len(queries) * PAGES_PER_QUERY * 10
    log.info(f"{len(queries)} queries × {PAGES_PER_QUERY} pages each  (~{est_jobs} raw jobs expected)")
    log.info(f"Total API calls: ~{len(queries) * PAGES_PER_QUERY}  (within free-tier limits)")

    # country="us" restricts JSearch results to US job postings only.
    # The embedding model handles all relevance matching — we don't filter by title here.
    raw = fetch_multiple_queries(queries, pages_per_query=PAGES_PER_QUERY, country="us")
    if raw.empty:
        raise RuntimeError("JSearch returned 0 results. Check your API key and rate limits.")

    log.info(f"Raw rows from JSearch: {len(raw):,}")
    df = clean_jobs(raw, save=False)
    df = df.drop_duplicates(subset=["job_id"]).reset_index(drop=True)
    log.info(f"After clean + dedup: {len(df):,} rows")

    if len(df) > TARGET_SIZE:
        df = df.sample(TARGET_SIZE, random_state=42).reset_index(drop=True)
    elif len(df) < 200:
        log.warning(
            f"Only {len(df):,} jobs fetched — very low. Check your API key and rate limits."
        )
    else:
        log.info(f"Snapshot has {len(df):,} jobs — sufficient for the match pool.")

    df["source"] = df.get("source", pd.Series(["jsearch"] * len(df))).fillna("jsearch")
    df = _fill_schema(df)
    df.to_parquet(out_path, index=False)
    log.info(f"✅  {len(df):,} rows  →  {out_path.name}  "
             f"({out_path.stat().st_size / 1e6:.1f} MB)  "
             f"[snapshot: {pd.Timestamp.now().strftime('%Y-%m-%d')}]")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description="Build JobPilot pre-loaded parquet files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Credentials are read automatically from:\n"
            "  Kaggle  → ~/.kaggle/kaggle.json  OR  KAGGLE_USERNAME + KAGGLE_KEY env vars\n"
            "  JSearch → .streamlit/secrets.toml  OR  .env  OR  JSEARCH_API_KEY env var"
        ),
    )
    p.add_argument("--kaggle-only",  action="store_true",
                   help="Build only the Kaggle training corpus")
    p.add_argument("--jsearch-only", action="store_true",
                   help="Build only the JSearch match-pool snapshot")
    p.add_argument("--queries", type=str, default="",
                   help="Comma-separated JSearch queries (overrides built-in list)")
    p.add_argument("--force", action="store_true",
                   help="Rebuild even if output files already exist")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    print()
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║  JobPilot — Pre-loaded Data Builder              ║")
    log.info("╚══════════════════════════════════════════════════╝")
    log.info(f"Project root : {PROJECT_ROOT}")
    log.info(f"Output dir   : {DATA_DIR}")
    log.info(f"Target size  : {TARGET_SIZE:,} rows per file\n")

    kaggle_out  = DATA_DIR / "preloaded_kaggle_50k.parquet"
    jsearch_out = DATA_DIR / "preloaded_jsearch_50k.parquet"

    queries = (
        [q.strip() for q in args.queries.split(",") if q.strip()]
        if args.queries.strip() else DEFAULT_QUERIES
    )

    build_kaggle  = not args.jsearch_only
    build_jsearch = not args.kaggle_only
    results       = {}

    # ── Kaggle ────────────────────────────────────────────────────────────────
    if build_kaggle:
        if kaggle_out.exists() and not args.force:
            n = len(pd.read_parquet(kaggle_out))
            log.info(f"⏭  Kaggle file already exists ({n:,} rows) — skipping.")
            log.info(f"   Use --force to rebuild: python scripts/build_preloaded_data.py --force")
            results["kaggle"] = True
        else:
            try:
                build_kaggle_20k(kaggle_out)
                results["kaggle"] = True
            except Exception as e:
                log.error(f"❌ Kaggle build failed: {e}")
                results["kaggle"] = False
        print()

    # ── JSearch ───────────────────────────────────────────────────────────────
    if build_jsearch:
        if jsearch_out.exists() and not args.force:
            n = len(pd.read_parquet(jsearch_out))
            log.info(f"⏭  JSearch snapshot already exists ({n:,} rows) — skipping.")
            log.info(f"   Use --force to refresh: python scripts/build_preloaded_data.py --force")
            results["jsearch"] = True
        else:
            try:
                build_jsearch_20k(jsearch_out, queries)
                results["jsearch"] = True
            except Exception as e:
                log.error(f"❌ JSearch build failed: {e}")
                results["jsearch"] = False
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("─" * 55)
    log.info("DONE")
    log.info("─" * 55)

    if results.get("kaggle"):
        mb = kaggle_out.stat().st_size / 1e6
        log.info(f"✅  {kaggle_out.name:40s}  {mb:.1f} MB  (training corpus)")
        if mb > 50:
            log.warning(f"   ⚠️  {mb:.0f} MB — too large for GitHub. Use Git LFS or share separately.")
        else:
            log.info(f"   ✅  GitHub-safe size — you can commit this file.")
    elif build_kaggle:
        log.info("❌  Kaggle training corpus — FAILED")

    if results.get("jsearch"):
        mb = jsearch_out.stat().st_size / 1e6
        log.info(f"✅  {jsearch_out.name:40s}  {mb:.1f} MB  (match pool snapshot)")
        if mb > 50:
            log.warning(f"   ⚠️  {mb:.0f} MB — too large for GitHub. Use Git LFS or share separately.")
        else:
            log.info(f"   ✅  GitHub-safe size — you can commit this file.")
    elif build_jsearch:
        log.info("❌  JSearch snapshot — FAILED")

    if any(results.values()):
        log.info("")
        log.info("🚀  The app will detect these files automatically on next launch.")
        log.info("    No upload needed — just start the app and run the pipeline.")
    log.info("─" * 55)
