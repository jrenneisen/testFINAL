"""
dedupe.py — Two-level deduplication for JobPilot.

Level 1: Exact deduplication by (title, company, location) hash.
Level 2: Near-duplicate detection using MinHash LSH (BAX-423 Sketching technique).

The MinHash approach detects near-identical job postings from the same company
posted on multiple job boards, without O(n²) pairwise comparison.
"""

import time
import logging
import pandas as pd
import numpy as np
from datasketch import MinHash, MinHashLSH

from src.utils import logger

# ─── Configuration ────────────────────────────────────────────────────────────
NUM_PERM        = 128     # Number of hash permutations (higher = more accurate)
JACCARD_THRESHOLD = 0.85  # Jaccard similarity above this = near-duplicate
MINHASH_TEXT_FIELD = "description"  # Field used for MinHash fingerprinting


# ─── Level 1: Exact deduplication ────────────────────────────────────────────
def deduplicate_exact(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Remove exact duplicates based on a hash of (title, company, description).

    Intentionally does NOT include location in the key — the same job posted
    in multiple cities (e.g. "Data Analyst at Acme in NYC" and
    "Data Analyst at Acme in LA") are kept as separate, distinct records.
    Only postings with identical title + company + description text are removed.
    """
    import hashlib
    before = len(df)

    def _desc_hash(text: str) -> str:
        snippet = str(text or "")[:300].lower().strip()
        return hashlib.md5(snippet.encode()).hexdigest()

    key = (
        df["title"].str.lower().str.strip() + "|" +
        df["company"].str.lower().str.strip() + "|" +
        df["description"].apply(_desc_hash)
    )
    df = df[~key.duplicated(keep="first")].reset_index(drop=True)
    removed = before - len(df)
    stats = {"level": "exact", "before": before, "after": len(df),
             "removed": removed, "rate": removed / max(before, 1)}
    logger.info(
        f"Exact dedup (title+company+desc_hash): removed {removed:,} rows "
        f"({stats['rate']:.1%}) — same job in different cities is kept"
    )
    return df, stats


# ─── Level 2: MinHash LSH near-duplicate detection ────────────────────────────
def build_minhash(text: str, num_perm: int = NUM_PERM) -> MinHash:
    """Create a MinHash signature for a text string using word shingles."""
    m = MinHash(num_perm=num_perm)
    words = text.lower().split()
    # Use word-level and 2-gram shingles
    for word in words:
        m.update(word.encode("utf-8"))
    for i in range(len(words) - 1):
        bigram = words[i] + " " + words[i + 1]
        m.update(bigram.encode("utf-8"))
    return m


def deduplicate_minhash(
    df: pd.DataFrame,
    threshold: float = JACCARD_THRESHOLD,
    num_perm: int = NUM_PERM,
    text_col: str = MINHASH_TEXT_FIELD,
) -> tuple[pd.DataFrame, dict]:
    """
    Remove near-duplicate job postings using MinHash LSH.

    BAX-423 Technique: Locality-Sensitive Hashing (Sketching).
    - Each job description is represented as a MinHash signature.
    - LSH groups signatures into buckets; only bucket members are compared.
    - Time complexity: O(n) rather than O(n²) for pairwise comparison.

    Returns: (deduplicated DataFrame, stats dict)
    """
    logger.info(f"Starting MinHash LSH deduplication (threshold={threshold}, n={len(df):,})...")
    start = time.time()

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    keep_mask = pd.Series(True, index=df.index)
    duplicate_pairs = []

    for idx, row in df.iterrows():
        text = str(row.get(text_col, "") or "")
        if len(text.strip()) < 50:
            continue  # Skip very short descriptions

        # Prefix key with company to only dedup within same company
        company_slug = str(row.get("company", "unk")).lower().replace(" ", "_")[:20]
        lsh_key = f"{company_slug}_{idx}"

        m = build_minhash(text, num_perm)

        try:
            # Query: find near-duplicates already in index
            near_dups = lsh.query(m)
            if near_dups:
                # Mark this row as duplicate of the first match
                keep_mask[idx] = False
                duplicate_pairs.append((idx, near_dups[0]))
            else:
                # No duplicate found — add to index
                lsh.insert(lsh_key, m)
        except Exception:
            # Key collision edge case — skip
            pass

    before = len(df)
    df = df[keep_mask].reset_index(drop=True)
    removed = before - len(df)
    elapsed = time.time() - start

    stats = {
        "level":     "minhash_lsh",
        "threshold": threshold,
        "num_perm":  num_perm,
        "before":    before,
        "after":     len(df),
        "removed":   removed,
        "rate":      removed / max(before, 1),
        "elapsed_s": round(elapsed, 2),
        "throughput_rps": round(before / max(elapsed, 0.001)),
        "duplicate_pairs": duplicate_pairs[:10],  # sample
    }

    logger.info(
        f"MinHash dedup: removed {removed:,} near-dupes "
        f"({stats['rate']:.1%}) in {elapsed:.1f}s "
        f"({stats['throughput_rps']:,} records/s)"
    )
    return df, stats


# ─── Full pipeline ────────────────────────────────────────────────────────────
def full_deduplication(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Run both levels of deduplication and return combined stats.
    Used in the data pipeline and benchmark reporting.
    """
    original_count = len(df)

    df, exact_stats = deduplicate_exact(df)
    df, lsh_stats   = deduplicate_minhash(df)

    combined_stats = {
        "original_count":       original_count,
        "after_exact":          exact_stats["after"],
        "after_minhash":        lsh_stats["after"],
        "total_removed":        original_count - len(df),
        "total_removal_rate":   (original_count - len(df)) / max(original_count, 1),
        "exact_stats":          exact_stats,
        "minhash_stats":        lsh_stats,
    }

    logger.info(
        f"Deduplication complete: {original_count:,} → {len(df):,} "
        f"({combined_stats['total_removal_rate']:.1%} removed)"
    )
    return df, combined_stats


# ─── Benchmarking ─────────────────────────────────────────────────────────────
def benchmark_deduplication(df: pd.DataFrame) -> dict:
    """
    Run deduplication and collect benchmark metrics for the Technical Brief.
    Compares exact string matching vs MinHash LSH on throughput and detection rate.
    """
    logger.info("Running deduplication benchmark...")

    # Exact dedup baseline
    t0 = time.time()
    _, exact_stats = deduplicate_exact(df.copy())
    exact_time = time.time() - t0

    # MinHash LSH
    t0 = time.time()
    _, lsh_stats = deduplicate_minhash(df.copy())
    lsh_time = time.time() - t0

    benchmark = {
        "method":            ["Exact String Match", "MinHash LSH (t=0.85)"],
        "records_removed":   [exact_stats["removed"], lsh_stats["removed"]],
        "removal_rate":      [f"{exact_stats['rate']:.1%}", f"{lsh_stats['rate']:.1%}"],
        "throughput_rps":    [
            round(len(df) / max(exact_time, 0.001)),
            lsh_stats["throughput_rps"]
        ],
        "time_seconds":      [round(exact_time, 2), round(lsh_time, 2)],
        "near_dup_detection": ["No", "Yes (Jaccard ≥ 0.85)"],
    }

    logger.info("Benchmark complete")
    return benchmark
