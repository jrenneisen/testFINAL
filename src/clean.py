"""
clean.py — Data cleaning and feature extraction for JobPilot.

Transforms raw job postings (from any source) into a standardized schema
with extracted features for embedding, ranking, and filtering.
"""

import re
import logging
import pandas as pd
import numpy as np
from datetime import datetime

from src.utils import (
    clean_text, make_job_id, extract_skills_from_text,
    detect_seniority, detect_remote, detect_experience_required,
    detect_visa_possible, parse_salary, salary_midpoint,
    CLEAN_PARQUET, SAMPLE_PARQUET, logger
)


# ─── Main cleaning pipeline ───────────────────────────────────────────────────
def clean_jobs(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Full cleaning pipeline. Takes raw DataFrame (any schema) and returns
    a clean, standardized DataFrame ready for embedding and ranking.
    """
    logger.info(f"Starting cleaning pipeline on {len(df):,} rows...")

    df = df.copy()

    # 1. Standardize column names
    df = _standardize_columns(df)

    # 2. Fill missing critical fields — never drop rows so every record is preserved
    #    for the download dataset. Records without meaningful descriptions are
    #    flagged as non-embeddable and filtered out of FAISS only.
    df["title"]       = df["title"].fillna("Unknown Position")
    df["description"] = df["description"].fillna("")
    df["title"]       = df["title"].apply(lambda x: x if str(x).strip() else "Unknown Position")
    df["description"] = df["description"].apply(lambda x: x if isinstance(x, str) else "")
    logger.info(f"After null-fill (no rows dropped): {len(df):,} rows")

    # 3. Clean text fields
    df["title"]       = df["title"].apply(_normalize_title)
    df["company"]     = df["company"].apply(lambda x: clean_text(str(x)).title() if isinstance(x, str) else "Unknown")
    df["description"] = df["description"].apply(clean_text)
    df["location"]    = df["location"].apply(lambda x: clean_text(str(x)) if isinstance(x, str) else "")

    # 4. Extract city/country
    df[["city", "country"]] = df["location"].apply(
        lambda loc: pd.Series(_parse_location(loc))
    )

    # 5. Remote detection
    if "remote" not in df.columns or df["remote"].dtype != bool:
        df["remote"] = df.apply(
            lambda row: detect_remote(str(row.get("location", "")),
                                      str(row.get("description", ""))), axis=1
        )
    else:
        df["remote"] = df["remote"].fillna(False).astype(bool)

    # 6. Salary parsing
    if "salary_raw" in df.columns:
        salaries = df["salary_raw"].apply(
            lambda s: pd.Series(parse_salary(str(s) if pd.notna(s) else ""))
        )
        df["salary_min"] = salaries[0]
        df["salary_max"] = salaries[1]
    else:
        df["salary_min"] = pd.to_numeric(df.get("salary_min", 0), errors="coerce").fillna(0)
        df["salary_max"] = pd.to_numeric(df.get("salary_max", 0), errors="coerce").fillna(0)

    df["salary_midpoint"] = df.apply(
        lambda r: salary_midpoint(r["salary_min"], r["salary_max"]), axis=1
    )

    # 7. Seniority detection
    df["seniority"] = df.apply(
        lambda r: detect_seniority(r["title"], r["description"][:300]), axis=1
    )

    # 8. Skill extraction
    df["skills_extracted"] = df.apply(
        lambda r: extract_skills_from_text(r["title"] + " " + r["description"]), axis=1
    )

    # 9. Experience required
    df["experience_required"] = df["description"].apply(detect_experience_required)

    # 10. Visa possible
    df["visa_possible"] = df.apply(
        lambda r: detect_visa_possible(r["description"], r.get("company", "")), axis=1
    )

    # 11. Employment type normalization
    if "employment_type" not in df.columns:
        df["employment_type"] = "Full-time"
    df["employment_type"] = df["employment_type"].apply(_normalize_employment_type)

    # 12. Date normalization
    if "date_posted" not in df.columns:
        df["date_posted"] = datetime.now().strftime("%Y-%m-%d")
    df["date_posted"] = df["date_posted"].apply(_normalize_date)

    # 13. Source
    if "source" not in df.columns:
        df["source"] = "unknown"

    # 14. URL
    if "url" not in df.columns:
        df["url"] = ""

    # 15. Generate stable job IDs
    df["job_id"] = df.apply(
        lambda r: make_job_id(r["title"], r["company"], r["description"]), axis=1
    )

    # 16. Build clean text for embedding
    df["job_text_clean"] = df.apply(_build_job_text, axis=1)

    # 17. Recency score (0–1, 1 = today)
    df["recency_score"] = df["date_posted"].apply(_compute_recency)

    # 18. Embeddable flag — True only if description has ≥50 chars of meaningful text.
    #     Records with embeddable=False are kept in the download dataset but
    #     excluded from FAISS index building and embedding scoring.
    df["embeddable"] = df["description"].str.strip().str.len() >= 50

    # 19. Drop duplicates by job_id (hash of title+company+description)
    before = len(df)
    df = df.drop_duplicates(subset=["job_id"])
    logger.info(f"Exact-dedup by job_id: removed {before - len(df):,} rows")
    logger.info(f"Embeddable records: {df['embeddable'].sum():,} / {len(df):,}")

    # 20. Select and order final columns
    final_cols = [
        "job_id", "title", "company", "location", "city", "country",
        "remote", "salary_min", "salary_max", "salary_midpoint",
        "description", "skills_extracted", "seniority", "employment_type",
        "experience_required", "visa_possible", "date_posted", "recency_score",
        "source", "url", "job_text_clean", "embeddable",
    ]
    # Only keep columns that exist
    df = df[[c for c in final_cols if c in df.columns]]

    logger.info(f"Cleaning complete: {len(df):,} clean records")

    if save:
        df.to_parquet(CLEAN_PARQUET, index=False)
        # Also save a 5000-row sample for fast startup
        sample = df.sample(min(5000, len(df)), random_state=42)
        sample.to_parquet(SAMPLE_PARQUET, index=False)
        logger.info(f"Saved clean data → {CLEAN_PARQUET}")
        logger.info(f"Saved sample → {SAMPLE_PARQUET}")

    return df


# ─── Column normalization ─────────────────────────────────────────────────────
def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map various raw column names to our standard schema."""
    col_map = {
        "job_title":        "title",
        "jobtitle":         "title",
        "position":         "title",
        "role":             "title",
        "employer":         "company",
        "company_name":     "company",
        "organization":     "company",
        "employer_name":    "company",
        "job_location":     "location",
        "city_state":       "location",
        "job_description":  "description",
        "body":             "description",
        "content":          "description",
        "full_description": "description",
        "apply_url":        "url",
        "link":             "url",
        "job_url":          "url",
        "apply_link":       "url",
        "salary":           "salary_raw",
        "salary_range":     "salary_raw",
        "compensation":     "salary_raw",
        "date":             "date_posted",
        "posted_date":      "date_posted",
        "published":        "date_posted",
        "posted_at":        "date_posted",
        "job_type":         "employment_type",
        "contract_type":    "employment_type",
        "employment":       "employment_type",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Ensure required columns exist
    for col in ["title", "company", "location", "description", "url", "source"]:
        if col not in df.columns:
            df[col] = "" if col != "source" else "unknown"

    return df


def _normalize_title(title: str) -> str:
    """Clean and standardize job title."""
    if not isinstance(title, str):
        return "Unknown Position"
    title = clean_text(title)
    # Remove parenthetical noise like "(f/m/x)" or "(Remote)"
    title = re.sub(r"\([^)]*\)", "", title).strip()
    # Title case but preserve acronyms
    words = title.split()
    normalized = []
    for w in words:
        if w.upper() in {"ML", "AI", "NLP", "SQL", "BI", "IT", "UI", "UX",
                         "API", "AWS", "GCP", "ETL", "DBA", "QA", "SRE"}:
            normalized.append(w.upper())
        else:
            normalized.append(w.capitalize())
    return " ".join(normalized)


def _parse_location(location: str) -> tuple[str, str]:
    """Extract city and country from location string."""
    if not location or not isinstance(location, str):
        return "", "US"

    location = location.strip()

    # Remote detection
    if any(kw in location.lower() for kw in ["remote", "work from home", "wfh"]):
        return "Remote", "US"

    # Try "City, State" or "City, Country"
    parts = [p.strip() for p in location.split(",")]
    if len(parts) >= 2:
        city = parts[0]
        country_part = parts[-1].strip().upper()
        # US state codes
        us_states = {"CA", "NY", "WA", "TX", "MA", "IL", "CO", "GA", "FL",
                     "VA", "NC", "OH", "MI", "MN", "AZ", "NV", "UT"}
        if country_part in us_states or len(country_part) == 2:
            country = "US" if country_part in us_states else country_part
        else:
            country = "US"
        return city, country

    return location, "US"


def _normalize_employment_type(et: str) -> str:
    """Standardize employment type string."""
    if not isinstance(et, str):
        return "Full-time"
    et = et.lower()
    if any(w in et for w in ["contract", "contractor", "temp", "temporary", "freelance"]):
        return "Contract"
    if any(w in et for w in ["part", "parttime"]):
        return "Part-time"
    if "intern" in et:
        return "Internship"
    return "Full-time"


def _normalize_date(date_val) -> str:
    """Normalize various date formats to YYYY-MM-DD."""
    if pd.isna(date_val) or date_val == "":
        return datetime.now().strftime("%Y-%m-%d")
    try:
        return pd.to_datetime(str(date_val), errors="coerce").strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")


def _build_job_text(row: pd.Series) -> str:
    """Build the canonical text representation of a job for embedding."""
    skills_str = ", ".join(row.get("skills_extracted", []) or [])
    return (
        f"Job Title: {row.get('title', '')}. "
        f"Company: {row.get('company', '')}. "
        f"Location: {row.get('location', '')}. "
        f"Required Skills: {skills_str}. "
        f"Description: {str(row.get('description', ''))[:600]}"
    ).strip()


def _compute_recency(date_str: str) -> float:
    """Convert date to 0–1 recency score (1 = today, 0 = 90+ days ago)."""
    try:
        posted = pd.to_datetime(date_str)
        days_ago = (datetime.now() - posted.to_pydatetime()).days
        return max(0.0, 1.0 - days_ago / 90.0)
    except Exception:
        return 0.5


# ─── Quick load utility ───────────────────────────────────────────────────────
def load_clean_data(sample: bool = False) -> pd.DataFrame:
    """Load clean parquet. If not found, run cleaning pipeline."""
    from src.utils import CLEAN_PARQUET, SAMPLE_PARQUET
    from src.ingest import load_offline_data

    target = SAMPLE_PARQUET if sample else CLEAN_PARQUET
    if target.exists():
        df = pd.read_parquet(target)
        # Ensure skills_extracted is list type (can be stored as str in parquet)
        if "skills_extracted" in df.columns:
            df["skills_extracted"] = df["skills_extracted"].apply(
                lambda x: x if isinstance(x, list) else (
                    eval(x) if isinstance(x, str) and x.startswith("[") else []
                )
            )
        return df

    logger.info("Clean data not found. Running full pipeline...")
    raw = load_offline_data()
    return clean_jobs(raw, save=True)
