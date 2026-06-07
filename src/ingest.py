"""
ingest.py — Data ingestion from Kaggle (offline) and JSearch API (live).

Usage:
    from src.ingest import load_offline_data, fetch_jsearch_jobs, save_raw_data
"""

import os
import json
import time
import random
import logging
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

if __package__ is None or __package__ == "":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (
    DATA_DIR, RAW_CSV, SAMPLE_PARQUET, JSEARCH_API_KEY,
    clean_text, make_job_id, logger
)

# ─── Schema ───────────────────────────────────────────────────────────────────
REQUIRED_COLS = [
    "job_id", "title", "company", "location", "city", "country",
    "remote", "salary_min", "salary_max", "salary_midpoint",
    "description", "skills_extracted", "seniority", "employment_type",
    "experience_required", "visa_possible", "date_posted", "source", "url",
]


# ─── Kaggle ingestion ─────────────────────────────────────────────────────────
def load_kaggle_data() -> pd.DataFrame:
    """
    Download and load the TechMap international job postings dataset via kagglehub.
    Falls back to sample data if Kaggle credentials are not set.
    """
    try:
        import kagglehub
        logger.info("Downloading Kaggle dataset via kagglehub...")
        path = kagglehub.dataset_download("techmap/international-job-postings-september-2021")
        logger.info(f"Dataset downloaded to: {path}")

        # Find all CSV files in the downloaded path
        csv_files = list(Path(path).rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError("No CSV files found in Kaggle download.")

        logger.info(f"Found {len(csv_files)} CSV file(s)")

        dfs = []
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path, low_memory=False, on_bad_lines="skip")
                dfs.append(df)
                logger.info(f"  Loaded {len(df):,} rows from {csv_path.name}")
            except Exception as e:
                logger.warning(f"  Skipped {csv_path.name}: {e}")

        if not dfs:
            raise ValueError("No readable CSVs found.")

        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Total rows loaded from Kaggle: {len(combined):,}")
        return combined

    except Exception as e:
        logger.warning(f"Kaggle load failed ({e}). Generating synthetic dataset.")
        return _generate_synthetic_jobs(n=5000)


def _map_kaggle_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Kaggle dataset columns to the JobPilot schema.
    Handles different column naming conventions in the TechMap dataset.
    """
    col_map = {
        # Possible Kaggle column names → our schema
        "job_title":       "title",
        "jobtitle":        "title",
        "position":        "title",
        "employer":        "company",
        "company_name":    "company",
        "organization":    "company",
        "job_location":    "location",
        "city_state":      "location",
        "job_description": "description",
        "body":            "description",
        "content":         "description",
        "apply_url":       "url",
        "link":            "url",
        "job_url":         "url",
        "salary":          "salary_raw",
        "salary_range":    "salary_raw",
        "date":            "date_posted",
        "posted_date":     "date_posted",
        "published":       "date_posted",
        "job_type":        "employment_type",
        "contract_type":   "employment_type",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    return df


# ─── JSearch / RapidAPI live ingestion ───────────────────────────────────────
#
# API Details:
#   Host:    jsearch.p.rapidapi.com
#   Method:  GET /search
#   Key:     set JSEARCH_API_KEY in .env  (same RapidAPI key used for all services)
#
# Equivalent curl:
#   curl --request GET \
#        --url 'https://jsearch.p.rapidapi.com/search?query=data+scientist&page=1&num_pages=1' \
#        --header 'x-rapidapi-host: jsearch.p.rapidapi.com' \
#        --header 'x-rapidapi-key: YOUR_KEY'
# ─────────────────────────────────────────────────────────────────────────────

JSEARCH_BASE_URL = "https://jsearch-mega.p.rapidapi.com"

def _jsearch_headers() -> dict:
    """Return the RapidAPI headers required for every JSearch Mega request."""
    return {
        "x-rapidapi-key":  JSEARCH_API_KEY,
        "x-rapidapi-host": "jsearch-mega.p.rapidapi.com",
        "Content-Type":    "application/json",
    }


def test_jsearch_connection() -> bool:
    """
    Verify the JSearch API key is valid by fetching a single result.
    Returns True if the connection succeeds, False otherwise.
    """
    if not JSEARCH_API_KEY:
        logger.warning("JSEARCH_API_KEY not configured.")
        return False
    try:
        resp = requests.get(
            f"{JSEARCH_BASE_URL}/search",
            headers=_jsearch_headers(),
            params={"query": "data analyst", "page": "1", "num_pages": "1"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        ok = "data" in data and isinstance(data["data"], list)
        logger.info(f"JSearch connection {'✅ OK' if ok else '❌ unexpected response'}")
        return ok
    except Exception as e:
        logger.error(f"JSearch connection failed: {e}")
        return False


def fetch_jsearch_jobs(
    query: str = "data scientist",
    num_pages: int = 3,
    country: str = "us",
    date_posted: str = "week",
    remote_only: bool = False,
) -> pd.DataFrame:
    """
    Fetch live job postings from JSearch API (RapidAPI / Google Jobs).

    Args:
        query:       Search query, e.g. "ML Engineer" or "data analyst remote"
        num_pages:   Number of result pages to fetch (10 jobs per page)
        country:     ISO country code, e.g. "us", "gb", "ca"
        date_posted: Freshness filter — "all" | "today" | "3days" | "week" | "month"
        remote_only: If True, adds "remote" to the query string

    Returns:
        DataFrame in JobPilot schema.
    """
    if not JSEARCH_API_KEY:
        logger.warning("JSEARCH_API_KEY not set — skipping live ingestion.")
        return pd.DataFrame()

    if remote_only:
        query = f"{query} remote"

    records = []
    for page in range(1, num_pages + 1):
        params = {
            "query":       query,
            "page":        str(page),
            "num_pages":   "1",
            "country":     country,
            "date_posted": date_posted,
        }
        try:
            resp = requests.get(
                f"{JSEARCH_BASE_URL}/search",
                headers=_jsearch_headers(),
                params=params,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            # Handle API error responses
            if data.get("status") == "ERROR":
                logger.error(f"JSearch API error: {data.get('error', {}).get('message', 'Unknown')}")
                break

            jobs = data.get("data", [])
            logger.info(f"JSearch '{query}' page {page}: {len(jobs)} jobs")

            for j in jobs:
                # Salary — JSearch returns annual or null
                sal_min = j.get("job_min_salary") or 0
                sal_max = j.get("job_max_salary") or 0

                # Location string
                city    = j.get("job_city", "") or ""
                state   = j.get("job_state", "") or ""
                country_code = j.get("job_country", "US") or "US"
                location = ", ".join(filter(None, [city, state, country_code]))

                records.append({
                    "title":            j.get("job_title", ""),
                    "company":          j.get("employer_name", ""),
                    "location":         location,
                    "city":             city,
                    "country":          country_code,
                    "remote":           bool(j.get("job_is_remote", False)),
                    "description":      j.get("job_description", ""),
                    "employment_type":  _normalize_jsearch_type(
                                            j.get("job_employment_type", "")
                                        ),
                    "date_posted":      j.get("job_posted_at_datetime_utc", ""),
                    "url":              j.get("job_apply_link", "")
                                        or j.get("job_google_link", ""),
                    "salary_min":       float(sal_min),
                    "salary_max":       float(sal_max),
                    "source":           "jsearch",
                })

            time.sleep(0.4)   # stay within free-tier rate limits

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else "?"
            if status == 429:
                logger.warning("JSearch rate limit hit — waiting 5s...")
                time.sleep(5)
                continue
            logger.error(f"JSearch HTTP {status} on page {page}: {e}")
            break
        except Exception as e:
            logger.error(f"JSearch page {page} failed: {e}")
            break

    if not records:
        logger.warning("JSearch returned no results.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    logger.info(f"JSearch total: {len(df):,} live jobs fetched")
    return df


def _normalize_jsearch_type(et: str) -> str:
    """Map JSearch employment_type strings to our schema."""
    mapping = {
        "FULLTIME":   "Full-time",
        "PARTTIME":   "Part-time",
        "CONTRACTOR": "Contract",
        "INTERN":     "Internship",
        "TEMPORARY":  "Contract",
    }
    return mapping.get((et or "").upper(), "Full-time")


def fetch_multiple_queries(
    queries: list[str],
    pages_per_query: int = 2,
    date_posted: str = "week",
    country: str = "us",
) -> pd.DataFrame:
    """
    Fetch jobs for multiple role queries and combine into one DataFrame.
    Automatically deduplicates by URL before returning.

    Args:
        country: ISO country code passed to JSearch (default "us" for USA-only results)
    """
    frames = []
    for q in queries:
        logger.info(f"JSearch fetching: '{q}' (country={country})")
        df = fetch_jsearch_jobs(query=q, num_pages=pages_per_query,
                                date_posted=date_posted, country=country)
        if not df.empty:
            frames.append(df)
        time.sleep(0.3)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Remove duplicates by URL (same job listed under multiple queries)
    if "url" in combined.columns:
        before = len(combined)
        combined = combined[combined["url"] != ""]
        combined = combined.drop_duplicates(subset=["url"])
        logger.info(f"JSearch dedup by URL: {before} → {len(combined)} records")

    return combined


# ─── Synthetic data (fallback) ────────────────────────────────────────────────
def _generate_synthetic_jobs(n: int = 5000) -> pd.DataFrame:
    """
    Generate a realistic synthetic job dataset for testing when Kaggle data
    is unavailable. Covers the four persona requirements.
    """
    logger.info(f"Generating {n:,} synthetic job records...")
    random.seed(42)
    np.random.seed(42)

    titles = [
        "Data Scientist", "Senior Data Scientist", "ML Engineer",
        "Senior ML Engineer", "Applied Scientist", "Data Analyst",
        "Senior Data Analyst", "BI Analyst", "Junior Data Analyst",
        "Analytics Engineer", "Data Engineer", "Senior Data Engineer",
        "MLOps Engineer", "ML Platform Engineer", "Research Scientist",
        "AI Engineer", "NLP Engineer", "Computer Vision Engineer",
        "Product Analyst", "Business Intelligence Developer",
        "Staff ML Engineer", "Principal Data Scientist",
    ]
    companies = [
        "Google", "Microsoft", "Amazon", "Meta", "Apple", "Netflix", "Uber",
        "Airbnb", "Stripe", "Salesforce", "IBM", "Oracle", "Intel", "NVIDIA",
        "Linkedin", "Twitter", "Shopify", "Databricks", "Snowflake", "Palantir",
        "Epic Systems", "Kaiser Permanente", "CVS Health", "UnitedHealth",
        "JPMorgan Chase", "Goldman Sachs", "BlackRock", "Citadel",
        "General Dynamics", "Lockheed Martin", "Raytheon",  # defense (for filter testing)
        "TechStartup Inc", "SmallBiz Analytics", "DataCo",
    ]
    locations = [
        "San Francisco, CA", "New York, NY", "Seattle, WA", "Austin, TX",
        "Boston, MA", "Chicago, IL", "Remote", "Remote - US",
        "Mountain View, CA", "Menlo Park, CA", "Redmond, WA",
        "London, UK", "Toronto, Canada", "Berlin, Germany",
    ]
    skills_pool = [
        "Python", "SQL", "machine learning", "deep learning", "PyTorch",
        "TensorFlow", "Spark", "Kafka", "Kubernetes", "Docker", "AWS",
        "scikit-learn", "NLP", "computer vision", "pandas", "R",
        "Tableau", "Power BI", "dbt", "Snowflake", "Databricks",
        "MLflow", "Airflow", "Java", "Scala", "Go",
    ]

    records = []
    for i in range(n):
        title = random.choice(titles)
        company = random.choice(companies)
        location = random.choice(locations)
        is_remote = "Remote" in location or random.random() < 0.35
        num_skills = random.randint(3, 10)
        req_skills = random.sample(skills_pool, num_skills)

        seniority = "senior" if any(s in title for s in ["Senior", "Staff", "Principal", "Lead"]) \
                    else "junior" if any(s in title for s in ["Junior", "Associate"]) \
                    else "mid"

        sal_base = {"junior": 80000, "mid": 120000, "senior": 180000}[seniority]
        sal_min = sal_base + random.randint(-15000, 0)
        sal_max = sal_base + random.randint(10000, 50000)

        is_defense = any(d in company for d in ["Dynamics", "Lockheed", "Raytheon"])
        is_contract = random.random() < 0.1
        exp_req = {"junior": random.randint(0, 2), "mid": random.randint(2, 5),
                   "senior": random.randint(5, 10)}[seniority]

        description = (
            f"{company} is looking for a {title} to join our team. "
            f"Required skills: {', '.join(req_skills)}. "
            f"{exp_req}+ years of experience required. "
            f"{'This is a remote position.' if is_remote else f'Based in {location}.'} "
            f"{'Contract only position.' if is_contract else 'Full-time permanent role.'} "
            f"Salary: ${sal_min:,} - ${sal_max:,}. "
            f"{'H-1B visa sponsorship available for qualified candidates.' if not is_defense and random.random() < 0.5 else ''}"
        )

        days_ago = random.randint(0, 90)
        date_posted = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")

        records.append({
            "title":            title,
            "company":          company,
            "location":         location,
            "description":      description,
            "skills_raw":       ", ".join(req_skills),
            "salary_raw":       f"${sal_min:,} - ${sal_max:,}",
            "employment_type":  "Contract" if is_contract else "Full-time",
            "date_posted":      date_posted,
            "url":              f"https://example.com/jobs/{i+1}",
            "source":           "synthetic",
        })

    df = pd.DataFrame(records)
    logger.info(f"Synthetic dataset: {len(df):,} records")
    return df


# broaden_jsearch_to_min and minimum-corpus logic removed — pre-loaded mode
# uses the Kaggle parquet as both training corpus and match pool.


# ─── Save helpers ─────────────────────────────────────────────────────────────
def save_raw_data(df: pd.DataFrame, path=None) -> Path:
    """Save raw ingested data as CSV."""
    path = Path(path) if path else RAW_CSV
    df.to_csv(path, index=False)
    logger.info(f"Saved {len(df):,} raw records to {path}")
    return path


def load_offline_data(sample: bool = False) -> pd.DataFrame:
    """
    Load offline data. Tries clean parquet first, then raw CSV,
    then falls back to synthetic data.
    """
    from src.utils import CLEAN_PARQUET

    target = SAMPLE_PARQUET if sample else CLEAN_PARQUET
    if target.exists():
        df = pd.read_parquet(target)
        logger.info(f"Loaded {len(df):,} jobs from {target.name}")
        return df

    if RAW_CSV.exists():
        logger.info(f"Loading raw CSV: {RAW_CSV}")
        return pd.read_csv(RAW_CSV, low_memory=False)

    logger.warning("No offline data found. Generating synthetic dataset.")
    return _generate_synthetic_jobs(5000)
