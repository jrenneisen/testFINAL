from __future__ import annotations
"""
JobPilot — Smart Job Matcher & Resume Builder
BAX-423 Big Data | Spring 2026 | Final Project Option B

Run with: streamlit run app.py
"""

import sys
import json
import time
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from typing import Optional

# ─── Path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import (
    DATA_DIR, PERSONAS_FILE, TOP_K_JOBS, RETRIEVAL_K,
    OPENAI_API_KEY, JSEARCH_API_KEY,
    VECTOR_INDEX_ZIP, JOB_META_PARQUET, CLUSTERS_FILE,
    logger
)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — SERVER-LEVEL CACHE  (@st.cache_resource)
#
# These four functions are called ONCE per server process (not per session).
# Streamlit pins the return value in RAM and returns it instantly to every
# subsequent caller — whether that's a new user session or a page rerun.
#
# Load order at startup:
#   1. _get_cached_model()   — sentence-transformer (~20s cold, instant warm)
#   2. _get_cached_index()   — FAISS index + embeddings + job_ids (~3-5s cold)
#   3. _get_cached_clusters()— K-Means cluster labels (~1s cold)
#   4. _get_cached_tfidf()   — TF-IDF matrix over corpus (~2s cold)
#
# By the time a user fills in their profile and clicks Run Pipeline, all four
# are already loaded.  Phase 2 (scoring + ranking) then takes < 2s.
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="🤖 Loading AI model (first-time setup, ~20s)…")
def _get_cached_model():
    """
    Load sentence-transformers model once and cache it across ALL user sessions
    on the same Streamlit worker.  Without this, every new session re-loads the
    model from disk, adding ~20s to every user's first pipeline run.
    """
    from src.embeddings import get_model
    return get_model()


@st.cache_resource(show_spinner="⚡ Loading pre-built FAISS index…")
def _get_cached_index():
    """
    Phase 1 — load the pre-built Kaggle FAISS index from disk once per server
    process.  Returns (index, embeddings, job_ids).

    If the index files are missing (first deploy / local dev without pre-built
    files), returns (None, None, None) and the pipeline falls through to the
    slow-path rebuild.
    """
    from src.embeddings import load_prebuilt_index, prebuilt_index_exists
    if not prebuilt_index_exists():
        logger.warning(
            "Pre-built index files not found — pipeline will use slow-path on first run. "
            "Run: python scripts/build_preloaded_data.py"
        )
        return None, None, None
    try:
        index, embeddings, job_ids = load_prebuilt_index()
        logger.info(f"Cached FAISS index loaded: {index.ntotal:,} vectors")
        return index, embeddings, job_ids
    except Exception as exc:
        logger.error(f"Failed to load pre-built index: {exc}")
        return None, None, None


@st.cache_resource(show_spinner="🧠 Loading job-family clusters…")
def _get_cached_clusters():
    """
    Phase 1 — load K-Means cluster assignments from disk once per server process.
    Returns (cluster_labels, job_ids) or (None, None) if not yet built.

    The pipeline will build clusters on first run if this returns None.
    """
    from src.embeddings import CLUSTERS_FILE
    import numpy as np
    if not CLUSTERS_FILE.exists():
        return None, None
    try:
        data = np.load(CLUSTERS_FILE, allow_pickle=True)
        labels   = data["labels"]
        job_ids  = data["job_ids"].tolist()
        logger.info(f"Cached cluster labels loaded: {len(job_ids):,} jobs, {len(set(labels))} clusters")
        return labels, job_ids
    except Exception as exc:
        logger.error(f"Failed to load cluster file: {exc}")
        return None, None


@st.cache_resource(show_spinner="📋 Building TF-IDF index…")
def _get_cached_tfidf(corpus_hash: str = ""):
    """
    Phase 1 — fit a TF-IDF vectorizer over job_meta.parquet and cache it.
    `corpus_hash` is a fingerprint (file size) so the cache invalidates
    if the parquet is replaced.

    Returns (vectorizer, tfidf_matrix, job_ids) or (None, None, None) on error.
    """
    from src.embeddings import _build_tfidf  # internal helper
    import pandas as pd
    if not JOB_META_PARQUET.exists():
        logger.warning("job_meta.parquet not found — TF-IDF disabled (run build_preloaded_data.py)")
        return None, None, None
    try:
        df     = pd.read_parquet(JOB_META_PARQUET)
        result = _build_tfidf(df)
        logger.info(f"Cached TF-IDF built: {len(df):,} documents from job_meta.parquet")
        return result
    except Exception as exc:
        logger.error(f"Failed to build TF-IDF cache: {exc}")
        return None, None, None


# ─── Persona pre-loading (Phase 1) ────────────────────────────────────────────
# Built-in persona profiles — pre-matched at server startup so results are
# served instantly when a user selects a persona from the dropdown.
_BUILTIN_PERSONAS = {
    "aisha": {
        "id": "aisha",
        "name": "Aisha",
        "current_title": "Data Analyst",
        "years_experience": 3,
        "education": "MS Data Science",
        "skills": ["Python", "SQL", "Tableau", "Machine Learning", "pandas", "sklearn"],
        "target_roles": ["Data Scientist", "ML Engineer", "Analytics Engineer"],
        "industries": ["Technology", "Finance"],
        "seniority_target": "mid",
        "location": "San Francisco, CA",
        "remote": True,
        "salary_min": 120000,
        "salary_max": 160000,
        "visa_needed": False,
    },
    "marcus": {
        "id": "marcus",
        "name": "Marcus",
        "current_title": "Software Engineer",
        "years_experience": 6,
        "education": "BS Computer Science",
        "skills": ["Python", "Java", "AWS", "Docker", "Kubernetes", "SQL", "Spark"],
        "target_roles": ["Senior Software Engineer", "ML Platform Engineer", "Data Engineer"],
        "industries": ["Technology", "Cloud"],
        "seniority_target": "senior",
        "location": "Remote",
        "remote": True,
        "salary_min": 150000,
        "salary_max": 200000,
        "visa_needed": False,
    },
    "priya": {
        "id": "priya",
        "name": "Priya",
        "current_title": "Business Intelligence Analyst",
        "years_experience": 4,
        "education": "MBA",
        "skills": ["SQL", "Power BI", "Tableau", "Excel", "Python", "A/B Testing"],
        "target_roles": ["Senior BI Analyst", "Data Analyst", "Product Analyst"],
        "industries": ["Retail", "E-Commerce", "Consulting"],
        "seniority_target": "mid",
        "location": "New York, NY",
        "remote": False,
        "salary_min": 100000,
        "salary_max": 135000,
        "visa_needed": True,
    },
    "kenji": {
        "id": "kenji",
        "name": "Kenji",
        "current_title": "Machine Learning Research Intern",
        "years_experience": 1,
        "education": "MS Computer Science",
        "skills": ["Python", "PyTorch", "TensorFlow", "Deep Learning", "NLP", "Computer Vision", "SQL"],
        "target_roles": ["ML Engineer", "Research Scientist", "AI Engineer"],
        "industries": ["Technology", "Research", "Healthcare AI"],
        "seniority_target": "junior",
        "location": "Seattle, WA",
        "remote": True,
        "salary_min": 110000,
        "salary_max": 145000,
        "visa_needed": True,
    },
}


@st.cache_resource(show_spinner="🎭 Pre-loading persona matches…")
def _get_cached_persona_results():
    """
    Phase 1 — run the full scoring pipeline for all 4 built-in personas at
    server startup and cache the results.  Selecting a persona from the dropdown
    then serves results instantly without re-running FAISS + RRF + rank.

    Uses the pre-built FAISS index for dense retrieval and the cached TF-IDF
    vectorizer for sparse retrieval, then merges via Reciprocal Rank Fusion.

    Returns a dict: {persona_id: list[RankedJob]} or {} on error.
    """
    from src.embeddings import (
        embed_single, build_profile_text, load_prebuilt_index,
        prebuilt_index_exists, load_jsearch_embeddings,
        tfidf_retrieve, reciprocal_rank_fusion,
    )
    from src.ranker import rank_jobs
    import numpy as np
    import pandas as pd

    if not prebuilt_index_exists() or not JOB_META_PARQUET.exists():
        logger.info("Persona pre-load skipped: index or metadata not available")
        return {}

    results: dict = {}

    try:
        # Load base index
        index, _embeddings, job_ids = load_prebuilt_index()
        all_ids = list(job_ids)

        # Merge live JSearch vectors into a per-session index copy
        js_vecs, js_ids = load_jsearch_embeddings()
        if js_vecs is not None and len(js_ids) > 0:
            import faiss as _faiss
            index = _faiss.deserialize_index(_faiss.serialize_index(index))
            index.add(js_vecs)
            all_ids = all_ids + list(js_ids)

        # Load job metadata (for the ranker DataFrame arg)
        meta_df = pd.read_parquet(JOB_META_PARQUET)

        # Get pre-fitted TF-IDF objects (already warm from _get_cached_tfidf)
        tfidf_cache = _get_cached_tfidf(_CORPUS_HASH)
        _vec, _mat, _cids = tfidf_cache if tfidf_cache[0] is not None else (None, None, None)

        for pid, persona in _BUILTIN_PERSONAS.items():
            try:
                # ── Dense (FAISS) candidates ──────────────────────────────────
                profile_text = build_profile_text(persona)
                q_emb = embed_single(profile_text).reshape(1, -1).astype("float32")
                k_ret = min(RETRIEVAL_K, index.ntotal)
                scores, idxs = index.search(q_emb, k_ret)
                dense_candidates: list[tuple[str, float]] = [
                    (all_ids[int(i)], float(scores[0][j]))
                    for j, i in enumerate(idxs[0])
                    if i >= 0 and int(i) < len(all_ids)
                ]

                # ── Sparse (TF-IDF) candidates ────────────────────────────────
                tfidf_candidates = tfidf_retrieve(
                    persona, meta_df,
                    k=min(RETRIEVAL_K, len(meta_df)),
                    _vectorizer=_vec,
                    _tfidf_matrix=_mat,
                    _cached_job_ids=_cids,
                )

                # ── RRF merge ─────────────────────────────────────────────────
                try:
                    merged = reciprocal_rank_fusion(dense_candidates, tfidf_candidates)
                except Exception:
                    merged = dense_candidates

                # ── Rank ──────────────────────────────────────────────────────
                ranked = rank_jobs(
                    meta_df, persona, merged,
                    weights=None,
                    feedback={},
                    top_n=TOP_K_JOBS,
                )
                results[pid] = ranked
                logger.info(f"Persona pre-load '{pid}': {len(ranked)} matches cached")

            except Exception as e:
                logger.warning(f"Persona pre-load failed for '{pid}': {e}")

    except Exception as exc:
        logger.error(f"Persona pre-load failed entirely: {exc}")

    return results

# ─── Init database on startup ─────────────────────────────────────────────────
from src.storage import (
    init_db, create_or_update_user, get_user, list_users, delete_user,
    save_profile, load_profile,
    save_feedback_event, load_feedback_history, get_feedback_summary,
    save_bandit_state, load_bandit_state, replay_feedback_into_bandit,
    save_ranking_weights, load_ranking_weights,
    save_resume, load_resumes,
    save_job_list, load_job_list,
    get_learning_insights, db_size_kb,
    save_jsearch_jobs, load_jsearch_jobs, count_jsearch_jobs, get_jsearch_job_ids,
)
init_db()

# ── Module-level corpus fingerprint — used as the TF-IDF cache key ────────────
# Derived from job_meta.parquet file size so the cache auto-invalidates
# whenever the build script produces a new parquet.
_CORPUS_HASH = (
    str(JOB_META_PARQUET.stat().st_size)
    if JOB_META_PARQUET.exists()
    else ""
)

# ── Phase 1 warm-up — runs once on cold start, instant on every session after ──
# All calls are no-ops after the first server boot (cache_resource hit).
# On cold start they run sequentially in ~25-35s total:
#   model (~20s) → index (~5s) → clusters (~1s) → TF-IDF (~2s) → personas (~5s)
# By the time a user fills in their profile all heavy lifting is already done.
_get_cached_model()
_get_cached_index()
_get_cached_clusters()
_get_cached_tfidf(_CORPUS_HASH)
_get_cached_persona_results()  # pre-compute matches for all 4 built-in personas

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JobPilot — Smart Job Matcher",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global */
:root {
  --brand-blue: #1F4E79;
  --accent-blue: #2E75B6;
  --light-blue: #D6E4F0;
  --success: #27AE60;
  --warning: #F39C12;
  --danger: #E74C3C;
}

/* Sidebar */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #1F4E79 0%, #2E4057 100%);
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stRadio label { color: white !important; font-size: 0.95rem; }

/* Job cards */
.job-card {
  background: white;
  border: 1px solid #E0ECF8;
  border-left: 5px solid #2E75B6;
  border-radius: 10px;
  padding: 18px 22px;
  margin-bottom: 16px;
  box-shadow: 0 2px 8px rgba(30,78,121,0.07);
  transition: box-shadow 0.2s;
}
.job-card:hover { box-shadow: 0 4px 16px rgba(30,78,121,0.14); }

/* Score badge */
.score-badge {
  display: inline-block;
  padding: 4px 12px;
  border-radius: 20px;
  font-weight: 700;
  font-size: 0.88rem;
  color: white;
  margin-right: 6px;
}
.score-high   { background: #27AE60; }
.score-mid    { background: #F39C12; }
.score-low    { background: #E74C3C; }

/* Skill pills */
.skill-pill {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 12px;
  font-size: 0.78rem;
  margin: 2px;
  font-weight: 500;
}
.skill-matched { background: #D5F5E3; color: #1E8449; border: 1px solid #82E0AA; }
.skill-missing { background: #FDEBD0; color: #A04000; border: 1px solid #F0B27A; }

/* Section headers */
.section-header {
  font-size: 1.5rem;
  font-weight: 700;
  color: #1F4E79;
  border-bottom: 3px solid #2E75B6;
  padding-bottom: 8px;
  margin-bottom: 20px;
}

/* Metric cards */
.metric-card {
  background: #F0F6FC;
  border-radius: 8px;
  padding: 16px;
  text-align: center;
  border: 1px solid #D6E4F0;
}
.metric-number { font-size: 2rem; font-weight: 800; color: #1F4E79; }
.metric-label  { font-size: 0.85rem; color: #5D6D7E; }

/* Hero banner */
.hero {
  background: linear-gradient(135deg, #1F4E79 0%, #2E75B6 100%);
  color: white;
  padding: 32px 40px;
  border-radius: 14px;
  margin-bottom: 28px;
}
.hero h1 { color: white; font-size: 2.4rem; margin-bottom: 6px; }
.hero p  { color: #BDD7EE; font-size: 1.05rem; margin: 0; }

/* Feedback buttons */
.stButton > button {
  border-radius: 8px;
  font-weight: 600;
  transition: all 0.15s;
}

/* Resume output */
.resume-output {
  background: white;
  border: 1px solid #D6E4F0;
  border-radius: 10px;
  padding: 28px;
  font-family: Georgia, serif;
  line-height: 1.6;
  max-height: 600px;
  overflow-y: auto;
}

/* Hide Streamlit footer */
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Session state initialization ─────────────────────────────────────────────
def init_session():
    defaults = {
        "page":             "🏠 Profile Setup",
        "current_user":     None,    # logged-in user_id string
        "profile":          None,
        "jobs_df":          None,    # training corpus (Kaggle) — analytics/benchmarks only
        "live_jobs_df":     None,    # live JSearch jobs — what the user actually sees
        "faiss_index":      None,
        "job_ids":          None,
        "ranked_jobs":      [],
        "feedback":         {},
        "adaptive":         None,
        "resumes":          {},
        "selected_job":     None,
        "pipeline_ready":   False,
        "analytics":        None,
        "benchmark_data":   {},
        "data_stats":       {},
        "tfidf_candidates": [],
        "emb_candidates":   [],
        "hybrid_candidates":[],
        "cluster_labels":   None,   # K-Means job family clusters (ndarray, training corpus)
        "live_cluster_map": {},     # {job_id: cluster_id} for live jobs (for cluster boost)
        "positive_ids":     set(),
        "auto_profile":          {},     # extracted resume fields → pre-fills Custom Profile form
        "resume_text_cache":     "",    # raw resume text (survives rerun after PDF parse)
        "preloaded_kaggle_df":        None,  # training corpus parquet (from disk or upload)
        "preloaded_jsearch_df":       None,  # match pool parquet (from disk or upload)
        "preloaded_autoloaded":       False, # True once disk auto-load has run this session
        # jobs_df = deduped corpus used for FAISS + scoring
        "retrieval_mode":             "hybrid",   # "hybrid" or "dense"
        "profile_emb":                None,       # (384,) float32 — profile embedding vector
        "job_embs":                   None,       # (n, 384) float32 — job embedding matrix
        "pre_feedback_top10":         [],         # snapshot of top-10 BEFORE adaptive feedback
        # ── Per-profile result cache ───────────────────────────────────────────
        # Maps cache_key → full pipeline result dict so switching profiles
        # doesn't require re-running the pipeline.
        "profile_results":            {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ─── Per-profile result cache helpers ─────────────────────────────────────────
# All pipeline results live in session state (in-memory). When the user switches
# profiles we save the current results into profile_results[key] before clearing,
# then attempt to restore from cache when the new profile is loaded.
#
# Cache keys:
#   • Personas  →  persona["id"]  (stable ID from personas.json)
#   • Logged-in custom profile  →  "{user_id}_custom"
#   • Anonymous custom profile  →  "anon_custom"

_CACHED_KEYS = [
    "ranked_jobs", "pipeline_ready", "benchmark_data", "analytics",
    "data_stats", "live_jobs_df", "jobs_df",
    "faiss_index", "job_ids", "cluster_labels",
    "hybrid_candidates", "tfidf_candidates", "emb_candidates",
    "live_cluster_map", "profile_emb", "job_embs",
    "pre_feedback_top10", "retrieval_mode",
]


def _profile_cache_key(profile: dict | None = None) -> str:
    """Return a stable string key for the given (or current) profile."""
    p = profile or st.session_state.profile
    if p is None:
        return "__none__"
    if "id" in p:                          # pre-built persona
        return f"persona_{p['id']}"
    uid = st.session_state.current_user
    return f"{uid}_custom" if uid else "anon_custom"


def _save_to_profile_cache(key: str | None = None):
    """Snapshot current pipeline results into profile_results[key]."""
    k = key or _profile_cache_key()
    if k == "__none__" or not st.session_state.pipeline_ready:
        return
    st.session_state.profile_results[k] = {
        ck: st.session_state.get(ck) for ck in _CACHED_KEYS
    }
    # Also store which profile produced these results (for display)
    st.session_state.profile_results[k]["_profile"] = (
        st.session_state.profile or {}
    )


def _load_from_profile_cache(profile: dict) -> bool:
    """
    Try to restore pipeline results for `profile` from the cache.
    Returns True if a cached result was found and loaded, False otherwise.
    """
    k = _profile_cache_key(profile)
    cached = st.session_state.profile_results.get(k)
    if not cached or not cached.get("pipeline_ready"):
        return False
    for ck in _CACHED_KEYS:
        if ck in cached:
            st.session_state[ck] = cached[ck]
    return True


def _switch_profile(new_profile: dict):
    """
    Save current results to cache, then switch to new_profile.
    Restores cached results for the new profile if available.
    Returns True if cache hit (no re-run needed).
    """
    # Save outgoing profile's results
    _save_to_profile_cache()
    # Set new profile
    st.session_state.profile = new_profile
    # Attempt cache restore
    if _load_from_profile_cache(new_profile):
        return True   # cache hit — pipeline_ready stays True
    # Cache miss — clear stale results
    st.session_state.pipeline_ready = False
    st.session_state.ranked_jobs    = []
    return False


# ─── Auto-detect pre-loaded parquet files from disk ───────────────────────────
def _autoload_preloaded_data():
    """
    If job_meta.parquet exists on disk (shipped with the runnable folder),
    load it into session state so the pipeline has metadata for scoring.
    Reads from the compact metadata parquet (not the 79 MB raw parquet).
    Only runs once per session.
    """
    if st.session_state.preloaded_autoloaded:
        return

    st.session_state.preloaded_autoloaded = True

    if JOB_META_PARQUET.exists() and st.session_state.preloaded_kaggle_df is None:
        try:
            st.session_state.preloaded_kaggle_df = pd.read_parquet(JOB_META_PARQUET)
            n = len(st.session_state.preloaded_kaggle_df)
            logger.info(f"Auto-loaded job_meta.parquet: {n:,} rows")
            st.toast(f"📂 {n:,} pre-loaded jobs ready — no API key needed.", icon="✅")
        except Exception as e:
            logger.warning(f"Could not auto-load job_meta.parquet: {e}")

_autoload_preloaded_data()


# ─── Login / logout helpers ───────────────────────────────────────────────────
def _login_user(user_id: str):
    """
    Load all persisted state for a user into session_state.
    Restores: profile, feedback dict, resumes, bandit arms, ranking weights.
    Replays full feedback history through the adaptive learner so the model
    continues improving exactly where it left off.
    """
    from src.adaptive_learning import AdaptiveLearner

    create_or_update_user(user_id, user_id.replace("_", " ").title())
    st.session_state.current_user = user_id

    # ── Profile ───────────────────────────────────────────────────────────────
    saved_profile = load_profile(user_id)
    if saved_profile:
        st.session_state.profile = saved_profile
        st.toast(f"✅ Welcome back! Profile loaded.")

    # ── Feedback dict (for current-session display) ────────────────────────────
    history = load_feedback_history(user_id)
    st.session_state.feedback = {
        e["job_id"]: e["feedback_type"] for e in history
    }

    # ── Positive IDs set ──────────────────────────────────────────────────────
    st.session_state.positive_ids = {
        e["job_id"] for e in history
        if e["feedback_type"] in ("good", "save")
    }

    # ── Adaptive learner — restore + replay ───────────────────────────────────
    weights  = load_ranking_weights(user_id)
    adaptive = AdaptiveLearner(initial_weights=weights)

    # Load saved arm distributions
    adaptive.bandit = load_bandit_state(user_id, adaptive.bandit)

    # If arms were empty (first replay), replay history chronologically
    if not adaptive.bandit.arms and history:
        adaptive.bandit, adaptive.updater = replay_feedback_into_bandit(
            user_id, adaptive.bandit, adaptive.updater
        )

    st.session_state.adaptive = adaptive

    # ── Cached resumes ────────────────────────────────────────────────────────
    st.session_state.resumes = load_resumes(user_id)

    logger.info(
        f"Login: {user_id} — {len(history)} feedback events, "
        f"{len(adaptive.bandit.arms)} bandit arms restored"
    )


def _save_session_to_db():
    """
    Persist current session state to database.
    Called automatically on logout and after every feedback event.
    """
    uid = st.session_state.current_user
    if not uid:
        return

    if st.session_state.profile:
        save_profile(uid, st.session_state.profile)

    if st.session_state.adaptive:
        save_bandit_state(uid, st.session_state.adaptive.bandit)
        save_ranking_weights(uid, st.session_state.adaptive.weights)


# ─── PDF extraction helper ────────────────────────────────────────────────────
def _extract_pdf_text(uploaded_file) -> str:
    """
    Extract plain text from an uploaded PDF resume.

    Process:
      1. Read raw bytes from the Streamlit UploadedFile object
      2. Open with pdfplumber (handles multi-column layouts, tables, headers)
      3. Extract text page by page and join with newlines
      4. Clean whitespace and return

    Works with: standard PDFs, Google Docs exports, Word-to-PDF, LaTeX resumes.
    May struggle with: scanned image PDFs (no embedded text layer).
    """
    import io
    import re
    try:
        import pdfplumber
    except ImportError:
        st.error("pdfplumber not installed. Run: pip install pdfplumber")
        return ""

    try:
        raw_bytes = uploaded_file.read()
        pages_text = []
        total_pages = 0

        with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
            total_pages = len(pdf.pages)
            for page in pdf.pages:
                text = page.extract_text(
                    x_tolerance=2,   # how close chars must be to join on same line
                    y_tolerance=3,   # how close lines must be to join in same block
                )
                if text:
                    pages_text.append(text)

        if not pages_text:
            # Fallback: extract individual words if extract_text() returns nothing
            # (happens with some PDF generators that don't embed a text layer linearly)
            with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
                for page in pdf.pages:
                    words = page.extract_words()
                    if words:
                        pages_text.append(" ".join(w["text"] for w in words))

        full_text = "\n\n".join(pages_text)

        # Clean common PDF extraction artifacts
        full_text = re.sub(r"\n{3,}", "\n\n", full_text)  # collapse excessive blank lines
        full_text = re.sub(r"[ \t]{2,}", " ", full_text)  # collapse multiple spaces
        full_text = full_text.strip()

        logger.info(f"PDF extracted: {total_pages} page(s), {len(full_text):,} chars")
        return full_text

    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        st.error(f"Could not read PDF: {e}. Try pasting your resume text manually below.")
        return ""


# ─── Resume → profile field extractor ────────────────────────────────────────
def _extract_profile_from_resume(resume_text: str) -> dict:
    """
    Parse a resume and return a dict of profile fields.

    Uses GPT-4o-mini when an OpenAI key is available (structured JSON response).
    Falls back to lightweight regex extraction otherwise — still catches name,
    education, and common tech skills from most English-language resumes.
    """
    # ── GPT path ─────────────────────────────────────────────────────────────
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                max_tokens=700,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You extract structured profile information from resumes. "
                            "Return ONLY a valid JSON object — no markdown, no explanation."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Extract the following fields from this resume and return as JSON:\n"
                            "- name (string): full name of the person\n"
                            "- current_title (string): most recent job title\n"
                            "- years_experience (integer): estimated total years of work experience\n"
                            "- education (string): highest degree + field, e.g. 'MS Data Science'\n"
                            "- skills (array of strings): up to 15 technical skills\n"
                            "- target_roles (array of strings): 2-3 likely target job titles\n"
                            "- industries (array of strings): industries the person has worked in\n"
                            "- seniority_target (string): one of junior/mid/senior\n\n"
                            f"Resume (first 3000 chars):\n{resume_text[:3000]}\n\n"
                            "Return ONLY the JSON object."
                        ),
                    },
                ],
            )
            raw = resp.choices[0].message.content.strip()
            # Strip markdown fences if model added them
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            extracted = json.loads(raw)
            logger.info("Profile extracted from resume via GPT")
            return extracted
        except Exception as exc:
            logger.warning(f"GPT resume extraction failed: {exc}")

    # ── Regex fallback ────────────────────────────────────────────────────────
    import re
    lines = [l.strip() for l in resume_text.split("\n") if l.strip()]
    name  = lines[0] if lines else "Alex Johnson"

    # Education — look for degree keywords
    edu = "Not specified"
    for line in lines:
        if re.search(r"\b(ms|bs|ba|mba|phd|master|bachelor|degree)\b", line, re.I):
            edu = line[:80].strip()
            break

    # Skills — look for tech keywords
    tech_kws = [
        "python", "sql", "java", "r ", "scala", "spark", "tableau", "power bi",
        "machine learning", "deep learning", "tensorflow", "pytorch", "keras",
        "scikit", "pandas", "numpy", "aws", "azure", "gcp", "docker", "kubernetes",
        "airflow", "dbt", "looker", "excel", "snowflake", "databricks",
    ]
    skills = []
    for line in lines:
        ll = line.lower()
        for kw in tech_kws:
            if kw in ll and kw.strip().title() not in skills:
                skills.append(kw.strip().title())
    skills = skills[:15]

    return {
        "name":             name,
        "current_title":    "",
        "years_experience": 0,
        "education":        edu,
        "skills":           skills or ["Python", "SQL"],
        "target_roles":     ["Data Scientist", "ML Engineer"],
        "industries":       ["Technology"],
        "seniority_target": "mid",
    }


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 16px;">
        <div style="font-size:2.5rem;">🚀</div>
        <div style="font-size:1.4rem; font-weight:800; letter-spacing:1px;">JobPilot</div>
        <div style="font-size:0.78rem; opacity:0.8;">Smart Job Matcher</div>
    </div>
    """, unsafe_allow_html=True)

    # ── User login ────────────────────────────────────────────────────────────
    st.markdown("**👤 User Account**")
    existing_users = list_users()
    user_names     = [u["display_name"] for u in existing_users]

    login_mode = st.radio("", ["Existing user", "New user"],
                          horizontal=True, label_visibility="collapsed")

    if login_mode == "New user":
        new_name = st.text_input("Your name", placeholder="e.g. Jacob R.")
        if st.button("Create Account", use_container_width=True, type="primary"):
            if new_name.strip():
                uid = new_name.strip().lower().replace(" ", "_")
                create_or_update_user(uid, new_name.strip())
                _login_user(uid)
                st.rerun()
            else:
                st.warning("Enter a name first.")
    else:
        if user_names:
            chosen = st.selectbox("Select account", user_names,
                                  label_visibility="collapsed")
            if st.button("Log In", use_container_width=True, type="primary"):
                uid = next(u["user_id"] for u in existing_users
                           if u["display_name"] == chosen)
                _login_user(uid)
                st.rerun()
        else:
            st.caption("No accounts yet — create one above.")

    # Show logged-in user
    if st.session_state.current_user:
        u = get_user(st.session_state.current_user)
        fb_summary = get_feedback_summary(st.session_state.current_user)
        total_fb   = sum(
            v["count"] for v in fb_summary.values()
            if isinstance(v, dict) and "count" in v
        )
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.12); border-radius:8px;
                    padding:10px 12px; margin:8px 0;">
            <div style="font-weight:700;">✅ {u['display_name']}</div>
            <div style="font-size:0.75rem; opacity:0.8;">
                {total_fb} feedback events saved<br>
                Last seen: {u['last_seen'][:10]}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚪 Log Out", use_container_width=True):
            _save_session_to_db()
            st.session_state.current_user = None
            st.session_state.profile      = None
            st.session_state.ranked_jobs  = []
            st.session_state.feedback     = {}
            st.session_state.adaptive     = None
            st.session_state.resumes      = {}
            st.session_state.pipeline_ready = False
            st.rerun()

    st.divider()

    # ── Navigation ────────────────────────────────────────────────────────────
    pages = [
        "🏠 Profile Setup",
        "🎯 Job Matches",
        "📄 Resume Generator",
        "📊 Market Analytics",
        "📈 Benchmarks",
        "🧠 My Learning Profile",
    ]
    page = st.radio("Navigate", pages, key="nav_radio",
                    index=pages.index(st.session_state.page)
                    if st.session_state.page in pages else 0)
    st.session_state.page = page

    st.divider()

    # ── Status indicators ─────────────────────────────────────────────────────
    def _status(ok, label):
        st.markdown(f"{'✅' if ok else '⚪'} {label}")

    _status(st.session_state.current_user is not None, "Logged in")
    _status(st.session_state.profile is not None,      "Profile loaded")
    _status(st.session_state.pipeline_ready,           "Pipeline ready")
    _status(len(st.session_state.ranked_jobs) > 0,     "Jobs ranked")
    _status(bool(OPENAI_API_KEY),                      "AI resume enabled")
    _status(bool(JSEARCH_API_KEY),                     "Live jobs enabled")

    st.divider()
    st.markdown(
        f"<div style='font-size:0.70rem; opacity:0.6;'>"
        f"BAX-423 · Spring 2026<br>DB: {db_size_kb()} KB</div>",
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PROFILE SETUP
# ══════════════════════════════════════════════════════════════════════════════
def page_profile():
    st.markdown(
        '<div class="hero"><h1>🚀 JobPilot</h1>'
        '<p>Upload your profile → get ranked job matches → generate a tailored resume</p></div>',
        unsafe_allow_html=True,
    )

    _persona_cached = _get_cached_persona_results()

    tab1, tab2 = st.tabs(
        ["👤 Built-in Personas", "✏️ Custom Profile"]
    )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — Pre-built test personas (cards + instant load)
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("### Choose a built-in persona")
        _builtin_list = list(_BUILTIN_PERSONAS.values())
        _emojis = {"aisha": "👩‍💼", "marcus": "👨‍💻", "priya": "👩‍🔬", "kenji": "👨‍🎓"}
        cols = st.columns(len(_builtin_list))
        for i, persona in enumerate(_builtin_list):
            pid = persona["id"]
            with cols[i]:
                is_cached = pid in _persona_cached
                st.markdown(f"""
                <div style="background:#F0F6FC; border:1px solid #D6E4F0; border-radius:10px;
                            padding:14px; text-align:center; min-height:160px;">
                    <div style="font-size:2rem;">{_emojis.get(pid, '👤')}</div>
                    <div style="font-weight:700; color:#1F4E79; font-size:0.9rem;">{persona['name']}</div>
                    <div style="font-size:0.75rem; color:#5D6D7E; margin-top:4px;">{persona['current_title']}</div>
                    <div style="font-size:0.72rem; color:#27AE60; margin-top:4px;">→ {persona['target_roles'][0]}</div>
                    {'<div style="font-size:0.68rem; color:#27AE60; margin-top:2px;">⚡ Results cached</div>' if is_cached else ''}
                </div>
                """, unsafe_allow_html=True)
                btn_label = f"{'⚡' if is_cached else '▶'} Load {persona['name']}"
                if st.button(btn_label, key=f"builtin_persona_{pid}",
                             use_container_width=True,
                             help="Instant — results pre-loaded at startup" if is_cached else "Run pipeline after loading"):
                    if is_cached:
                        from src.adaptive_learning import AdaptiveLearner
                        st.session_state.profile        = persona
                        st.session_state.ranked_jobs    = _persona_cached[pid]
                        st.session_state.pipeline_ready = True
                        st.session_state.analytics      = None
                        st.session_state.adaptive       = AdaptiveLearner()
                        st.success(f"✅ {persona['name']}'s matches loaded — switch to **Job Matches**")
                    else:
                        hit = _switch_profile(persona)
                        if not hit:
                            st.success(f"✅ Profile set: {persona['name']} — run the pipeline to find matches")
                    st.rerun()


    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — Custom Profile (with PDF auto-fill)
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("### Build Your Custom Profile")

        # ── Step 1: Upload resume PDF (OUTSIDE the form so we can rerun) ──────
        st.markdown(
            "<div style='background:#EBF5FB; border-left:4px solid #2E75B6; "
            "border-radius:6px; padding:10px 16px; margin-bottom:12px;'>"
            "<strong>📎 Upload Resume PDF</strong> — JobPilot will auto-fill your "
            "name, title, skills, and more using AI.</div>",
            unsafe_allow_html=True,
        )

        uploaded_pdf = st.file_uploader(
            "Drop your resume PDF here",
            type=["pdf"],
            label_visibility="collapsed",
            key="resume_uploader_tab2",
        )

        if uploaded_pdf is not None:
            # Only re-extract if it's a new file
            if st.session_state.get("_last_pdf_name") != uploaded_pdf.name:
                with st.spinner("📄 Reading PDF…"):
                    extracted_text = _extract_pdf_text(uploaded_pdf)
                if extracted_text:
                    st.session_state.resume_text_cache = extracted_text
                    st.session_state["_last_pdf_name"] = uploaded_pdf.name
                    st.success(
                        f"✅ Resume read — {len(extracted_text):,} characters extracted."
                    )
                else:
                    st.warning("⚠️ Couldn't extract text. Try pasting it manually below.")

        # Show auto-fill button once we have resume text
        if st.session_state.resume_text_cache:
            af_col1, af_col2 = st.columns([3, 1])
            with af_col1:
                if st.session_state.auto_profile:
                    st.success("✨ Fields below have been auto-filled from your resume. "
                               "Review and adjust as needed, then click **Save Profile**.")
                else:
                    st.caption(
                        "Resume loaded. Click to let AI extract your details into the form."
                    )
            with af_col2:
                if st.button("✨ Auto-fill Fields", type="primary",
                             use_container_width=True, key="autofill_btn"):
                    with st.spinner("🤖 Analysing resume with AI…"):
                        parsed = _extract_profile_from_resume(
                            st.session_state.resume_text_cache
                        )
                    st.session_state.auto_profile = parsed
                    st.rerun()

        st.markdown("---")

        # ── Step 2: Profile form (values seeded from auto_profile if available) ─
        auto = st.session_state.auto_profile  # empty dict → defaults apply

        # Helper to pick between auto-extracted value and hard-coded default
        def _av(key, default):
            return auto.get(key, default) if auto else default

        seniority_opts = ["junior", "mid", "senior"]
        default_sen    = _av("seniority_target", "mid")
        default_sen_i  = seniority_opts.index(default_sen) if default_sen in seniority_opts else 1

        with st.form("profile_form"):
            col1, col2 = st.columns(2)
            with col1:
                name      = st.text_input("Your Name",            value=_av("name", "Alex Johnson"))
                title     = st.text_input("Current Title",         value=_av("current_title", "Data Analyst"))
                exp       = st.number_input("Years of Experience", 0, 40,
                                            value=int(_av("years_experience", 2)))
                edu       = st.text_input("Education",             value=_av("education", "BS Computer Science"))
            with col2:
                salary    = st.number_input("Minimum Salary ($)", 0, 500_000, 90_000, step=5_000)
                seniority = st.selectbox("Seniority Target", seniority_opts, index=default_sen_i)
                remote    = st.checkbox("Remote required?", False)
                visa      = st.checkbox("Need visa sponsorship?", False)

            default_roles = ", ".join(_av("target_roles", ["Data Scientist", "ML Engineer"]))
            target_roles  = st.text_input("Target Roles (comma-separated)", value=default_roles)

            default_skills = ", ".join(_av("skills", ["Python", "SQL", "pandas",
                                                       "scikit-learn", "Tableau"]))
            skills_input   = st.text_area("Your Skills (comma-separated)", value=default_skills)

            locations = st.text_input(
                "Preferred Locations (comma-separated)", "Remote, San Francisco, New York"
            )
            dealbreakers = st.text_input(
                "Dealbreakers (keywords to avoid — comma-separated)", "defense, contract only"
            )

            resume_text = st.text_area(
                "Resume Text (auto-filled from PDF, or paste manually)",
                value=st.session_state.resume_text_cache,
                height=140,
                placeholder="Upload a PDF above to auto-fill, or paste your resume text here…",
            )

            default_inds = ", ".join(_av("industries", ["Technology", "Healthcare", "Finance"]))
            industries   = st.text_input("Industries of interest", value=default_inds)

            submitted = st.form_submit_button(
                "💾 Save Profile", use_container_width=True, type="primary"
            )

        if submitted:
            profile_data = {
                "id":               "custom",
                "name":             name,
                "emoji":            "👤",
                "current_title":    title,
                "years_experience": int(exp),
                "education":        edu,
                "skills":           [s.strip() for s in skills_input.split(",") if s.strip()],
                "target_roles":     [r.strip() for r in target_roles.split(",") if r.strip()],
                "industries":       [i.strip() for i in industries.split(",") if i.strip()],
                "location_preference": locations.split(",")[0].strip() if locations else "Any",
                "locations":        [l.strip() for l in locations.split(",") if l.strip()],
                "remote_required":  remote,
                "salary_min":       int(salary),
                "visa_required":    visa,
                "seniority_target": seniority,
                "dealbreakers":     [d.strip() for d in dealbreakers.split(",") if d.strip()],
                "career_goal":      f"Seeking {target_roles.split(',')[0].strip()} role.",
                "resume_text":      resume_text,
            }
            st.session_state.profile = profile_data
            st.session_state.pipeline_ready = False
            st.session_state.ranked_jobs    = []

            uid = st.session_state.current_user
            # Invalidate any cached results for this custom profile slot —
            # the profile changed so old matches are no longer valid
            stale_key = f"{uid}_custom" if uid else "anon_custom"
            st.session_state.profile_results.pop(stale_key, None)

            if uid:
                save_profile(uid, profile_data)
                st.success(
                    "✅ Profile saved! Run the pipeline below to find your matches."
                )
            else:
                st.success("✅ Profile set. Log in to save it permanently.")

    # ── Current profile preview + pipeline launcher ───────────────────────────
    if st.session_state.profile:
        p = st.session_state.profile
        st.divider()
        st.markdown("### Active Profile")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**{p.get('emoji','')} {p.get('name','')}**")
            st.markdown(f"*{p.get('current_title','')} · {p.get('years_experience',0)} yrs exp*")
            st.markdown(f"🎯 {', '.join(p.get('target_roles', [])[:2])}")
        with col2:
            st.markdown(f"📍 {p.get('location_preference','Any')}")
            st.markdown(f"💰 ${p.get('salary_min', 0):,}+ target")
            st.markdown(f"🏷️ {p.get('seniority_target','mid').title()}-level")
        with col3:
            st.markdown(f"**Skills ({len(p.get('skills', []))})**")
            skills_preview = p.get('skills', [])[:8]
            st.markdown(" ".join(f"`{s}`" for s in skills_preview))

        st.markdown("---")
        _run_pipeline_section()


def _run_pipeline_section():
    """Data loading + indexing — triggered from profile page."""
    profile = st.session_state.profile
    if not profile:
        return

    st.markdown("### 🔧 Run Data Pipeline")

    # ── Data mode banner ──────────────────────────────────────────────────────
    _has_api_key = bool(JSEARCH_API_KEY)
    if _has_api_key:
        st.info(
            "📚 **Live + Preloaded mode** — 35k pre-loaded Kaggle jobs are indexed at startup. "
            "Use **Refresh Live Jobs** to pull real-time postings from JSearch and merge them in.",
            icon="ℹ️",
        )
    else:
        st.success(
            "📦 **Preloaded Data Only mode** — No API key needed. "
            "Matching against the full 35k pre-loaded Kaggle corpus. "
            "Results load from the pre-built index with no internet calls.",
            icon="✅",
        )

    # ── Corpus status ─────────────────────────────────────────────────────────
    n_kg = len(st.session_state.preloaded_kaggle_df) if st.session_state.preloaded_kaggle_df is not None else 0
    n_js = count_jsearch_jobs()
    _stat_cols = st.columns(3 if _has_api_key else 2)
    _stat_cols[0].metric("Pre-loaded jobs", f"{n_kg:,}")
    if _has_api_key:
        _stat_cols[1].metric("Live JSearch jobs", f"{n_js:,}")
        _stat_cols[2].metric("Total indexed", f"{n_kg + n_js:,}")
    else:
        _stat_cols[1].metric("Total indexed", f"{n_kg:,}")

    # ── Retrieval mode ────────────────────────────────────────────────────────
    retrieval_mode = st.selectbox(
        "Retrieval mode",
        ["Hybrid (FAISS + TF-IDF)", "Dense FAISS only"],
        index=0,
        help=(
            "Hybrid: merges dense semantic search (FAISS) with keyword recall (TF-IDF). "
            "Dense: pure embedding cosine similarity — faster, fully semantic."
        ),
    )
    retrieval_mode_key = "dense" if "Dense" in retrieval_mode else "hybrid"

    # ── Cached results banner ─────────────────────────────────────────────────
    if st.session_state.pipeline_ready and st.session_state.ranked_jobs:
        n_cached = len(st.session_state.ranked_jobs)
        st.success(
            f"✅ Pipeline results cached — **{n_cached} matches** ready. "
            "Navigate to **Job Matches** to view, or re-run below to refresh.",
            icon="✅",
        )

    # ── Action buttons ────────────────────────────────────────────────────────
    if _has_api_key:
        col_run, col_refresh = st.columns([3, 2])
        with col_run:
            run_btn = st.button("🚀 Run Pipeline", type="primary", use_container_width=True,
                                help="Match your profile against the full indexed corpus.")
        with col_refresh:
            refresh_btn = st.button(
                "🔄 Refresh Live Jobs", use_container_width=True,
                help="Fetch fresh postings from JSearch and merge into the index.",
            )
    else:
        run_btn = st.button("🚀 Run Pipeline", type="primary", use_container_width=True,
                            help="Match your profile against the 35k pre-loaded job corpus.")
        refresh_btn = False
        st.caption("💡 Add `JSEARCH_API_KEY` to enable live job refresh.")

    if run_btn:
        _run_full_pipeline(retrieval_mode_key)

    if refresh_btn:
        _fetch_and_add_jsearch(profile)


def _fetch_and_add_jsearch(profile: dict) -> None:
    """
    Fetch fresh job postings from JSearch, save them to the database, embed
    them, and merge the new vectors into the on-disk JSearch embedding file.

    This does NOT re-run the full ranking pipeline — it just grows the corpus.
    The user should click Run Pipeline afterwards to see the new jobs ranked.

    Called by the "🔄 Refresh Live Jobs" button on the Pipeline tab.
    """
    if not JSEARCH_API_KEY:
        st.error(
            "⚠️ **JSearch API key not configured.** "
            "Add `JSEARCH_API_KEY` to `.streamlit/secrets.toml` or Streamlit Cloud Secrets.",
            icon="⚠️",
        )
        return

    with st.spinner("Fetching live jobs from JSearch…"):
        try:
            from src.ingest import fetch_multiple_queries
            from src.clean import clean_jobs
            from src.embeddings import embed, build_job_text, add_jsearch_embeddings

            # Build queries from profile target roles
            queries = profile.get("target_roles", profile.get("target_role", ["data analyst"]))
            if isinstance(queries, str):
                queries = [queries]
            queries = [q.strip() for q in queries if q.strip()][:5]
            if not queries:
                queries = ["data analyst", "data scientist"]

            st.info(f"Querying JSearch: {', '.join(queries)}")

            # ── Fetch + clean ────────────────────────────────────────────
            raw = fetch_multiple_queries(queries, pages_per_query=3, country="us")
            if raw.empty:
                st.warning("JSearch returned no results. Check your API key and try again.")
                return

            live_df = clean_jobs(raw, save=False)
            live_df["source"] = "jsearch"
            live_df = live_df.drop_duplicates(subset=["job_id"]).reset_index(drop=True)

            # ── Deduplicate against existing DB ──────────────────────────
            existing_ids = get_jsearch_job_ids()
            new_df = live_df[~live_df["job_id"].isin(existing_ids)].copy()

            if new_df.empty:
                st.info(
                    f"No new jobs found — all {len(live_df):,} fetched postings are already in the database."
                )
                return

            # ── Save to SQLite ───────────────────────────────────────────
            n_saved = save_jsearch_jobs(new_df)

            # ── Embed new jobs ───────────────────────────────────────────
            if "job_text_clean" in new_df.columns:
                texts = new_df["job_text_clean"].fillna("").tolist()
            else:
                texts = [build_job_text(row) for _, row in new_df.iterrows()]

            vectors = embed(texts, batch_size=128, show_progress=False)
            add_jsearch_embeddings(vectors, new_df["job_id"].tolist())

            # ── Maybe rebuild K-Means clusters ───────────────────────────
            # Triggers if live pool grew ≥25% since last cluster build,
            # or if ≥7 days have elapsed — thresholds from embeddings.py.
            try:
                from src.embeddings import maybe_rebuild_clusters
                maybe_rebuild_clusters()
            except Exception as _cex:
                logger.warning(f"Cluster rebuild check failed (non-fatal): {_cex}")

            # ── Invalidate cached pipeline results ───────────────────────
            st.session_state.pipeline_ready = False

            st.success(
                f"✅ **{n_saved:,} new jobs added** from JSearch and indexed. "
                "Click **Run Pipeline** to see them in your match results.",
                icon="✅",
            )

        except Exception as exc:
            st.error(f"❌ JSearch refresh failed: {exc}", icon="❌")


def _run_full_pipeline(retrieval_mode: str = "hybrid"):
    """
    Phase 2 — profile-dependent scoring and ranking.

    Phase 1 (FAISS index, model, clusters, TF-IDF) is already loaded into
    @st.cache_resource by the time this function is called.  Pipeline now
    only needs to: build the unified corpus, grab cached resources, score
    candidates against the user profile, and rank results.

    Steps:
      1. Build unified corpus (Kaggle parquet + JSearch from DB)
      2. Retrieve pre-loaded FAISS index from cache (instant — already in RAM)
         Merge any JSearch embeddings into a per-run copy of the index
      3. Retrieve cached cluster labels (instant)
      4. Assemble match pool = full unified corpus
      4b. Score: embed profile → FAISS ANN + TF-IDF → RRF merge
      5. Rank with adaptive weights

    Cold start:  ~2s  (Phase 1 already warmed the heavy resources)
    Warm start:  <1s  (all cache_resource hits + per-profile result cache)

    retrieval_mode: "hybrid" = dense FAISS + TF-IDF merged via RRF
                   "dense"  = dense FAISS cosine only
    """
    profile = st.session_state.profile

    with st.spinner("Running JobPilot pipeline..."):
        progress = st.progress(0)
        status   = st.empty()

        try:
            import numpy as np
            from src.dedupe import full_deduplication
            from src.embeddings import (
                load_or_build_index, prebuilt_index_exists,
                build_job_clusters, get_cluster_labels,
                embed_and_score_live_jobs, tfidf_retrieve,
                load_jsearch_embeddings,
            )
            from src.ranker import rank_jobs
            from src.adaptive_learning import AdaptiveLearner

            # ─────────────────────────────────────────────────────────────────
            # STEP 1: Build unified corpus (Kaggle + saved JSearch)
            # ─────────────────────────────────────────────────────────────────
            kaggle_df = st.session_state.preloaded_kaggle_df

            if kaggle_df is None:
                status.empty(); progress.empty()
                st.error(
                    "❌ Job metadata not found. "
                    "Run `python scripts/build_preloaded_data.py` to generate `data/job_meta.parquet`."
                )
                return

            status.text(f"📥 Step 1/5: Building unified corpus ({len(kaggle_df):,} Kaggle jobs)…")

            # Merge any JSearch jobs saved to the database
            jsearch_db_df = load_jsearch_jobs()
            if not jsearch_db_df.empty:
                for col in kaggle_df.columns:
                    if col not in jsearch_db_df.columns:
                        jsearch_db_df[col] = "" if kaggle_df[col].dtype == object else 0
                source_df = pd.concat(
                    [kaggle_df, jsearch_db_df[kaggle_df.columns]],
                    ignore_index=True,
                )
                status.text(
                    f"📥 Step 1/5: Unified corpus — "
                    f"{len(kaggle_df):,} Kaggle + {len(jsearch_db_df):,} JSearch = "
                    f"{len(source_df):,} total jobs…"
                )
            else:
                source_df = kaggle_df.copy()

            if "job_id" not in source_df.columns:
                source_df["job_id"] = source_df.index.astype(str) + "_corpus"
            source_df = source_df.drop_duplicates(subset=["job_id"]).reset_index(drop=True)

            progress.progress(15)

            # ─────────────────────────────────────────────────────────────────
            # STEPS 2+3: Retrieve cached FAISS index + K-Means clusters
            #
            # FAST PATH  — _get_cached_index() already loaded these into RAM
            #   during Phase 1 (app startup).  This call is instant.
            #   We make a shallow copy of the index to safely append JSearch
            #   vectors without dirtying the server-wide cached object.
            #
            # SLOW PATH  — cache returned (None, None, None) meaning the
            #   pre-built index files don't exist yet (first deploy / local dev).
            #   Falls through to full dedup + embed rebuild (~3-4 min, once only).
            # ─────────────────────────────────────────────────────────────────
            cached_index, cached_embeddings, cached_job_ids = _get_cached_index()
            _use_fast_path = cached_index is not None

            if _use_fast_path:
                status.text("⚡ Step 2/5: Retrieving pre-built index from cache (instant)…")

                # Clone the FAISS index so we can add JSearch vectors without
                # mutating the shared cache_resource object.
                import faiss as _faiss
                index = _faiss.deserialize_index(_faiss.serialize_index(cached_index))
                embeddings       = cached_embeddings.copy()
                training_job_ids = list(cached_job_ids)

                # Append any saved JSearch embeddings to the per-run index copy
                js_vecs, js_ids = load_jsearch_embeddings()
                if js_vecs is not None and len(js_ids) > 0:
                    index.add(js_vecs)
                    embeddings = np.vstack([embeddings, js_vecs])
                    training_job_ids = training_job_ids + js_ids
                    status.text(
                        f"⚡ Step 2/5: Cached index ready + {len(js_ids):,} JSearch vectors merged…"
                    )

                training_df = source_df
                st.session_state.data_stats = {
                    "prebuilt":     True,
                    "corpus_size":  len(training_df),
                    "jsearch_jobs": len(js_ids) if js_vecs is not None else 0,
                    "note": "Pre-built Kaggle index served from Phase 1 cache",
                }
                progress.progress(30)

                # ── Step 3: Clusters from cache ───────────────────────────────
                status.text("⚡ Step 3/5: Retrieving job-family clusters from cache…")
                cached_cluster_labels, cached_cluster_ids = _get_cached_clusters()

                if cached_cluster_labels is not None:
                    # Build a job_id → cluster_label lookup from the cached arrays
                    _id_to_cluster = dict(zip(cached_cluster_ids, cached_cluster_labels))
                    cluster_labels = np.array(
                        [_id_to_cluster.get(jid, 0) for jid in training_job_ids],
                        dtype=np.int32,
                    )
                else:
                    # Clusters not yet built — build them now (first run only)
                    status.text("🧠 Step 3/5: Building job-family clusters (first time only)…")
                    cluster_labels = build_job_clusters(embeddings, training_job_ids)

            else:
                # SLOW PATH — full dedup + embed (first deploy, no pre-built index)
                status.text("🔍 Step 2/5: Deduplicating corpus (MinHash LSH)…")
                training_df, dedup_stats = full_deduplication(source_df)
                st.session_state.data_stats = dedup_stats
                progress.progress(30)

                status.text("🧠 Step 3/5: Building embedding index + job-family clusters…")
                if "embeddable" in training_df.columns:
                    embeddable_df = training_df[training_df["embeddable"] == True].copy()
                else:
                    embeddable_df = training_df.copy()

                index, embeddings, training_job_ids = load_or_build_index(embeddable_df)
                cluster_labels = get_cluster_labels(training_job_ids)
                if cluster_labels is None:
                    cluster_labels = build_job_clusters(embeddings, training_job_ids)

            st.session_state.faiss_index    = index
            st.session_state.job_ids        = training_job_ids
            st.session_state.jobs_df        = training_df
            st.session_state.cluster_labels = cluster_labels
            progress.progress(50)

            # ─────────────────────────────────────────────────────────────────
            # STEP 4: Assemble match pool = full unified corpus
            # ─────────────────────────────────────────────────────────────────
            path_label = "⚡ pre-built" if _use_fast_path else "🧠 embedded"
            status.text(
                f"🎯 Step 4/5: Preparing {len(training_df):,} jobs as match pool ({path_label})…"
            )
            live_df = training_df.copy().drop_duplicates(subset=["job_id"]).reset_index(drop=True)
            st.session_state.live_jobs_df = live_df
            st.toast(f"✅ {len(live_df):,} jobs ready ({path_label})")

            # ─────────────────────────────────────────────────────────────────
            # STEP 4b: Embed + score match pool
            # ─────────────────────────────────────────────────────────────────
            preferred_clusters, avoided_clusters = set(), set()
            live_cluster_map = st.session_state.live_cluster_map or {}
            if st.session_state.feedback and live_cluster_map:
                pos_counts: dict[int, int] = {}
                neg_counts: dict[int, int] = {}
                for job_id, fb_type in st.session_state.feedback.items():
                    cluster = live_cluster_map.get(job_id)
                    if cluster is None:
                        continue
                    if fb_type in ("good", "save"):
                        pos_counts[cluster] = pos_counts.get(cluster, 0) + 1
                    elif fb_type == "bad":
                        neg_counts[cluster] = neg_counts.get(cluster, 0) + 1
                preferred_clusters = {c for c, n in pos_counts.items() if n >= 2}
                avoided_clusters   = {c for c, n in neg_counts.items() if n >= 2} - preferred_clusters

            dense_candidates, new_cluster_map, job_embs, profile_emb = embed_and_score_live_jobs(
                profile, live_df,
                preferred_clusters=preferred_clusters,
                avoided_clusters=avoided_clusters,
                retrieval_mode=retrieval_mode,
            )
            st.session_state.live_cluster_map = {**live_cluster_map, **new_cluster_map}

            # TF-IDF retrieval — always computed for benchmarking; merged only in hybrid mode.
            # Pass cached vectorizer + matrix from Phase 1 warm-up if available.
            # This skips the ~2s fit() call and only runs a fast transform() instead.
            _tfidf_cache = _get_cached_tfidf(_CORPUS_HASH)
            _vec, _mat, _cids = _tfidf_cache if _tfidf_cache[0] is not None else (None, None, None)
            tfidf_candidates = tfidf_retrieve(
                profile, live_df,
                k=min(RETRIEVAL_K, len(live_df)),
                _vectorizer=_vec,
                _tfidf_matrix=_mat,
                _cached_job_ids=_cids,
            )
            st.session_state.tfidf_candidates = tfidf_candidates

            if retrieval_mode == "hybrid":
                # RRF merge: dense + tfidf
                from src.embeddings import reciprocal_rank_fusion
                try:
                    hybrid_candidates = reciprocal_rank_fusion(dense_candidates, tfidf_candidates)
                except Exception:
                    hybrid_candidates = dense_candidates  # graceful fallback
            else:
                hybrid_candidates = dense_candidates  # dense only

            st.session_state.hybrid_candidates = hybrid_candidates
            st.session_state.emb_candidates    = dense_candidates

            # Store embeddings for feedback-impact visualisation
            st.session_state.profile_emb = profile_emb
            st.session_state.job_embs    = job_embs

            progress.progress(80)

            # ─────────────────────────────────────────────────────────────────
            # STEP 5: Rank
            # ─────────────────────────────────────────────────────────────────
            status.text("🏆 Step 5/5: Ranking job matches…")

            if st.session_state.adaptive is None:
                st.session_state.adaptive = AdaptiveLearner()

            # Snapshot BEFORE feedback adaptation — used in Benchmarks tab
            ranked_pre = rank_jobs(
                live_df, profile,
                hybrid_candidates,
                weights=None,           # default weights, no feedback
                feedback={},
            )
            st.session_state.pre_feedback_top10 = [
                {"rank": i + 1, "title": j.title, "company": j.company,
                 "score": j.final_score, "job_id": j.job_id}
                for i, j in enumerate(ranked_pre[:10])
            ]

            # Full ranking with adaptive weights + feedback
            ranked = rank_jobs(
                live_df, profile,
                hybrid_candidates,
                weights=st.session_state.adaptive.weights,
                feedback=st.session_state.feedback,
            )
            st.session_state.ranked_jobs    = ranked
            st.session_state.pipeline_ready = True
            st.session_state.retrieval_mode = retrieval_mode
            progress.progress(100)

            # Auto-save job list to SQLite
            uid = st.session_state.current_user
            if uid and ranked:
                save_job_list(uid, ranked)

            # Save results to per-profile in-memory cache
            _save_to_profile_cache()

            # Analytics
            from src.analytics import get_full_analytics
            st.session_state.analytics = get_full_analytics(training_df, profile)

            # Benchmarks
            from src.ranker import benchmark_ranking
            from src.embeddings import benchmark_retrieval
            st.session_state.benchmark_data = {
                "retrieval": benchmark_retrieval(
                    profile, training_df, index, training_job_ids,
                    cluster_labels=cluster_labels,
                ),
                "ranking": benchmark_ranking(
                    live_df, profile, hybrid_candidates, tfidf_candidates
                ),
            }

            status.empty()
            mode_label = "Dense FAISS" if retrieval_mode == "dense" else "Hybrid FAISS+TF-IDF"
            n_jsearch_in_pool = sum(1 for jid in live_df["job_id"] if "_jsearch" in str(jid) or
                                    live_df.loc[live_df["job_id"] == jid, "source"].eq("jsearch").any())
            pool_label = f"{len(live_df):,} unified jobs"
            st.success(
                f"✅ Pipeline complete! **{len(ranked)} matches** from **{pool_label}** · "
                f"Retrieval: **{mode_label}**."
            )
            time.sleep(0.5)
            st.session_state.page = "🎯 Job Matches"
            st.rerun()

        except Exception as e:
            status.empty()
            progress.empty()
            st.error(f"Pipeline error: {e}")
            import traceback
            st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — JOB MATCHES
# ══════════════════════════════════════════════════════════════════════════════
def page_matches():
    if not st.session_state.ranked_jobs:
        st.warning("⚠️ No job matches yet. Go to **Profile Setup** and run the pipeline.")
        return

    ranked = st.session_state.ranked_jobs
    profile = st.session_state.profile
    feedback = st.session_state.feedback
    adaptive = st.session_state.adaptive

    # ── "Ranked for" profile banner ───────────────────────────────────────────
    if profile and profile.get("name"):
        pname  = profile["name"]
        pemoji = profile.get("emoji", "👤")
        ptitle = profile.get("current_title", "")
        ploc   = profile.get("location_preference", "Any")
        psen   = profile.get("seniority_target", "mid").title()
        st.markdown(f"""
        <div style="background:linear-gradient(90deg,#1F4E79 0%,#2E75B6 100%);
                    color:white; border-radius:10px; padding:14px 22px;
                    margin-bottom:18px; display:flex; align-items:center; gap:14px;">
            <span style="font-size:1.8rem;">{pemoji}</span>
            <div>
                <div style="font-size:1.05rem; font-weight:800; line-height:1.2;">
                    Job Matches for {pname}
                </div>
                <div style="font-size:0.82rem; opacity:0.85; margin-top:2px;">
                    {ptitle} &nbsp;·&nbsp; {psen}-level &nbsp;·&nbsp; 📍 {ploc}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Header metrics ────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Jobs Matched", len(ranked))
    with col2:
        avg_score = np.mean([j.final_score for j in ranked[:10]])
        st.metric("Avg Match Score (Top 10)", f"{avg_score:.1%}")
    with col3:
        fb_pos = sum(1 for v in feedback.values() if v in ("good","save"))
        st.metric("Positive Feedback Given", fb_pos)
    with col4:
        remote_count = sum(1 for j in ranked if j.remote)
        st.metric("Remote Positions", remote_count)

    st.divider()

    # ── Data source banner ────────────────────────────────────────────────────
    live_count    = sum(1 for j in ranked if j.source == "jsearch")
    kaggle_count  = sum(1 for j in ranked if j.source not in ("jsearch", "synthetic"))
    synth_count   = sum(1 for j in ranked if j.source == "synthetic")
    training_size = len(st.session_state.jobs_df) if st.session_state.jobs_df is not None else 0
    mode_label    = (
        "Dense FAISS" if st.session_state.retrieval_mode == "dense"
        else "Hybrid FAISS+TF-IDF"
    )

    if synth_count > 0:
        st.warning(
            f"⚠️ **{synth_count} synthetic jobs detected.** "
            "Check that `JSEARCH_API_KEY` is set correctly in Streamlit Secrets.",
            icon="⚠️",
        )
    else:
        source_breakdown = f"{kaggle_count} Kaggle"
        if live_count > 0:
            source_breakdown += f" · {live_count} live JSearch"
        st.success(
            f"🟢 **{len(ranked)} matches** ({source_breakdown}) · "
            f"Retrieval: **{mode_label}** · "
            f"Corpus: **{training_size:,} total jobs**.",
            icon="✅",
        )

    # ── Filters ───────────────────────────────────────────────────────────────
    with st.expander("🔧 Filter Results", expanded=False):
        fcol1, fcol2, fcol3, fcol4 = st.columns(4)
        with fcol1:
            min_score = st.slider("Min match score", 0.0, 1.0, 0.0, 0.05)
        with fcol2:
            remote_filter = st.selectbox("Location", ["All", "Remote only", "On-site only"])
        with fcol3:
            seniority_filter = st.multiselect("Seniority", ["junior","mid","senior","staff"],
                                               default=["junior","mid","senior","staff"])
        with fcol4:
            show_n = st.slider("Show top N", 5, len(ranked), min(10, len(ranked)))

    # Apply filters
    filtered = [j for j in ranked
                if j.final_score >= min_score
                and j.seniority in seniority_filter
                and (remote_filter == "All"
                     or (remote_filter == "Remote only" and j.remote)
                     or (remote_filter == "On-site only" and not j.remote))
               ][:show_n]

    # ── Download CSV ──────────────────────────────────────────────────────────
    col_dl1, col_dl2 = st.columns([3,1])
    with col_dl2:
        if filtered:
            df_export = pd.DataFrame([{
                "Rank":        j.rank,
                "Title":       j.title,
                "Company":     j.company,
                "Location":    j.location,
                "Remote":      j.remote,
                "Salary Min":  j.salary_min,
                "Salary Max":  j.salary_max,
                "Match Score": f"{j.final_score:.1%}",
                "Description": j.description[:300],
                "Apply Link":  j.url,
                "Source":      j.source,
            } for j in filtered])
            csv_bytes = df_export.to_csv(index=False).encode()
            st.download_button("⬇️ Download CSV", csv_bytes,
                               "top_jobs.csv", "text/csv",
                               use_container_width=True)

    # ── Job cards ─────────────────────────────────────────────────────────────
    for job in filtered:
        _render_job_card(job, profile, adaptive, feedback)

    # Re-rank after feedback
    st.markdown("---")
    rcol1, rcol2 = st.columns([2, 1])
    with rcol1:
        st.caption(
            "💡 Rate jobs with 👍 / 💾 / 👎 above, then click **Re-rank** to see "
            "the adaptive model reprioritise your top 20 based on your preferences."
        )
    with rcol2:
        if st.button("🔄 Re-rank with Feedback", type="primary", use_container_width=True):
            from src.ranker import rank_jobs
            candidates = (st.session_state.hybrid_candidates
                          or st.session_state.emb_candidates)
            # Always rank against the live jobs pool, not the training corpus
            rank_df = (st.session_state.live_jobs_df
                       if st.session_state.live_jobs_df is not None
                       else st.session_state.jobs_df)
            ranked_new = rank_jobs(
                rank_df,
                profile,
                candidates,
                weights=adaptive.weights if adaptive else None,
                feedback=feedback,
            )
            if adaptive:
                ranked_new = adaptive.apply_bandit_boost(ranked_new)
                ranked_new.sort(key=lambda j: j.final_score, reverse=True)
                for i, j in enumerate(ranked_new):
                    j.rank = i + 1
            # Show a diff summary: how many positions changed
            old_ids = [j.job_id for j in st.session_state.ranked_jobs]
            new_ids = [j.job_id for j in ranked_new]
            moved   = sum(1 for i, jid in enumerate(new_ids)
                          if i < len(old_ids) and jid != old_ids[i])
            st.session_state.ranked_jobs = ranked_new
            st.success(f"✅ Re-ranked! **{moved}** positions changed in the top {len(ranked_new)}.")
            st.rerun()


def _render_job_card(job, profile, adaptive, feedback):
    """Render a single job card with scores, explanations, and feedback buttons."""
    fb = feedback.get(job.job_id, "")
    border_color = {"good":"#27AE60","save":"#2E75B6","bad":"#E74C3C","skip":"#95A5A6"}.get(fb, "#2E75B6")
    score_pct = int(job.final_score * 100)
    score_class = "score-high" if score_pct >= 70 else "score-mid" if score_pct >= 45 else "score-low"

    # Source badge
    source_badge_map = {
        "jsearch":   ('<span style="background:#27AE60;color:white;padding:2px 8px;'
                      'border-radius:4px;font-size:0.72rem;font-weight:600;">🟢 LIVE</span>'),
        "kaggle":    ('<span style="background:#2E75B6;color:white;padding:2px 8px;'
                      'border-radius:4px;font-size:0.72rem;font-weight:600;">📦 KAGGLE</span>'),
        "synthetic": ('<span style="background:#E67E22;color:white;padding:2px 8px;'
                      'border-radius:4px;font-size:0.72rem;font-weight:600;">🔶 DEMO</span>'),
    }
    src_badge = source_badge_map.get(job.source, "")

    with st.container():
        st.markdown(f"""
        <div class="job-card" style="border-left-color:{border_color}">
          <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap;">
            <div>
              <span style="font-size:1.1rem; font-weight:700; color:#1F4E79;">
                #{job.rank} {job.title}
              </span>
              &nbsp;{src_badge}
              <br>
              <span style="color:#5D6D7E; font-size:0.88rem;">
                🏢 {job.company} &nbsp;|&nbsp;
                📍 {job.location} &nbsp;|&nbsp;
                {'🌐 Remote' if job.remote else '🏢 On-site'} &nbsp;|&nbsp;
                {job.employment_type}
              </span>
            </div>
            <div style="text-align:right;">
              <span class="score-badge {score_class}">{score_pct}% match</span>
              {f'<br><span style="color:#5D6D7E; font-size:0.8rem;">${job.salary_min:,.0f}–${job.salary_max:,.0f}</span>' if job.salary_max > 0 else ''}
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander(f"📋 Details & Explanation — {job.title} at {job.company}", expanded=False):
            exp_col1, exp_col2 = st.columns([3, 2])

            with exp_col1:
                # Why ranked explanation
                st.markdown("**🎯 Why This Ranked Here**")
                st.markdown(job.why_ranked)

                # Score breakdown
                st.markdown("**📊 Score Breakdown**")
                breakdown_data = {
                    "Dimension": ["Semantic Match", "Skill Match", "Title Alignment",
                                  "Location Fit", "Salary Fit", "Recency"],
                    "Score": [
                        f"{job.embedding_score:.1%}",
                        f"{job.skill_match_score:.1%}",
                        f"{job.title_match_score:.1%}",
                        f"{job.location_fit_score:.1%}",
                        f"{job.salary_fit_score:.1%}",
                        f"{job.recency_score:.1%}",
                    ],
                }
                st.dataframe(pd.DataFrame(breakdown_data), hide_index=True, use_container_width=True)

                # Job description preview
                st.markdown("**📝 Job Description**")
                st.markdown(job.description[:600] + "..." if len(job.description) > 600 else job.description)
                if job.url:
                    st.markdown(f"[🔗 View Full Posting]({job.url})")

            with exp_col2:
                # Skills visualization
                st.markdown("**✅ Matched Skills**")
                if job.matched_skills:
                    pills = " ".join(
                        f'<span class="skill-pill skill-matched">{s}</span>'
                        for s in job.matched_skills[:8]
                    )
                    st.markdown(f'<div>{pills}</div>', unsafe_allow_html=True)
                else:
                    st.caption("No direct skill matches found")

                st.markdown("**⚠️ Missing Skills**")
                if job.missing_skills:
                    pills = " ".join(
                        f'<span class="skill-pill skill-missing">{s}</span>'
                        for s in job.missing_skills[:5]
                    )
                    st.markdown(f'<div>{pills}</div>', unsafe_allow_html=True)
                else:
                    st.caption("You have all listed required skills!")

                # Metadata
                st.markdown("**ℹ️ Details**")
                st.markdown(f"- **Seniority:** {job.seniority.title()}")
                st.markdown(f"- **Exp. Required:** {job.experience_required}+ yrs")
                st.markdown(f"- **Visa:** {'✅ Sponsorship indicated' if job.visa_possible else '❓ Not specified'}")
                st.markdown(f"- **Posted:** {job.date_posted}")
                st.markdown(f"- **Source:** {job.source}")

        # Feedback + action buttons
        btn_cols = st.columns([1, 1, 1, 1, 2])
        with btn_cols[0]:
            if st.button("✅ Good Fit", key=f"good_{job.job_id}",
                         type="primary" if fb == "good" else "secondary"):
                _record_feedback(job, "good", adaptive)
        with btn_cols[1]:
            if st.button("❌ Not For Me", key=f"bad_{job.job_id}"):
                _record_feedback(job, "bad", adaptive)
        with btn_cols[2]:
            if st.button("⭐ Save", key=f"save_{job.job_id}"):
                _record_feedback(job, "save", adaptive)
        with btn_cols[3]:
            if st.button("⏭️ Skip", key=f"skip_{job.job_id}"):
                _record_feedback(job, "skip", adaptive)
        with btn_cols[4]:
            if st.button(f"📄 Generate Resume", key=f"resume_{job.job_id}",
                         type="primary"):
                st.session_state.selected_job = job
                st.session_state.page = "📄 Resume Generator"
                st.rerun()

        if fb:
            fb_label = {"good":"✅ Marked: Good fit","bad":"❌ Marked: Not for me",
                        "save":"⭐ Saved","skip":"⏭️ Skipped"}.get(fb,"")
            st.caption(fb_label)

        st.markdown("")


def _record_feedback(job, feedback_type, adaptive):
    """
    Record feedback, update adaptive learner, and persist to database.
    Every click is saved immediately — nothing is lost on refresh.
    """
    st.session_state.feedback[job.job_id] = feedback_type

    if adaptive:
        adaptive.record_feedback(job, feedback_type)
        if feedback_type in ("good", "save"):
            st.session_state.positive_ids.add(job.job_id)
        if adaptive.bandit.total_interactions % 5 == 0 and st.session_state.ranked_jobs:
            adaptive.record_precision(
                st.session_state.ranked_jobs,
                st.session_state.positive_ids
            )

    # ── Persist to SQLite ─────────────────────────────────────────────────────
    uid = st.session_state.current_user
    if uid:
        save_feedback_event(uid, job, feedback_type)          # log the event
        save_bandit_state(uid, adaptive.bandit)               # save arm distributions
        save_ranking_weights(uid, adaptive.weights)           # save updated weights

    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RESUME GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
def page_resume():
    st.markdown('<div class="section-header">📄 Resume Generator</div>', unsafe_allow_html=True)

    profile = st.session_state.profile
    ranked  = st.session_state.ranked_jobs

    if not ranked:
        st.warning("⚠️ No jobs ranked yet. Run the pipeline first.")
        return

    if not profile:
        st.warning("⚠️ No profile loaded.")
        return

    # Job selector
    col1, col2 = st.columns([3, 1])
    with col1:
        job_options = {f"#{j.rank} {j.title} — {j.company}": j for j in ranked[:20]}
        selected_label = st.selectbox("Select a job to tailor your resume for", list(job_options.keys()))
        selected_job = job_options[selected_label]
    with col2:
        st.markdown("")
        st.markdown("")
        generate_btn = st.button("🤖 Generate Tailored Resume", type="primary", use_container_width=True)

    if generate_btn or (st.session_state.selected_job and st.session_state.selected_job.job_id == selected_job.job_id):
        if selected_job.job_id in st.session_state.resumes:
            result = st.session_state.resumes[selected_job.job_id]
        else:
            with st.spinner("✍️ Generating tailored resume..."):
                from src.resume_generator import generate_resume
                result = generate_resume(profile, selected_job)
                st.session_state.resumes[selected_job.job_id] = result
                # Persist resume to database
                uid = st.session_state.current_user
                if uid:
                    save_resume(uid, selected_job, result)

        # Display result
        st.markdown(f"""
        <div style="background:#D5F5E3; border-left:4px solid #27AE60;
             border-radius:8px; padding:12px 16px; margin-bottom:16px;">
            <strong>{'🤖 AI-Generated' if result['method']=='ai' else '📋 Template'} Resume</strong>
            — tailored for <strong>{selected_job.title}</strong> at <strong>{selected_job.company}</strong>
        </div>
        """, unsafe_allow_html=True)

        st.warning(result["warning"])

        col_r1, col_r2 = st.columns([3, 1])
        with col_r1:
            st.markdown('<div class="resume-output">', unsafe_allow_html=True)
            st.markdown(result["markdown"])
            st.markdown('</div>', unsafe_allow_html=True)

        with col_r2:
            # Download
            st.download_button(
                "⬇️ Download (.md)",
                result["markdown"].encode(),
                f"resume_{selected_job.company.replace(' ','_')}.md",
                "text/markdown",
                use_container_width=True,
            )

            st.markdown("**✅ Your Matched Skills**")
            for s in result["matched_skills"][:8]:
                st.markdown(f"- {s}")

            st.markdown("**⚠️ Skill Gaps**")
            if result["missing_skills"]:
                for s in result["missing_skills"][:5]:
                    st.markdown(f"- ❌ {s}")
            else:
                st.markdown("*You cover all listed requirements!*")

    # Show all previously generated resumes
    if st.session_state.resumes:
        st.divider()
        st.markdown(f"**Generated {len(st.session_state.resumes)} resume(s) this session**")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MARKET ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
def page_analytics():
    st.markdown('<div class="section-header">📊 Market Analytics</div>', unsafe_allow_html=True)

    analytics = st.session_state.analytics
    if not analytics:
        st.warning("⚠️ Run the pipeline first to see analytics.")
        return

    from src.analytics import (
        plot_top_skills, plot_salary_distribution, plot_remote_pie,
        plot_top_companies, plot_skill_gaps
    )

    # Summary metrics
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.markdown(f'<div class="metric-card"><div class="metric-number">{analytics["total_jobs"]:,}</div><div class="metric-label">Total Jobs</div></div>', unsafe_allow_html=True)
    with mcol2:
        sal_pct = int(analytics["with_salary"] / max(analytics["total_jobs"],1) * 100)
        st.markdown(f'<div class="metric-card"><div class="metric-number">{sal_pct}%</div><div class="metric-label">Jobs with Salary Data</div></div>', unsafe_allow_html=True)
    with mcol3:
        rem_pct = int(analytics["remote_count"] / max(analytics["total_jobs"],1) * 100)
        st.markdown(f'<div class="metric-card"><div class="metric-number">{rem_pct}%</div><div class="metric-label">Remote Positions</div></div>', unsafe_allow_html=True)
    with mcol4:
        gaps_count = len(analytics["skill_gaps"])
        st.markdown(f'<div class="metric-card"><div class="metric-number">{gaps_count}</div><div class="metric-label">Skill Gaps Found</div></div>', unsafe_allow_html=True)

    st.markdown("")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🔧 Top Skills", "💰 Salaries", "🌐 Remote Split", "🏢 Companies", "📉 Your Skill Gaps"]
    )

    with tab1:
        if not analytics["top_skills"].empty:
            st.plotly_chart(plot_top_skills(analytics["top_skills"]), use_container_width=True)
            st.dataframe(analytics["top_skills"], hide_index=True, use_container_width=True)

    with tab2:
        sal_df = analytics["salary_dist"]
        if not sal_df.empty:
            st.plotly_chart(plot_salary_distribution(sal_df), use_container_width=True)
            st.caption("Box plots show median, IQR, and outliers for jobs with listed salary data.")
        else:
            st.info("No salary data available in current dataset.")

    with tab3:
        rd = analytics["remote_dist"]
        if rd:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.plotly_chart(plot_remote_pie(rd), use_container_width=True)
            with col2:
                for label, count in rd.items():
                    pct = count / max(sum(rd.values()),1) * 100
                    st.markdown(f"**{label}:** {count:,} ({pct:.1f}%)")

    with tab4:
        if not analytics["top_companies"].empty:
            st.plotly_chart(plot_top_companies(analytics["top_companies"]), use_container_width=True)

    with tab5:
        gaps_df = analytics["skill_gaps"]
        if not gaps_df.empty:
            st.plotly_chart(plot_skill_gaps(gaps_df), use_container_width=True)
            st.markdown("*These skills appear frequently in your target roles but are not in your current profile.*")
            st.dataframe(gaps_df, hide_index=True, use_container_width=True)
        else:
            st.success("No significant skill gaps found for your target roles!")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — BENCHMARKS
# ══════════════════════════════════════════════════════════════════════════════
def page_benchmarks():
    st.markdown('<div class="section-header">📈 Benchmarks & Technical Results</div>', unsafe_allow_html=True)

    bm  = st.session_state.benchmark_data
    ada = st.session_state.adaptive

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🧠 Retrieval Comparison", "🏆 Ranking Pipeline", "🤖 Adaptive Learning",
         "✅ Persona Tests", "🔄 Feedback Impact"]
    )

    # ── Retrieval benchmark ───────────────────────────────────────────────────
    with tab1:
        st.markdown("### Embedding vs. TF-IDF Retrieval (BAX-423 Technique 1)")
        st.markdown("""
        Dense embeddings (sentence-transformers) vs. keyword-based TF-IDF retrieval.
        Embeddings capture semantic equivalence — a resume saying "statistical modeling"
        matches jobs requiring "predictive analytics".
        """)

        retrieval_bm = bm.get("retrieval", {})
        if retrieval_bm:
            ret_df = pd.DataFrame({
                "Method":        retrieval_bm.get("method", []),
                "Recall@10":     retrieval_bm.get("recall_at_10", []),
                "Recall@50":     retrieval_bm.get("recall_at_50", []),
                "Latency (ms)":  retrieval_bm.get("latency_ms_p50", []),
            })
            st.dataframe(ret_df, hide_index=True, use_container_width=True)
            improvement = retrieval_bm.get("improvement", "")
            if improvement:
                st.success(f"📈 Embedding improvement: **{improvement}**")

            import plotly.graph_objects as go
            methods = retrieval_bm.get("method", [])
            r10 = retrieval_bm.get("recall_at_10", [])
            fig = go.Figure([go.Bar(x=methods, y=r10,
                                    marker_color=["#D6E4F0","#1F4E79"],
                                    text=[f"{v:.0%}" for v in r10],
                                    textposition="auto")])
            fig.update_layout(title="Recall@10 Comparison", yaxis_tickformat=".0%",
                               plot_bgcolor="white", height=300,
                               margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run the pipeline to see retrieval benchmarks.")

    # ── Ranking benchmark ─────────────────────────────────────────────────────
    with tab2:
        st.markdown("### Multi-Stage Ranking Pipeline (BAX-423 Technique 2)")
        st.markdown("""
        TF-IDF → Embedding-only → Full multi-stage pipeline (hard filters + scoring + MMR re-ranking).
        Reports persona fit score (fraction of top-10 that match persona's target roles)
        and dealbreaker violations.
        """)

        ranking_bm = bm.get("ranking", {})
        if ranking_bm:
            rank_df = pd.DataFrame({
                "Method":               ranking_bm.get("method", []),
                "Persona Fit (Top-10)": ranking_bm.get("top10_persona_fit", []),
                "Dealbreaker Violations": ranking_bm.get("dealbreaker_violations", []),
                "Avg Match Score":      ranking_bm.get("avg_match_score", []),
            })
            st.dataframe(rank_df, hide_index=True, use_container_width=True)

            import plotly.graph_objects as go
            methods = ranking_bm.get("method", [])
            fits    = ranking_bm.get("top10_persona_fit", [])
            fig = go.Figure([go.Bar(x=methods, y=fits,
                                    marker_color=["#D6E4F0","#5BA3D0","#1F4E79"],
                                    text=[f"{v:.0%}" for v in fits],
                                    textposition="auto")])
            fig.update_layout(title="Persona Fit Score by Ranking Method",
                               yaxis_tickformat=".0%", plot_bgcolor="white",
                               height=300, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run the pipeline to see ranking benchmarks.")

    # ── Adaptive learning ─────────────────────────────────────────────────────
    with tab3:
        st.markdown("### Thompson Sampling Adaptive Learning (BAX-423 Technique 3)")
        st.markdown("""
        Thompson Sampling models user preferences as Beta distributions over job-feature clusters.
        Weight Updater adjusts ranking formula weights based on feedback correlation.
        """)

        if ada and ada.bandit.total_interactions > 0:
            bm_data = ada.get_benchmark_data()

            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            with metrics_col1:
                st.metric("Feedback Events", bm_data["total_feedback"])
            with metrics_col2:
                fb = bm_data["feedback_breakdown"]
                pos = fb.get("good",0) + fb.get("save",0)
                neg = fb.get("bad",0)
                st.metric("Positive / Negative", f"{pos} / {neg}")
            with metrics_col3:
                st.metric("Preference Clusters Learned", len(ada.bandit.arms))

            # Precision@5 curve
            if bm_data["precision_history"]:
                from src.analytics import plot_adaptive_learning_curve, plot_weight_evolution
                st.plotly_chart(
                    plot_adaptive_learning_curve(bm_data["precision_history"]),
                    use_container_width=True
                )

            # Top preferences
            st.markdown("**🎯 Learned Preferences (Top Arms)**")
            pref_df = pd.DataFrame(bm_data["top_preferences"],
                                    columns=["Cluster", "Preference Score"])
            pref_df["Preference Score"] = pref_df["Preference Score"].apply(lambda x: f"{x:.2f}")
            st.dataframe(pref_df, hide_index=True, use_container_width=True)

            # Weight evolution
            if len(bm_data["weight_evolution"]) > 1:
                from src.analytics import plot_weight_evolution
                st.plotly_chart(plot_weight_evolution(bm_data["weight_evolution"]),
                                use_container_width=True)

            # Weight changes
            st.markdown("**⚖️ Weight Changes from Initial**")
            delta = bm_data["weight_changes"]
            delta_df = pd.DataFrame([
                {"Dimension": k.replace("_"," ").title(),
                 "Initial": f"{DEFAULT_WEIGHTS[k]:.2f}",
                 "Current": f"{v:.2f}",
                 "Change":  f"{delta[k]:+.3f}"}
                for k, v in bm_data["current_weights"].items()
            ])
            from src.utils import DEFAULT_WEIGHTS
            st.dataframe(delta_df, hide_index=True, use_container_width=True)
        else:
            st.info("Give feedback on job matches (Good Fit / Not For Me) to see the adaptive learning benchmark.")
            # Show simulated preview
            st.markdown("**📊 Simulated Preview (from test run):**")
            sim_data = {"Round": [0,1,2,3,4],
                        "Signals": [0,5,10,20,30],
                        "Precision@5": [0.40, 0.52, 0.61, 0.68, 0.74],
                        "Dealbreaker Violations": [2,1,1,0,0]}
            st.dataframe(pd.DataFrame(sim_data), hide_index=True, use_container_width=True)

    # ── Persona tests ─────────────────────────────────────────────────────────
    with tab4:
        st.markdown("### Persona Pass/Fail Results")
        st.markdown("Evaluation of the pipeline against all 4 required test personas.")

        # Build results from current session
        ranked = st.session_state.ranked_jobs
        profile = st.session_state.profile

        persona_results = []
        if PERSONAS_FILE.exists():
            with open(PERSONAS_FILE) as f:
                personas = json.load(f)
            for p in personas:
                is_active = profile and profile.get("id") == p["id"]
                criteria  = p.get("pass_criteria", {})
                if is_active and ranked:
                    top10 = ranked[:10]
                    db_violations = sum(
                        1 for j in top10
                        for db in p.get("dealbreakers", [])
                        if db.lower() in (j.title + " " + j.company + " " + j.description[:200]).lower()
                    )
                    target_roles = [r.lower() for r in p["target_roles"]]
                    fit = sum(1 for j in top10 if any(r in j.title.lower() for r in target_roles))
                    passed = db_violations == 0 and fit >= 5
                else:
                    passed = None  # not tested yet

                persona_results.append({
                    "Persona":            p["emoji"] + " " + p["name"].split("—")[0].strip(),
                    "Target Roles":       p["target_roles"][0],
                    "Salary Target":      f"${p['salary_min']:,}+",
                    "Key Dealbreaker":    p["dealbreakers"][0] if p["dealbreakers"] else "—",
                    "Status":             "✅ Active" if is_active else "⚪ Not tested",
                    "Pass":               "✅" if passed is True else "❓ Pending" if passed is None else "❌",
                })

        if persona_results:
            st.dataframe(pd.DataFrame(persona_results), hide_index=True, use_container_width=True)
            st.info("Switch personas in Profile Setup to test each one.")

    # ── Feedback Impact ───────────────────────────────────────────────────────
    with tab5:
        st.markdown("### 🔄 Feedback Impact on Job Rankings")
        st.markdown("""
        Shows how thumbs-up / thumbs-down feedback changed your top job matches
        via **Thompson Sampling** weight adaptation and **cluster preference boosting**.
        Also plots your profile's position in the job embedding space.
        """)

        pre_top10  = st.session_state.pre_feedback_top10
        post_ranked = st.session_state.ranked_jobs
        feedback   = st.session_state.feedback

        if not pre_top10 or not post_ranked:
            st.info("Run the pipeline and give at least one thumbs-up or thumbs-down to see feedback impact.")
        else:
            # ── Before / After table ─────────────────────────────────────────
            st.markdown("#### Before vs After Feedback — Top 10 Jobs")
            post_top10 = [
                {"rank": i + 1, "title": j.title, "company": j.company,
                 "score": j.final_score, "job_id": j.job_id}
                for i, j in enumerate(post_ranked[:10])
            ]

            pre_ids  = [r["job_id"] for r in pre_top10]
            post_ids = [r["job_id"] for r in post_top10]

            rows = []
            for i, post in enumerate(post_top10):
                pre_rank = pre_ids.index(post["job_id"]) + 1 if post["job_id"] in pre_ids else None
                fb_icon  = (
                    "👍" if feedback.get(post["job_id"]) in ("good", "save")
                    else "👎" if feedback.get(post["job_id"]) == "bad"
                    else "—"
                )
                move = ""
                if pre_rank is not None:
                    delta = pre_rank - (i + 1)
                    move = f"▲ +{delta}" if delta > 0 else (f"▼ {delta}" if delta < 0 else "═ 0")
                else:
                    move = "⭐ New"
                rows.append({
                    "Post Rank":    i + 1,
                    "Pre Rank":     pre_rank if pre_rank else "—",
                    "Moved":        move,
                    "Feedback":     fb_icon,
                    "Title":        post["title"][:55],
                    "Company":      post["company"][:30],
                    "Score (Post)": f"{post['score']:.3f}",
                })

            diff_df = pd.DataFrame(rows)
            st.dataframe(diff_df, hide_index=True, use_container_width=True)

            n_new = sum(1 for r in post_top10 if r["job_id"] not in pre_ids)
            n_moved_up = sum(1 for r in rows if isinstance(r["Moved"], str) and r["Moved"].startswith("▲"))
            st.caption(
                f"**{n_new}** new entries entered the top 10 · "
                f"**{n_moved_up}** positions moved upward from pre-feedback ranking"
            )

            # ── PCA scatter of job embeddings + profile vector ───────────────
            st.markdown("#### Profile & Job Position Vectors (PCA Projection)")
            st.markdown("""
            Each point = one job in the match pool, projected to 2D via PCA.
            Your profile vector is plotted at ⭐. Jobs you liked (👍) are green,
            jobs you disliked (👎) are red, unrated jobs are grey.
            Your profile should cluster near your liked jobs after feedback.
            """)

            profile_emb = st.session_state.profile_emb
            job_embs    = st.session_state.job_embs
            live_df     = st.session_state.live_jobs_df

            if profile_emb is not None and job_embs is not None and live_df is not None and len(job_embs) > 0:
                try:
                    from sklearn.decomposition import PCA
                    import plotly.graph_objects as go

                    # Limit to at most 3000 points for rendering speed
                    MAX_PLOT = 3_000
                    job_ids_plot = live_df["job_id"].tolist()
                    if len(job_embs) > MAX_PLOT:
                        idx = np.random.choice(len(job_embs), MAX_PLOT, replace=False)
                        embs_sub = job_embs[idx]
                        ids_sub  = [job_ids_plot[i] for i in idx]
                        titles_sub = live_df["title"].tolist()
                        titles_sub = [titles_sub[i] for i in idx]
                    else:
                        embs_sub   = job_embs
                        ids_sub    = job_ids_plot
                        titles_sub = live_df["title"].tolist()

                    # Stack profile + jobs for joint PCA
                    all_embs = np.vstack([profile_emb.reshape(1, -1), embs_sub])
                    pca = PCA(n_components=2, random_state=42)
                    coords = pca.fit_transform(all_embs)
                    profile_xy = coords[0]
                    job_coords = coords[1:]

                    fig = go.Figure()

                    # Build boolean masks for each feedback category
                    liked_mask    = np.array([feedback.get(jid) in ("good", "save") for jid in ids_sub])
                    disliked_mask = np.array([feedback.get(jid) == "bad"            for jid in ids_sub])
                    unrated_mask  = ~liked_mask & ~disliked_mask

                    # Unrated jobs (grey, smaller)
                    if unrated_mask.any():
                        fig.add_trace(go.Scatter(
                            x=job_coords[unrated_mask, 0],
                            y=job_coords[unrated_mask, 1],
                            mode="markers",
                            name="Unrated",
                            marker=dict(color="#aab7c4", size=5, opacity=0.55),
                            text=[titles_sub[i] for i, m in enumerate(unrated_mask) if m],
                            hovertemplate="%{text}<extra>unrated</extra>",
                        ))

                    # Liked jobs
                    if liked_mask.any():
                        fig.add_trace(go.Scatter(
                            x=job_coords[liked_mask, 0],
                            y=job_coords[liked_mask, 1],
                            mode="markers",
                            name="👍 Liked",
                            marker=dict(color="#27ae60", size=10, symbol="circle",
                                        line=dict(width=1, color="white")),
                            text=[titles_sub[i] for i, m in enumerate(liked_mask) if m],
                            hovertemplate="%{text}<extra>liked</extra>",
                        ))

                    # Disliked jobs
                    if disliked_mask.any():
                        fig.add_trace(go.Scatter(
                            x=job_coords[disliked_mask, 0],
                            y=job_coords[disliked_mask, 1],
                            mode="markers",
                            name="👎 Disliked",
                            marker=dict(color="#e74c3c", size=10, symbol="x",
                                        line=dict(width=1, color="white")),
                            text=[titles_sub[i] for i, m in enumerate(disliked_mask) if m],
                            hovertemplate="%{text}<extra>disliked</extra>",
                        ))

                    # Profile vector star
                    fig.add_trace(go.Scatter(
                        x=[profile_xy[0]],
                        y=[profile_xy[1]],
                        mode="markers+text",
                        name="⭐ Your Profile",
                        marker=dict(color="#f1c40f", size=22, symbol="star",
                                    line=dict(width=2, color="#333")),
                        text=["⭐ You"],
                        textposition="top center",
                        hovertemplate="Your profile vector<extra></extra>",
                    ))

                    explained = pca.explained_variance_ratio_
                    fig.update_layout(
                        title=f"Job Embedding Space (PCA · PC1={explained[0]:.1%}, PC2={explained[1]:.1%})",
                        xaxis_title=f"PC1 ({explained[0]:.1%} variance)",
                        yaxis_title=f"PC2 ({explained[1]:.1%} variance)",
                        plot_bgcolor="white",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                        height=520,
                        margin=dict(l=0, r=0, t=60, b=0),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(
                        f"Showing {len(ids_sub):,} of {len(job_embs):,} jobs. "
                        f"Profile vector proximity to liked (green) jobs validates semantic alignment."
                    )
                except ImportError:
                    st.warning("Install scikit-learn (`pip install scikit-learn`) to enable the PCA plot.")
                except Exception as exc:
                    st.warning(f"PCA plot skipped: {exc}")
            else:
                st.info("Re-run the pipeline to generate embedding vectors for this visualisation.")

    # ── Deduplication stats ───────────────────────────────────────────────────
    if st.session_state.data_stats:
        st.divider()
        st.markdown("### 📊 Deduplication Pipeline Stats")
        ds = st.session_state.data_stats
        dcol1, dcol2, dcol3 = st.columns(3)
        with dcol1:
            st.metric("Original Records", f"{ds.get('original_count',0):,}")
        with dcol2:
            st.metric("After Exact Dedup", f"{ds.get('after_exact',0):,}")
        with dcol3:
            st.metric("After MinHash LSH", f"{ds.get('after_minhash',0):,}")

        mh = ds.get("minhash_stats", {})
        if mh:
            st.markdown(f"""
            - **Threshold:** Jaccard ≥ {mh.get('threshold', 0.85)}
            - **Hash permutations:** {mh.get('num_perm', 128)}
            - **Throughput:** {mh.get('throughput_rps',0):,} records/second
            - **Near-duplicates removed:** {mh.get('removed',0):,}
            """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — MY LEARNING PROFILE
# ══════════════════════════════════════════════════════════════════════════════
def page_learning_profile():
    st.markdown('<div class="section-header">🧠 My Learning Profile</div>',
                unsafe_allow_html=True)

    uid = st.session_state.current_user
    if not uid:
        st.warning("⚠️ Log in first to see your learning profile.")
        return

    insights = get_learning_insights(uid)
    fb_summary = get_feedback_summary(uid)
    adaptive = st.session_state.adaptive

    if not insights:
        st.info("No feedback recorded yet. Rate some jobs on the Job Matches page "
                "and come back here to see what the model has learned about you.")
        return

    # ── Summary metrics ───────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-number">'
                    f'{insights["total_feedback"]}</div>'
                    f'<div class="metric-label">Total Feedback Events</div></div>',
                    unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-number">'
                    f'{insights["liked_count"]}</div>'
                    f'<div class="metric-label">Jobs Liked / Saved</div></div>',
                    unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-number">'
                    f'{insights["sessions_count"]}</div>'
                    f'<div class="metric-label">Sessions Recorded</div></div>',
                    unsafe_allow_html=True)
    with col4:
        arms = len(adaptive.bandit.arms) if adaptive else 0
        st.markdown(f'<div class="metric-card"><div class="metric-number">'
                    f'{arms}</div>'
                    f'<div class="metric-label">Preference Clusters Learned</div></div>',
                    unsafe_allow_html=True)

    st.markdown("")
    col_l, col_r = st.columns(2)

    # ── What the model has learned ────────────────────────────────────────────
    with col_l:
        st.markdown("### ✅ What You Tend to Like")

        if insights["preferred_seniority"]:
            st.markdown("**Seniority levels:**")
            for level, count in insights["preferred_seniority"]:
                st.markdown(f"- {level.title()} ({count} likes)")

        if insights["preferred_industry"]:
            st.markdown("**Industries:**")
            for ind, count in insights["preferred_industry"]:
                st.markdown(f"- {ind.replace('_',' ').title()} ({count} likes)")

        if insights["top_matched_skills"]:
            st.markdown("**Most valued skills (in liked jobs):**")
            pills = " ".join(
                f'<span class="skill-pill skill-matched">{s}</span>'
                for s, _ in insights["top_matched_skills"]
            )
            st.markdown(f'<div>{pills}</div>', unsafe_allow_html=True)

        if fb_summary.get("top_liked_companies"):
            st.markdown("**Companies you liked:**")
            for co in fb_summary["top_liked_companies"]:
                st.markdown(f"- {co}")

    with col_r:
        st.markdown("### ❌ What the Model Avoids for You")

        if insights["avoided_seniority"]:
            st.markdown("**Seniority levels:**")
            for level, count in insights["avoided_seniority"]:
                st.markdown(f"- {level.title()} ({count} dislikes)")

        if insights["avoided_industry"]:
            st.markdown("**Industries:**")
            for ind, count in insights["avoided_industry"]:
                st.markdown(f"- {ind.replace('_',' ').title()} ({count} dislikes)")

        if fb_summary.get("disliked_patterns"):
            st.markdown("**Disliked patterns:**")
            for p in fb_summary["disliked_patterns"][:3]:
                st.markdown(f"- {p['seniority'].title()} {p['industry'].replace('_',' ')} roles")

    st.divider()

    # ── Current ranking weights ───────────────────────────────────────────────
    st.markdown("### ⚖️ Your Personalised Ranking Weights")
    st.caption("These weights shift based on your feedback — the model "
               "emphasises the dimensions that best predict your preferences.")

    if adaptive:
        weights = adaptive.weights
        from src.utils import DEFAULT_WEIGHTS
        w_data = []
        for k, v in weights.items():
            default = DEFAULT_WEIGHTS.get(k, 0)
            delta   = v - default
            arrow   = "⬆️" if delta > 0.005 else ("⬇️" if delta < -0.005 else "➡️")
            w_data.append({
                "Dimension":   k.replace("_", " ").title(),
                "Default":     f"{default:.2f}",
                "Your Weight": f"{v:.2f}",
                "Change":      f"{arrow} {delta:+.3f}",
            })
        st.dataframe(pd.DataFrame(w_data), hide_index=True, use_container_width=True)

    # ── Full feedback history ─────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📋 Full Feedback History")

    history = load_feedback_history(uid)
    if history:
        hist_df = pd.DataFrame([{
            "Date":      h["recorded_at"][:10],
            "Job Title": h["job_title"],
            "Company":   h["company"],
            "Feedback":  h["feedback_type"].title(),
            "Score":     f"{h['final_score']:.2f}" if h["final_score"] else "—",
        } for h in history])

        fb_filter = st.multiselect(
            "Filter by feedback type",
            ["Good", "Bad", "Save", "Skip"],
            default=["Good", "Save", "Bad", "Skip"],
        )
        filtered_hist = hist_df[hist_df["Feedback"].isin(fb_filter)]
        st.dataframe(filtered_hist, hide_index=True, use_container_width=True)

        # Export feedback history
        csv = hist_df.to_csv(index=False).encode()
        st.download_button("⬇️ Export Feedback History (CSV)",
                           csv, "feedback_history.csv", "text/csv")

    # ── Account management ────────────────────────────────────────────────────
    st.divider()
    st.markdown("### ⚙️ Account Management")
    col_save, col_del = st.columns(2)
    with col_save:
        if st.button("💾 Save All Data Now", use_container_width=True):
            _save_session_to_db()
            st.success("✅ All data saved to database.")
    with col_del:
        with st.expander("🗑️ Delete My Account"):
            st.warning("This permanently deletes your profile, feedback history, "
                       "and all learned preferences.")
            if st.button("Confirm Delete Account", type="primary"):
                delete_user(uid)
                st.session_state.current_user = None
                st.session_state.profile      = None
                st.session_state.adaptive     = None
                st.session_state.feedback     = {}
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════
page_map = {
    "🏠 Profile Setup":       page_profile,
    "🎯 Job Matches":         page_matches,
    "📄 Resume Generator":    page_resume,
    "📊 Market Analytics":    page_analytics,
    "📈 Benchmarks":          page_benchmarks,
    "🧠 My Learning Profile": page_learning_profile,
}

current_page = st.session_state.page
page_fn = page_map.get(current_page, page_profile)
page_fn()
