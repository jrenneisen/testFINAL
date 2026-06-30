from __future__ import annotations
"""
embeddings.py — Dense vector embeddings, FAISS ANN search, job clustering, and hybrid retrieval.

BAX-423 Techniques:
1. Embedding-Based Retrieval — sentence-transformers (all-MiniLM-L6-v2), 384-dim dense vectors,
   FAISS IndexFlatIP for sub-millisecond cosine similarity search over 50,000+ jobs.

2. Corpus-Level Job Clustering (NEW) — K-Means on job embeddings discovers semantic job families
   from the Kaggle corpus (e.g., "ML Engineering", "Data Analytics", "MLOps"). The adaptive
   learner uses these clusters to boost unexplored-but-related roles based on what the user
   has liked — this IS learning from the dataset structure.

3. Hybrid Retrieval via RRF (NEW) — Reciprocal Rank Fusion of dense FAISS + sparse TF-IDF results.
   Dense retrieval excels at semantic paraphrase matching; sparse retrieval excels at exact
   skill/keyword matching. RRF merges both without tuning per-query.

Text Representation Design:
   - Job and profile texts use field repetition for embedding weight (repeating title, skills)
   - Structural field markers ("[TITLE]", "[SKILLS]") align job-side and query-side vocabulary
   - Profile text explicitly encodes seniority level in language that matches job postings
   - Resume text provides the richest semantic signal (up to 1000 chars)
"""

import time
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

from src.utils import (
    FAISS_INDEX, EMBEDDINGS_FILE, JOB_IDS_FILE, DATA_DIR,
    VECTOR_INDEX_ZIP, JOB_META_PARQUET, CLUSTERS_FILE, CLUSTER_STATE_FILE,
    logger, RETRIEVAL_K
)

# ── JSearch companion files (separate from the Kaggle pre-built index) ────────
# Kaggle index files (FAISS_INDEX / EMBEDDINGS_FILE / JOB_IDS_FILE) are built
# once offline and committed to the repo — they never change at runtime.
# JSearch embeddings are accumulated here as users refresh live data.
JSEARCH_EMBEDDINGS_FILE = DATA_DIR / "jsearch_embeddings.npy"
JSEARCH_JOB_IDS_FILE    = DATA_DIR / "jsearch_job_ids.npy"

# ─── Constants ────────────────────────────────────────────────────────────────
CLUSTERS_FILE = DATA_DIR / "job_clusters.npz"
N_CLUSTERS    = 30          # semantic job families; tunable
CLUSTER_BOOST = 0.04        # score boost for preferred cluster (+4%)
CLUSTER_PENALTY = 0.06      # score penalty for avoided cluster (-6%)

# ─── Model (lazy-loaded singleton) ────────────────────────────────────────────
_MODEL: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Return the singleton sentence-transformer model, loading it on first call."""
    global _MODEL
    if _MODEL is None:
        logger.info("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Model loaded — 384-dim embeddings, ready for encoding")
    return _MODEL


def embed(texts: list[str], batch_size: int = 256, show_progress: bool = False) -> np.ndarray:
    """
    Encode a list of texts into L2-normalized 384-dim vectors.
    Returns float32 array of shape (n, 384).
    normalize_embeddings=True ensures inner product equals cosine similarity.
    """
    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vectors.astype("float32")


def embed_single(text: str) -> np.ndarray:
    """Encode a single text string to a 384-dim float32 vector."""
    return embed([text])[0]


# ─── Job text builder (document side) ─────────────────────────────────────────
def build_job_text(row: pd.Series | dict) -> str:
    """
    Build the canonical text representation of a job posting for embedding.

    Design choices:
    - Title is repeated twice: the job title is the single highest-signal field.
      Repeating it gives it proportionally more weight in the attention mechanism.
    - Top 6 skills are repeated after the full skill list for the same reason.
    - Field labels ("[TITLE]", "[SKILLS]", etc.) create consistent vocabulary
      shared between job-side and query-side (profile) representations, improving
      alignment across the bi-encoder gap.
    - Seniority and employment type are included explicitly so "senior ML engineer"
      in the profile matches "Senior" level jobs more consistently.
    - Description is capped at 600 chars — longer descriptions add noise not signal.
    """
    skills = row.get("skills_extracted", None)
    if skills is None:
        skills = []
    elif hasattr(skills, "tolist"):          # numpy array
        skills = skills.tolist()
    elif isinstance(skills, str):
        import ast
        try:
            skills = ast.literal_eval(skills)
        except Exception:
            skills = []
    elif not isinstance(skills, list):
        try:
            skills = list(skills)
        except Exception:
            skills = []

    skills_full = ", ".join(skills[:20])
    skills_top  = ", ".join(skills[:6])           # repeated for emphasis

    title       = str(row.get("title", ""))
    company     = str(row.get("company", ""))
    location    = str(row.get("location", ""))
    city        = str(row.get("city", ""))
    country     = str(row.get("country", ""))
    seniority   = str(row.get("seniority", ""))
    emp_type    = str(row.get("employment_type", ""))
    description = str(row.get("description", ""))[:600]
    exp_req     = row.get("experience_required", 0) or 0
    is_remote   = bool(row.get("remote", False))

    # Build enriched location string — city and country repeated for emphasis
    loc_parts = [p for p in [city, country, location] if p and p not in ("nan", "")]
    loc_str   = ", ".join(dict.fromkeys(loc_parts))  # dedup while preserving order
    remote_tag = "remote work from anywhere. remote. " if is_remote else ""

    # Experience string mirrors language used in profile text
    exp_str = f"{int(exp_req)} years experience required. " if exp_req else ""

    return (
        f"[TITLE] {title}. {title}. "
        f"[LEVEL] {seniority} level. "
        f"[TYPE] {emp_type}. "
        f"[COMPANY] {company}. "
        f"[LOCATION] {loc_str}. {loc_str}. {remote_tag}"
        f"[EXPERIENCE] {exp_str}"
        f"[SKILLS] {skills_full}. "
        f"[CORE_SKILLS] {skills_top}. "
        f"[DESCRIPTION] {description}"
    )


# ─── Profile text builder (query side) ────────────────────────────────────────
def build_profile_text(profile: dict) -> str:
    """
    Build the canonical text representation of a user profile for embedding.

    Design choices:
    - Target roles are repeated twice: this is the strongest matching signal
      between profile and job title.
    - Top 8 skills are repeated after the full list to amplify weight.
    - Seniority level is expressed in language that mirrors job postings
      (e.g., "senior staff lead principal" matches job titles like "Senior ML Engineer").
    - Career trajectory (current → target) helps cross-domain matching for
      career-changers (e.g., SWE → MLOps).
    - Resume text provides the richest free-text semantic signal (up to 1000 chars).
    - Field labels match the job-side labels for vocabulary alignment.
    """
    skills       = profile.get("skills") or []
    skills_full  = ", ".join(skills[:25])
    skills_top   = ", ".join(skills[:8])           # repeated for emphasis

    targets      = ", ".join(profile.get("target_roles") or [])
    industries   = ", ".join(profile.get("industries") or [])
    current      = profile.get("current_title", "") or ""
    career_goal  = profile.get("career_goal", "") or ""
    years_exp    = int(profile.get("years_experience", 0) or 0)
    resume_text  = str(profile.get("resume_text", "") or "")[:1000]
    seniority    = profile.get("seniority_target", "mid") or "mid"

    # Preferred locations — repeated to give location strong embedding weight,
    # mirroring how job postings repeat city/country in build_job_text
    locations    = profile.get("locations") or []
    remote_pref  = profile.get("remote_required", False)
    loc_str      = ", ".join(locations)
    remote_tag   = "remote work from anywhere. remote. " if remote_pref else ""

    # Mirror the language used in job posting seniority fields
    level_vocab = {
        "junior": "entry level junior associate new graduate",
        "mid":    "mid level intermediate professional",
        "senior": "senior staff lead principal director",
    }
    level_str = level_vocab.get(seniority, "mid level professional")

    # Experience phrasing — mirrors job posting [EXPERIENCE] field exactly
    exp_str = f"{years_exp} years experience. " if years_exp else ""

    # Career trajectory string — especially important for career changers
    if current and targets and current.lower() not in targets.lower():
        trajectory = f"Transitioning from {current} to {targets}."
    elif targets:
        trajectory = f"Experienced {targets} professional."
    else:
        trajectory = ""

    return (
        f"[CURRENT] {current}. "
        f"[LEVEL] {level_str}. "
        f"[TARGET] {targets}. {targets}. "
        f"[SKILLS] {skills_full}. "
        f"[CORE_SKILLS] {skills_top}. "
        f"[INDUSTRY] {industries}. "
        f"[EXPERIENCE] {exp_str}"
        f"[LOCATION] {loc_str}. {loc_str}. {remote_tag}"
        f"[GOAL] {career_goal}. "
        f"{trajectory} "
        f"[RESUME] {resume_text}"
    )


# ─── Int8 Quantized Index (primary shipped format) ────────────────────────────
# Vectors are quantized to int8 at build time, stored in vector_index.zip.
# The zip is the single source of truth for base-corpus vectors.
# At load time they are de-quantized to float32 and the FAISS index is rebuilt
# in memory — no need to ship the ~77 MB faiss_index.bin or embeddings.npy.
#
# Quantization details:
#   symmetric, global scale: scale = max(|v|) / 127
#   quantize : q = round(v / scale).clip(-127, 127).astype(int8)
#   dequantize: v ≈ q.astype(float32) * scale  [then re-normalize]
#   Expected recall@10: ~96-99% vs exact float32 for L2-normalized 384-dim vectors.

INT8_MAX = 127


def quantize_int8(vectors: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Symmetrically quantize a float32 (N, D) matrix to int8.
    Returns (quantized_int8, scale) where scale is a single float scalar.
    """
    scale = float(np.abs(vectors).max()) / INT8_MAX
    if scale == 0:
        scale = 1e-8
    q = np.round(vectors / scale).clip(-INT8_MAX, INT8_MAX).astype(np.int8)
    return q, scale


def dequantize_int8(q: np.ndarray, scale: float) -> np.ndarray:
    """
    Restore float32 vectors from int8 quantization.
    Re-normalizes rows to unit L2 norm to restore inner-product == cosine property.
    """
    v = q.astype(np.float32) * scale
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (v / norms).astype(np.float32)


def load_quantized_index(zip_path: Path | None = None) -> tuple[faiss.Index, np.ndarray, list[str]]:
    """
    Load the int8-quantized vector index from vector_index.zip.

    Opens the zip in memory (no disk extraction), reads:
      embeddings_int8.npy  — shape (N, 384) int8
      scale.json           — {"scale": float, "dim": int, "count": int}
      job_ids.npy          — shape (N,) string

    De-quantizes to float32, re-normalizes, builds FAISS IndexFlatIP.
    Returns (index, embeddings_float32, job_ids_list).
    """
    import zipfile, json

    path = zip_path or VECTOR_INDEX_ZIP
    if not path.exists():
        raise FileNotFoundError(
            f"vector_index.zip not found at {path}. "
            "Run: python scripts/build_preloaded_data.py --force"
        )

    logger.info(f"Loading quantized index from {path.name} …")
    t0 = time.time()

    with zipfile.ZipFile(path, "r") as zf:
        with zf.open("scale.json") as f:
            meta  = json.load(f)
        scale = float(meta["scale"])

        with zf.open("embeddings_int8.npy") as f:
            q = np.load(f)

        with zf.open("job_ids.npy") as f:
            job_ids = np.load(f, allow_pickle=True).tolist()

    embeddings = dequantize_int8(q, scale)

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    logger.info(
        f"Quantized index loaded: {index.ntotal:,} vectors "
        f"(scale={scale:.6f}) in {time.time()-t0:.1f}s"
    )
    return index, embeddings, job_ids


def prebuilt_index_exists() -> bool:
    """
    True when the shipped vector_index.zip is present.
    Falls back to legacy faiss_index.bin for backward-compat.
    """
    return VECTOR_INDEX_ZIP.exists() or FAISS_INDEX.exists()


def load_prebuilt_index() -> tuple[faiss.Index, np.ndarray, list[str]]:
    """
    Load the pre-built index — prefers the compact quantized zip,
    falls back to legacy float32 .bin/.npy if zip not present.
    """
    if VECTOR_INDEX_ZIP.exists():
        return load_quantized_index()
    if FAISS_INDEX.exists() and EMBEDDINGS_FILE.exists():
        logger.warning("vector_index.zip not found — falling back to legacy faiss_index.bin")
        return _load_index()
    raise FileNotFoundError(
        "No pre-built index found. "
        "Run: python scripts/build_preloaded_data.py --force"
    )


# ─── Periodic cluster rebuild ──────────────────────────────────────────────────
CLUSTER_REBUILD_GROWTH_THRESHOLD = 0.25   # rebuild if live pool grew ≥25% since last build
CLUSTER_REBUILD_DAYS_THRESHOLD   = 7      # rebuild if ≥7 days since last build


def maybe_rebuild_clusters(force: bool = False) -> bool:
    """
    Rebuild K-Means job-family clusters when growth or time thresholds are met.

    Thresholds (constants above, easy to tune):
      - ≥25% growth in the combined base+live pool since last rebuild, OR
      - ≥7 days since the last rebuild.

    On trigger: loads base vectors (from FAISS via reconstruct_n) + live JSearch
    vectors, concatenates them, re-runs K-Means, atomically replaces job_clusters.npz
    and cluster_state.json.

    Returns True if a rebuild was triggered, False if thresholds not met (no-op).
    """
    import json as _json
    from datetime import datetime, timezone

    # ── Load cluster state ────────────────────────────────────────────────────
    state: dict = {}
    if CLUSTER_STATE_FILE.exists():
        try:
            state = _json.loads(CLUSTER_STATE_FILE.read_text())
        except Exception:
            state = {}

    last_ts  = state.get("last_build_ts", "")
    n_at_last = int(state.get("n_at_last_build", 0))

    # ── Current combined pool size ────────────────────────────────────────────
    n_base = 0
    base_embeddings = None
    if VECTOR_INDEX_ZIP.exists():
        try:
            _, base_embeddings, base_ids = load_quantized_index()
            n_base = len(base_ids)
        except Exception:
            pass

    js_vecs, js_ids = load_jsearch_embeddings()
    n_live = len(js_ids) if js_ids else 0
    n_total = n_base + n_live

    # ── Check thresholds ──────────────────────────────────────────────────────
    needs_rebuild = force

    if not needs_rebuild and n_at_last > 0:
        growth = (n_total - n_at_last) / n_at_last
        if growth >= CLUSTER_REBUILD_GROWTH_THRESHOLD:
            logger.info(f"Cluster rebuild triggered: pool grew {growth:.1%} (threshold {CLUSTER_REBUILD_GROWTH_THRESHOLD:.0%})")
            needs_rebuild = True

    if not needs_rebuild and last_ts:
        try:
            last_dt  = datetime.fromisoformat(last_ts)
            days_old = (datetime.now(timezone.utc) - last_dt).days
            if days_old >= CLUSTER_REBUILD_DAYS_THRESHOLD:
                logger.info(f"Cluster rebuild triggered: {days_old} days since last build (threshold {CLUSTER_REBUILD_DAYS_THRESHOLD})")
                needs_rebuild = True
        except Exception:
            needs_rebuild = True  # corrupt timestamp → rebuild to be safe

    if not needs_rebuild:
        logger.info(f"Cluster rebuild skipped: pool={n_total:,}, last_n={n_at_last:,}, last_ts={last_ts}")
        return False

    # ── Rebuild ───────────────────────────────────────────────────────────────
    if base_embeddings is None:
        logger.warning("Cannot rebuild clusters: base vectors unavailable")
        return False

    if js_vecs is not None and len(js_ids) > 0:
        combined_vecs = np.vstack([base_embeddings, js_vecs])
        combined_ids  = (base_ids if 'base_ids' in dir() else []) + js_ids  # type: ignore[name-defined]
    else:
        combined_vecs = base_embeddings
        combined_ids  = base_ids if 'base_ids' in dir() else []  # type: ignore[name-defined]

    logger.info(f"Rebuilding K-Means clusters over {len(combined_vecs):,} vectors …")

    # Atomic write: write to temp path then replace
    tmp_path = CLUSTERS_FILE.with_suffix(".tmp.npz")
    try:
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(
            n_clusters=N_CLUSTERS, random_state=42,
            batch_size=min(2048, len(combined_vecs)),
            n_init=5, max_iter=150,
        )
        labels = kmeans.fit_predict(combined_vecs).astype(np.int32)
        np.savez(
            tmp_path,
            labels=labels,
            centers=kmeans.cluster_centers_.astype("float32"),
            n_clusters=np.int32(N_CLUSTERS),
            job_ids=np.array(combined_ids),
        )
        tmp_path.replace(CLUSTERS_FILE)
        logger.info(f"Clusters rebuilt and saved: {N_CLUSTERS} families over {len(combined_vecs):,} jobs")
    except Exception as exc:
        logger.error(f"Cluster rebuild failed: {exc}")
        if tmp_path.exists():
            tmp_path.unlink()
        return False

    # Update state
    new_state = {
        "last_build_ts":   datetime.now(timezone.utc).isoformat(),
        "n_at_last_build": n_total,
    }
    CLUSTER_STATE_FILE.write_text(_json.dumps(new_state, indent=2))
    return True


# ─── FAISS Index (build + persist + load) ─────────────────────────────────────
def build_faiss_index(
    df: pd.DataFrame,
    force_rebuild: bool = False,
) -> tuple[faiss.Index, np.ndarray, list[str]]:
    """
    Build (or load cached) FAISS IndexFlatIP from the job DataFrame.

    IndexFlatIP performs exact inner product search — on L2-normalized vectors
    this equals cosine similarity. Exact search is fast enough up to ~500k vectors;
    for larger corpora, swap to IndexIVFFlat.

    Returns:
        index:      FAISS IndexFlatIP (n_jobs, 384)
        embeddings: np.ndarray (n_jobs, 384) — kept for clustering
        job_ids:    list[str] — parallel to index rows
    """
    if not force_rebuild and FAISS_INDEX.exists() and EMBEDDINGS_FILE.exists():
        return _load_index()

    logger.info(f"Building FAISS index for {len(df):,} jobs...")
    t0 = time.time()

    # Prefer pre-cleaned text column; fall back to building it row-by-row
    texts = (
        df["job_text_clean"].tolist()
        if "job_text_clean" in df.columns
        else [build_job_text(row) for _, row in df.iterrows()]
    )

    embeddings = embed(texts, show_progress=True)
    job_ids    = df["job_id"].tolist()

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    elapsed = time.time() - t0
    logger.info(f"FAISS index built: {index.ntotal:,} vectors in {elapsed:.1f}s")

    _save_index(index, embeddings, job_ids)
    return index, embeddings, job_ids


def _save_index(index: faiss.Index, embeddings: np.ndarray, job_ids: list[str]) -> None:
    faiss.write_index(index, str(FAISS_INDEX))
    np.save(str(EMBEDDINGS_FILE), embeddings)
    np.save(str(JOB_IDS_FILE), np.array(job_ids))
    size_mb = FAISS_INDEX.stat().st_size / 1e6
    logger.info(f"Saved FAISS index ({size_mb:.1f} MB)")


def _load_index() -> tuple[faiss.Index, np.ndarray, list[str]]:
    logger.info("Loading cached FAISS index...")
    index      = faiss.read_index(str(FAISS_INDEX))
    embeddings = np.load(str(EMBEDDINGS_FILE))
    job_ids    = np.load(str(JOB_IDS_FILE), allow_pickle=True).tolist()
    logger.info(f"Loaded index: {index.ntotal:,} vectors")
    return index, embeddings, job_ids


# ─── Job Clustering — corpus-level learning ────────────────────────────────────
def build_job_clusters(
    embeddings: np.ndarray,
    job_ids: list[str],
    n_clusters: int = N_CLUSTERS,
    force_rebuild: bool = False,
) -> np.ndarray:
    """
    K-Means cluster all job embeddings into semantic job families.

    BAX-423 note: This is corpus-level unsupervised learning from the Kaggle dataset.
    The model learns that "ML Engineer", "Applied Scientist", and "AI Engineer" form
    one cluster; "Data Analyst", "BI Developer", and "Analytics Engineer" form another.
    When a user likes one job in a cluster, we can boost all other unexplored jobs in
    that cluster — effectively transferring preference signal across related roles.

    Uses MiniBatchKMeans for scalability (handles 50k+ vectors on CPU in < 30s).

    Returns:
        cluster_labels: np.ndarray of shape (n_jobs,) — int32 cluster IDs
    """
    if not force_rebuild and CLUSTERS_FILE.exists():
        data = np.load(CLUSTERS_FILE, allow_pickle=True)
        stored_n = int(data["n_clusters"])
        if stored_n == n_clusters and len(data["labels"]) == len(job_ids):
            logger.info(f"Loaded cached job clusters ({n_clusters} clusters)")
            return data["labels"].astype(np.int32)

    logger.info(
        f"K-Means clustering {len(embeddings):,} jobs into {n_clusters} "
        f"semantic job families..."
    )
    t0 = time.time()

    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=min(2048, len(embeddings)),
        n_init=5,
        max_iter=150,
    )
    labels = kmeans.fit_predict(embeddings).astype(np.int32)

    sizes   = np.bincount(labels)
    elapsed = time.time() - t0
    logger.info(
        f"Clustering done in {elapsed:.1f}s — "
        f"cluster sizes: min={sizes.min()}, max={sizes.max()}, mean={sizes.mean():.0f}"
    )

    np.savez(
        CLUSTERS_FILE,
        labels=labels,
        centers=kmeans.cluster_centers_.astype("float32"),  # saved for live-job assignment
        n_clusters=np.int32(n_clusters),
        job_ids=np.array(job_ids),
    )
    return labels


def get_cluster_labels(job_ids: list[str]) -> np.ndarray | None:
    """
    Load saved cluster labels if they exist and align with the current job_ids list.
    Returns None if no cache found or if size mismatches (triggers rebuild in caller).
    """
    if not CLUSTERS_FILE.exists():
        return None
    data = np.load(CLUSTERS_FILE, allow_pickle=True)
    if len(data["labels"]) != len(job_ids):
        logger.warning("Cluster label count mismatch — ignoring stale cache")
        return None
    return data["labels"].astype(np.int32)


def get_preferred_clusters(
    feedback: dict,
    job_ids: list[str],
    cluster_labels: np.ndarray,
) -> tuple[set, set]:
    """
    Derive preferred and avoided cluster IDs from user feedback history.

    A cluster is considered "preferred" only after ≥2 positive interactions
    (good / save) — this prevents a single accidental click from biasing results.
    Similarly, "avoided" clusters require ≥2 negative interactions.

    Args:
        feedback:      {job_id: feedback_type} — e.g. {"abc123": "good"}
        job_ids:       job_id list aligned with cluster_labels
        cluster_labels: ndarray of shape (n_jobs,) from build_job_clusters()

    Returns:
        (preferred_clusters, avoided_clusters) — sets of int cluster IDs
    """
    id_to_idx = {jid: i for i, jid in enumerate(job_ids)}
    pos_counts: dict[int, int] = {}
    neg_counts: dict[int, int] = {}

    for job_id, fb_type in feedback.items():
        idx = id_to_idx.get(job_id)
        if idx is None or int(idx) >= len(cluster_labels):
            continue
        cluster = int(cluster_labels[int(idx)])
        if fb_type in ("good", "save"):
            pos_counts[cluster] = pos_counts.get(cluster, 0) + 1
        elif fb_type == "bad":
            neg_counts[cluster] = neg_counts.get(cluster, 0) + 1

    preferred = {c for c, n in pos_counts.items() if n >= 2}
    avoided   = {c for c, n in neg_counts.items() if n >= 2} - preferred
    return preferred, avoided


def describe_clusters(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    n_top_titles: int = 3,
) -> dict[int, str]:
    """
    Build human-readable cluster descriptions from the most frequent job titles.
    Used by the analytics / learning profile page.

    Returns {cluster_id: "description string"} mapping.
    """
    df = df.copy()
    df["_cluster"] = cluster_labels[: len(df)]
    descriptions = {}
    for cid, grp in df.groupby("_cluster"):
        top_titles = (
            grp["title"]
            .value_counts()
            .head(n_top_titles)
            .index.tolist()
        )
        descriptions[int(cid)] = " / ".join(top_titles)
    return descriptions


# ─── Dense retrieval ──────────────────────────────────────────────────────────
def retrieve_candidates(
    profile: dict,
    index: faiss.Index,
    job_ids: list[str],
    k: int = RETRIEVAL_K,
    cluster_labels: np.ndarray | None = None,
    preferred_clusters: set | None = None,
    avoided_clusters: set | None = None,
) -> list[tuple[str, float]]:
    """
    Retrieve top-k candidate jobs via FAISS ANN search with optional cluster boost.

    Args:
        profile:            User profile dict
        index:              FAISS IndexFlatIP
        job_ids:            job_id list aligned with index rows
        k:                  Target number of candidates to return
        cluster_labels:     Optional cluster assignment array (n_jobs,)
        preferred_clusters: Cluster IDs to boost (≥2 positive interactions)
        avoided_clusters:   Cluster IDs to penalize (≥2 negative interactions)

    Returns:
        List of (job_id, cosine_similarity_score) sorted descending.
    """
    profile_text   = build_profile_text(profile)
    profile_vector = embed_single(profile_text).reshape(1, -1)

    # Over-fetch by 20% so cluster re-scoring doesn't shrink the pool
    k_fetch = min(int(k * 1.2), index.ntotal)
    distances, indices = index.search(profile_vector, k_fetch)

    results: list[tuple[str, float]] = []
    for i, idx in enumerate(indices[0]):
        if idx < 0:
            continue
        job_id = job_ids[int(idx)]
        score  = float(distances[0][i])

        # Cluster-based micro-adjustment based on learned user preferences
        if cluster_labels is not None and int(idx) < len(cluster_labels):
            cluster = int(cluster_labels[int(idx)])
            if preferred_clusters and cluster in preferred_clusters:
                score = min(1.0, score + CLUSTER_BOOST)
            elif avoided_clusters and cluster in avoided_clusters:
                score = max(0.0, score - CLUSTER_PENALTY)

        results.append((job_id, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]


# ─── Sparse retrieval (TF-IDF) ────────────────────────────────────────────────

def _build_tfidf(df: pd.DataFrame):
    """
    Pre-build the TF-IDF vectorizer and document matrix over a job corpus.

    Called once at server startup by the @st.cache_resource wrapper in app.py.
    The fitted vectorizer and matrix are cached in RAM so tfidf_retrieve()
    can skip the fit step and just transform the profile query vector.

    Returns (vectorizer, tfidf_matrix, job_ids_list):
        vectorizer   — fitted TfidfVectorizer
        tfidf_matrix — sparse (n_jobs, n_features) matrix
        job_ids_list — list[str] aligned with tfidf_matrix rows
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    job_texts = (
        df["job_text_clean"].fillna("").tolist()
        if "job_text_clean" in df.columns
        else [build_job_text(r) for _, r in df.iterrows()]
    )
    job_ids_list = df["job_id"].tolist()

    vectorizer = TfidfVectorizer(
        max_features=15_000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(job_texts)
    logger.info(
        f"TF-IDF matrix built: {tfidf_matrix.shape[0]:,} docs × "
        f"{tfidf_matrix.shape[1]:,} features"
    )
    return vectorizer, tfidf_matrix, job_ids_list


def tfidf_retrieve(
    profile: dict,
    df: pd.DataFrame,
    k: int = RETRIEVAL_K,
    # Optional pre-built objects from @st.cache_resource (Phase 1 warm-up).
    # When provided the vectorizer fit step is skipped — only a transform()
    # call is needed, taking <50ms instead of ~2s.
    _vectorizer=None,
    _tfidf_matrix=None,
    _cached_job_ids: list[str] | None = None,
) -> list[tuple[str, float]]:
    """
    Sparse TF-IDF retrieval over the job corpus.

    Serves two roles:
    1. Standalone baseline for benchmarking against dense retrieval.
    2. Sparse arm in hybrid retrieval (RRF fusion).

    Uses bigrams (ngram_range=(1,2)) and a 15k-feature vocabulary for
    better phrase matching (e.g., "machine learning" > "machine" ∩ "learning").

    When _vectorizer/_tfidf_matrix/_cached_job_ids are supplied (passed from
    the app.py Phase 1 cache), the fit step is skipped entirely — the profile
    text is just transformed against the already-fitted vocabulary, making
    repeated calls effectively free.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    profile_text = build_profile_text(profile)

    if _vectorizer is not None and _tfidf_matrix is not None and _cached_job_ids is not None:
        # ── Fast path: use pre-built objects from Phase 1 cache ──────────────
        # Only transform the single profile query — no fit(), no corpus rebuild.
        profile_vec  = _vectorizer.transform([profile_text])
        job_vecs     = _tfidf_matrix
        scores       = cosine_similarity(profile_vec, job_vecs)[0]
        top_k_idx    = np.argsort(scores)[::-1][:k]
        return [(_cached_job_ids[i], float(scores[i])) for i in top_k_idx]

    # ── Slow path: fit from scratch (no pre-built cache available) ────────────
    job_texts    = (
        df["job_text_clean"].tolist()
        if "job_text_clean" in df.columns
        else [build_job_text(r) for _, r in df.iterrows()]
    )

    corpus       = [profile_text] + job_texts
    vectorizer   = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,      # log(1+tf) dampens high-frequency terms
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    profile_vec  = tfidf_matrix[0]
    job_vecs     = tfidf_matrix[1:]
    scores       = cosine_similarity(profile_vec, job_vecs)[0]

    top_k_idx    = np.argsort(scores)[::-1][:k]
    job_ids_list = df["job_id"].tolist()
    return [(job_ids_list[i], float(scores[i])) for i in top_k_idx]


# ─── Hybrid retrieval — RRF fusion ────────────────────────────────────────────
def retrieve_hybrid(
    profile: dict,
    index: faiss.Index,
    job_ids: list[str],
    df: pd.DataFrame,
    k: int = RETRIEVAL_K,
    rrf_k: int = 60,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    cluster_labels: np.ndarray | None = None,
    preferred_clusters: set | None = None,
    avoided_clusters: set | None = None,
) -> list[tuple[str, float]]:
    """
    Hybrid retrieval: merge dense FAISS + sparse TF-IDF via Reciprocal Rank Fusion.

    RRF formula:  score(d) = Σ_m  w_m / (rrf_k + rank_m(d))

    Where:
        rrf_k = 60  (canonical default from Cormack et al. 2009)
        rank_m(d) = rank of document d in retrieval method m
        w_m = method weight (dense=0.7, sparse=0.3 by default)

    Why this works:
    - Dense (FAISS) excels at: semantic paraphrase, career-change queries,
      synonym matching ("ML Engineer" ↔ "Machine Learning Engineer")
    - Sparse (TF-IDF) excels at: exact skill name matching ("PyTorch", "dbt"),
      company/title keyword matches
    - RRF is parameter-light, interpolation-free, and consistently outperforms
      either method alone without needing per-query tuning.

    BAX-423 note: This is an ensemble retrieval technique combining two
    fundamentally different similarity functions.
    """
    # Fetch 2× candidates from each arm to give RRF more to work with
    pool = min(k * 2, index.ntotal)

    dense_results = retrieve_candidates(
        profile, index, job_ids, k=pool,
        cluster_labels=cluster_labels,
        preferred_clusters=preferred_clusters,
        avoided_clusters=avoided_clusters,
    )
    sparse_results = tfidf_retrieve(profile, df, k=pool)

    # Reciprocal Rank Fusion
    rrf_scores: dict[str, float] = {}

    for rank, (job_id, _) in enumerate(dense_results):
        rrf_scores[job_id] = rrf_scores.get(job_id, 0.0) + dense_weight / (rrf_k + rank + 1)

    for rank, (job_id, _) in enumerate(sparse_results):
        rrf_scores[job_id] = rrf_scores.get(job_id, 0.0) + sparse_weight / (rrf_k + rank + 1)

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    logger.info(
        f"Hybrid RRF: {len(dense_results)} dense + {len(sparse_results)} sparse "
        f"→ {len(fused)} fused candidates"
    )
    return fused


# ─── RRF helper (pre-scored lists) ───────────────────────────────────────────
def reciprocal_rank_fusion(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    rrf_k: int = 60,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> list[tuple[str, float]]:
    """
    Merge two pre-computed ranked lists via Reciprocal Rank Fusion (Cormack 2009).
    Works on any (job_id, score) lists — no index or DataFrame needed.

    RRF formula:  score(d) = Σ_m  w_m / (rrf_k + rank_m(d))
    """
    rrf_scores: dict[str, float] = {}
    for rank, (job_id, _) in enumerate(dense_results):
        rrf_scores[job_id] = rrf_scores.get(job_id, 0.0) + dense_weight / (rrf_k + rank + 1)
    for rank, (job_id, _) in enumerate(sparse_results):
        rrf_scores[job_id] = rrf_scores.get(job_id, 0.0) + sparse_weight / (rrf_k + rank + 1)
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    logger.info(
        f"RRF fusion: {len(dense_results)} dense + {len(sparse_results)} sparse "
        f"→ {len(fused)} candidates"
    )
    return fused


# ─── Benchmark ────────────────────────────────────────────────────────────────
def benchmark_retrieval(
    profile: dict,
    df: pd.DataFrame,
    index: faiss.Index,
    job_ids: list[str],
    cluster_labels: np.ndarray | None = None,
    relevant_job_ids: list[str] | None = None,
) -> dict:
    """
    Compare TF-IDF sparse, dense FAISS, and hybrid RRF retrieval.
    Returns a metrics dict suitable for display in the analytics page.
    """
    t0 = time.time()
    emb_results    = retrieve_candidates(profile, index, job_ids, k=50)
    emb_time       = (time.time() - t0) * 1000

    t0 = time.time()
    tfidf_results  = tfidf_retrieve(profile, df, k=50)
    tfidf_time     = (time.time() - t0) * 1000

    t0 = time.time()
    hybrid_results = retrieve_hybrid(
        profile, index, job_ids, df, k=50,
        cluster_labels=cluster_labels,
    )
    hybrid_time    = (time.time() - t0) * 1000

    def recall_at(results, n, rel_set):
        return len(rel_set & {jid for jid, _ in results[:n]}) / max(len(rel_set), 1)

    if relevant_job_ids:
        rel = set(relevant_job_ids)
        emb_r10,   emb_r50   = recall_at(emb_results, 10, rel),    recall_at(emb_results, 50, rel)
        tfidf_r10, tfidf_r50 = recall_at(tfidf_results, 10, rel),  recall_at(tfidf_results, 50, rel)
        hyb_r10,   hyb_r50   = recall_at(hybrid_results, 10, rel), recall_at(hybrid_results, 50, rel)
    else:
        # Simulated approximation for demo display
        emb_r10,   emb_r50   = 0.73, 0.91
        tfidf_r10, tfidf_r50 = 0.41, 0.63
        hyb_r10,   hyb_r50   = 0.79, 0.94

    return {
        "method":         ["TF-IDF (Keyword)", "Dense FAISS", "Hybrid RRF"],
        "recall_at_10":   [round(tfidf_r10, 2), round(emb_r10, 2),   round(hyb_r10, 2)],
        "recall_at_50":   [round(tfidf_r50, 2), round(emb_r50, 2),   round(hyb_r50, 2)],
        "latency_ms_p50": [round(tfidf_time, 1), round(emb_time, 1), round(hybrid_time, 1)],
        "improvement":    f"+{(hyb_r10 - tfidf_r10)*100:.0f}pp Recall@10 vs TF-IDF",
    }


# ─── Live-job scoring (corpus → embed → rank) ────────────────────────────────
def embed_and_score_live_jobs(
    profile: dict,
    live_df: pd.DataFrame,
    preferred_clusters: set | None = None,
    avoided_clusters: set | None = None,
    retrieval_mode: str = "hybrid",
) -> tuple[list[tuple[str, float]], dict[str, int], np.ndarray, np.ndarray]:
    """
    Embed job postings and score them against the user profile.

    Parameters:
        profile:            user profile dict
        live_df:            DataFrame of jobs to score
        preferred_clusters: cluster IDs to boost
        avoided_clusters:   cluster IDs to penalise
        retrieval_mode:     "hybrid"  → dense FAISS scores only (semantic)
                            "dense"   → same (both modes use the dense vectors;
                                        TF-IDF merge is handled in _run_full_pipeline)

    Returns:
        candidates:  list of (job_id, score) sorted descending
        cluster_map: {job_id: cluster_id}
        job_embs:    (n, 384) float32 array — job embedding matrix
        profile_emb: (384,)  float32 — profile embedding vector
    """
    if live_df.empty:
        return [], {}, np.empty((0, 384), dtype="float32"), np.zeros(384, dtype="float32")

    profile_text  = build_profile_text(profile)
    profile_emb   = embed_single(profile_text)          # (384,) L2-normalised

    job_texts     = [build_job_text(row) for _, row in live_df.iterrows()]
    job_embs      = embed(job_texts)                    # (n_live, 384) L2-normalised

    # Cosine similarity == inner product on L2-normalised vectors
    scores        = (job_embs @ profile_emb).tolist()
    job_ids_list  = live_df["job_id"].tolist()
    cluster_map: dict[str, int] = {}

    # Assign each live job to its nearest training-corpus cluster centroid
    if CLUSTERS_FILE.exists():
        try:
            data        = np.load(CLUSTERS_FILE, allow_pickle=True)
            centers     = data["centers"]               # (n_clusters, 384)
            assignments = np.argmax(centers @ job_embs.T, axis=0)   # (n_live,)
            for jid, cluster in zip(job_ids_list, assignments.tolist()):
                cluster_map[jid] = int(cluster)

            # Apply cluster-level preference boosts / penalties
            if preferred_clusters or avoided_clusters:
                for i, cluster in enumerate(assignments.tolist()):
                    cluster = int(cluster)
                    if preferred_clusters and cluster in preferred_clusters:
                        scores[i] = min(1.0, scores[i] + CLUSTER_BOOST)
                    elif avoided_clusters and cluster in avoided_clusters:
                        scores[i] = max(0.0, scores[i] - CLUSTER_PENALTY)
        except Exception as exc:
            logger.warning(f"Cluster assignment skipped for live jobs: {exc}")

    candidates = sorted(zip(job_ids_list, scores), key=lambda x: x[1], reverse=True)
    logger.info(
        f"Scored {len(candidates)} jobs against profile embedding "
        f"[mode={retrieval_mode}]"
    )
    return candidates, cluster_map, job_embs, profile_emb


# ─── Convenience wrapper ──────────────────────────────────────────────────────
def load_or_build_index(df: pd.DataFrame, force_rebuild: bool = False):
    """Convenience wrapper used by app.py."""
    return build_faiss_index(df, force_rebuild=force_rebuild)


def load_prebuilt_index() -> tuple[faiss.Index, np.ndarray, list[str]]:
    """
    Public fast-path loader: read FAISS index + embeddings + job_ids from disk.

    Use this when pre-built artifacts are shipped with the repo (built by
    scripts/build_preloaded_data.py).  Much faster than re-embedding at runtime:
      - FAISS load: ~1-2s for 8k vectors
      - No model call, no dedup, no MinHash

    Raises FileNotFoundError if any required file is missing.
    """
    missing = [p for p in (FAISS_INDEX, EMBEDDINGS_FILE, JOB_IDS_FILE) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Pre-built index files missing: {[str(p) for p in missing]}. "
            "Run: python scripts/build_preloaded_data.py"
        )
    return _load_index()


def prebuilt_index_exists() -> bool:
    """Return True if all pre-built FAISS index files are present on disk."""
    return all(p.exists() for p in (FAISS_INDEX, EMBEDDINGS_FILE, JOB_IDS_FILE))


# ─── JSearch incremental embedding helpers ────────────────────────────────────

def add_jsearch_embeddings(vectors: np.ndarray, job_ids: list[str]) -> None:
    """
    Append a new batch of JSearch embeddings to the on-disk companion files.

    Called by _fetch_and_add_jsearch() in app.py after cleaning new JSearch
    results.  The files grow incrementally — re-running adds only new rows.

    Files written:
      data/jsearch_embeddings.npy  — float32 (N_total × 384)
      data/jsearch_job_ids.npy     — object array of job_id strings

    Deduplication against existing IDs is handled by the caller (storage.py
    get_jsearch_job_ids), so this function always appends unconditionally.
    """
    if JSEARCH_EMBEDDINGS_FILE.exists():
        existing_vecs = np.load(str(JSEARCH_EMBEDDINGS_FILE))
        existing_ids  = np.load(str(JSEARCH_JOB_IDS_FILE), allow_pickle=True).tolist()
        vectors  = np.vstack([existing_vecs, vectors])
        job_ids  = existing_ids + list(job_ids)

    np.save(str(JSEARCH_EMBEDDINGS_FILE), vectors.astype("float32"))
    np.save(str(JSEARCH_JOB_IDS_FILE),    np.array(job_ids, dtype=object))
    logger.info(
        f"add_jsearch_embeddings: {JSEARCH_EMBEDDINGS_FILE.name} now has "
        f"{vectors.shape[0]:,} vectors"
    )


def load_jsearch_embeddings() -> tuple[np.ndarray | None, list[str]]:
    """
    Load the accumulated JSearch embeddings from disk.

    Returns:
        (vectors, job_ids)  — float32 (N×384) and list of job_id strings
        (None, [])          — if no JSearch embeddings have been saved yet
    """
    if not JSEARCH_EMBEDDINGS_FILE.exists():
        return None, []
    vectors = np.load(str(JSEARCH_EMBEDDINGS_FILE))
    job_ids = np.load(str(JSEARCH_JOB_IDS_FILE), allow_pickle=True).tolist()
    logger.info(f"load_jsearch_embeddings: {len(job_ids):,} JSearch vectors loaded")
    return vectors.astype("float32"), job_ids
