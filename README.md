# JobPilot 🚀

**AI-powered job matching for international students**  
BAX-423 Big Data Final Project · UC Davis MSBA · Spring 2026

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://jobpilot.streamlit.app)

---

## What It Does

JobPilot matches job seekers to roles using a two-phase ML pipeline:

- **Phase A — Corpus Intelligence:** A 35,000-job Kaggle corpus is cleaned, deduplicated with MinHash LSH, embedded with `sentence-transformers`, and clustered with K-Means into job families.
- **Phase B — Personalized Ranking:** Jobs are scored across seven weighted dimensions (embedding similarity, skill match, title match, location fit, experience match, salary fit, recency) and ranked for the user's profile.
- **Adaptive Learning:** A Thompson Sampling bandit adjusts scoring weights based on thumbs-up / thumbs-down feedback across sessions.
- **Resume Generation:** Inference-based rewriting — rewrites existing bullets using language from the job description. No credentials invented.

---

## Quick Start (Local)

```bash
git clone https://github.com/<your-handle>/jobpilot.git
cd jobpilot
pip install -r requirements.txt
streamlit run app.py
```

The pre-loaded Kaggle corpus (`data/preloaded_kaggle_50k.parquet`) is included in the repo. The app detects it on startup — no API key needed to run job matches immediately.

To enable live JSearch job search, add your key to `.streamlit/secrets.toml`:

```toml
RAPIDAPI_KEY = "your_key_here"
OPENAI_API_KEY = "your_key_here"   # optional — enables resume rewriting
```

---

## Project Structure

```
jobpilot/
├── app.py                          # Main Streamlit application
├── requirements.txt
├── data/
│   └── preloaded_kaggle_50k.parquet  # 35k pre-cleaned, pre-deduped job corpus
├── src/
│   ├── ingest.py                   # JSearch API + fallback query broadening
│   ├── clean.py                    # Text normalization, feature extraction
│   ├── dedupe.py                   # Level-1 exact + Level-2 MinHash LSH dedup
│   ├── embeddings.py               # FAISS index, K-Means clustering, hybrid retrieval
│   ├── ranker.py                   # 7-dimension weighted scorer + Thompson Sampling
│   ├── resume_generator.py         # LLM-based inference resume rewriting
│   ├── storage.py                  # SQLite persistence (profiles, feedback, saved jobs)
│   └── utils.py                    # Shared config, paths, logger
├── scripts/
│   └── build_preloaded_data.py     # Offline corpus builder (run once, not at app startup)
└── outputs/
    └── kaggle_build.py             # Kaggle notebook script that produced the parquet
```

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│  PHASE A — Corpus Intelligence (runs once at startup)   │
│                                                         │
│  Kaggle 35k corpus                                      │
│       │                                                 │
│       ▼                                                 │
│  clean.py ──► dedupe.py (L1 exact + L2 MinHash LSH)    │
│       │                                                 │
│       ▼                                                 │
│  embeddings.py                                          │
│    sentence-transformers/all-MiniLM-L6-v2 (384-dim)    │
│    FAISS IndexFlatIP                                    │
│    K-Means job-family clusters                          │
└─────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────┐
│  PHASE B — Personalized Ranking (per user query)        │
│                                                         │
│  User Profile (skills, experience, location, salary)   │
│       │                                                 │
│       ▼                                                 │
│  Hybrid Retrieval: FAISS ANN + TF-IDF keyword          │
│       │                                                 │
│       ▼                                                 │
│  ranker.py — 7-dimension weighted score                 │
│    embedding_similarity  0.27                           │
│    skill_match           0.25                           │
│    location_fit          0.18                           │
│    salary_fit            0.10                           │
│    experience_match      0.08                           │
│    title_match           0.07                           │
│    recency               0.05                           │
│       │                                                 │
│       ▼                                                 │
│  Thompson Sampling bandit → weight adaptation           │
│       │                                                 │
│       ▼                                                 │
│  Ranked job cards + Resume rewriter                     │
└─────────────────────────────────────────────────────────┘
```

---

## Key Techniques (BAX-423)

| Technique | Where Used | Why |
|---|---|---|
| **MinHash LSH** (Locality-Sensitive Hashing) | `dedupe.py` | Near-duplicate detection in O(n) instead of O(n²) |
| **FAISS IndexFlatIP** | `embeddings.py` | Sub-millisecond approximate nearest-neighbor search over 384-dim embeddings |
| **K-Means Clustering** | `embeddings.py` | Groups jobs into families for cluster-boosted retrieval |
| **Sentence Transformers** | `embeddings.py` | Semantic embedding with `all-MiniLM-L6-v2` |
| **Thompson Sampling** | `ranker.py` | Multi-armed bandit — adapts scoring weights from user feedback |
| **TF-IDF Hybrid Retrieval** | `embeddings.py` | Keyword recall fallback alongside dense ANN retrieval |
| **SQLite Persistence** | `storage.py` | Cross-session profile, feedback, and saved job storage |

---

## Deduplication Pipeline

The corpus goes through two levels before any embedding is computed:

**Level 1 — Exact dedup:** Hash of `title + company + description[:300]`. Location is excluded intentionally — the same job posted in NYC and LA are kept as distinct records.

**Level 2 — MinHash LSH:** Jaccard similarity threshold of 0.85 using 128 permutations. Scoped per company slug to avoid cross-company false positives. Sequential loop: each record queries the index then inserts itself, ensuring the first occurrence is kept. Reduces the 50k sample to ~35k unique jobs.

---

## Scoring Dimensions

Each job receives a composite score between 0 and 1:

```
final_score = Σ (weight_i × dimension_score_i)
```

- **Embedding similarity** — cosine similarity between profile embedding and job embedding
- **Skill match** — Jaccard overlap between extracted job skills and user-listed skills
- **Title match** — fuzzy match between target roles and job title
- **Location fit** — exact/partial city match; remote jobs score 1.0 if user is open to remote
- **Experience match** — gap-based decay: 1.0 if requirement met, decays for under/over-qualified
- **Salary fit** — overlap between user range and job salary range
- **Recency** — linear decay over 365 days

---

## Adaptive Learning

After each session, user thumbs-up/down feedback updates a Beta distribution per scoring weight via Thompson Sampling. On the next run, weights are sampled from the posterior — arms with more positive signal get higher weight. Stored in SQLite and persists across sessions.

---

## Data

The pre-loaded corpus is built from the [TechMap International Job Postings (September 2021)](https://www.kaggle.com/datasets/techmap/international-job-postings-september-2021) dataset on Kaggle.

- 48 GB raw JSON → 50,000 sampled → 35,623 after cleaning and deduplication
- Build script: `outputs/kaggle_build.py` (run in Kaggle environment)

---

## Deployment

Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud).  
The `data/preloaded_kaggle_50k.parquet` file (76 MB) is committed to the repo and served directly — no build step at runtime.

---

## Limitations

- Corpus is static (September 2021 snapshot) — job market conditions have shifted
- Resume rewriting requires an OpenAI API key; degrades gracefully without one
- JSearch live search requires a RapidAPI key; pre-loaded mode works without it
- MinHash scoped per company — near-duplicates across different companies are not caught
- Salary data is sparse in the Kaggle corpus (~15% of records have parseable salary ranges)

---

## Author

Jacob Renneisen · UC Davis MSBA · Spring 2026  
BAX-423 Big Data with Professor [Instructor Name]
