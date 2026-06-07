# JobPilot — AI Prompts Documentation

BAX-423 Big Data Final Project · UC Davis MSBA · Spring 2026

This document records all prompts used to build JobPilot with Claude (Anthropic).
Per course requirements, all AI-assisted code generation is documented here.

---

## 1. Project Scaffolding

**Prompt:**
> "Build a Streamlit job-matching app called JobPilot for international MSBA students. It should have a two-phase pipeline: Phase A trains on a Kaggle job corpus using sentence-transformers and FAISS, Phase B fetches live jobs from JSearch and ranks them using a weighted scoring function across embedding similarity, skill match, title match, location fit, experience match, salary fit, and recency. Use SQLite for persistence. Include MinHash LSH deduplication as the BAX-423 big data technique."

**Used for:** Initial project architecture, file structure, `src/` module design.

---

## 2. Deduplication Module

**Prompt:**
> "Write a two-level deduplication pipeline for job postings. Level 1: exact dedup by hashing title + company + first 300 chars of description — intentionally exclude location so the same job in different cities is kept. Level 2: MinHash LSH near-duplicate detection using datasketch, Jaccard threshold 0.85, 128 permutations, scoped per company slug. The loop must be sequential — each record queries the LSH index then inserts itself, so the first occurrence is always kept. Include benchmark metrics: throughput in records/sec and dedup rate."

**Used for:** `src/dedupe.py` — `deduplicate_exact()`, `deduplicate_minhash()`, `full_deduplication()`.

---

## 3. Embedding & Retrieval Module

**Prompt:**
> "Build an embeddings module using sentence-transformers all-MiniLM-L6-v2 (384 dimensions) and FAISS IndexFlatIP. Include: (1) a rich job text builder that concatenates title, company, location, seniority, skills, and description; (2) load_or_build_index that caches the index to disk; (3) K-Means clustering into job families; (4) hybrid retrieval combining FAISS ANN and TF-IDF keyword recall; (5) cluster-boosted scoring that promotes jobs from clusters the user has liked."

**Used for:** `src/embeddings.py`.

---

## 4. Ranking & Adaptive Learning

**Prompt:**
> "Write a job ranker that scores each job across 7 dimensions: embedding_similarity (0.27), skill_match (0.25), location_fit (0.18), salary_fit (0.10), experience_match (0.08), title_match (0.07), recency (0.05). Weights sum to 1.0. For experience: score 1.0 if user meets the requirement, decay toward 0 if under or overqualified. Add Thompson Sampling to adapt weights from thumbs-up/down feedback — maintain a Beta(alpha, beta) distribution per weight and sample from the posterior each run."

**Used for:** `src/ranker.py` — `score_job()`, `RankedJob` dataclass, bandit weight adaptation.

---

## 5. Resume Generator

**Prompt:**
> "Write a resume rewriting module that uses inference-based rewriting only — it may reframe existing experience bullets using language and phrases extracted from the job description, but must never invent credentials, degrees, companies, or years of experience the user hasn't listed. First extract 8-12 capability phrases from the job description. Then pass those phrases plus the user's existing resume bullets to the LLM and ask it to rewrite using the job's language while keeping every factual claim grounded in the original resume."

**Used for:** `src/resume_generator.py` — `RESUME_SYSTEM_PROMPT`, `_extract_key_phrases()`, `generate_resume()`.

---

## 6. Data Ingestion & Query Broadening

**Prompt:**
> "Write a JSearch ingestion module that fetches job postings for a list of queries. If the total pool after fetching is less than 5,000 records (measured pre-deduplication), fire 12 broad fallback queries ('full time jobs usa', 'remote jobs united states', 'hiring now', etc.) one at a time until the minimum is met. Do not pad with synthetic data — only use real JSearch results. Deduplicate by URL before appending each batch."

**Used for:** `src/ingest.py` — `fetch_multiple_queries()`, `broaden_jsearch_to_min()`.

---

## 7. Kaggle Corpus Builder

**Prompt:**
> "Write a self-contained Python script for the Kaggle environment that: streams the 48 GB TechMap JSON dump without loading it all into memory; samples 50,000 rows; applies dropna on title and description; standardizes columns to the JobPilot schema; extracts features (remote flag, seniority, skills, experience years, salary range, recency score, embeddable flag); runs Level-1 exact dedup then Level-2 MinHash LSH dedup; and saves the result to /kaggle/working/preloaded_kaggle_50k.parquet. All helpers must be inline — no imports from the JobPilot src/ package."

**Used for:** `outputs/kaggle_build.py`.

---

## 8. SQLite Persistence Layer

**Prompt:**
> "Build a SQLite storage module for a Streamlit app. Tables needed: users (login), profiles (job seeker attributes), feedback (thumbs up/down per job per user), bandit_weights (Thompson Sampling Beta parameters per weight per user), resumes (generated resume text), job_lists (saved job IDs per user). Include save and load functions for each. Use WAL mode for concurrent access safety."

**Used for:** `src/storage.py`.

---

## 9. Streamlit UI

**Prompt:**
> "Build a multi-tab Streamlit app with: (1) Login tab with username/password; (2) Custom Profile tab that auto-fills from an uploaded PDF resume using regex extraction, with a Saved Profile tab to reload the last profile; (3) Job Matches tab with a pipeline runner (pre-loaded Kaggle corpus or live JSearch), ranked job cards with score breakdown, thumbs up/down feedback, and a save list; (4) Resume Generator tab that rewrites the resume for a selected job; (5) Analytics tab showing score distributions and feedback history."

**Used for:** `app.py`.

---

## 10. Technical Brief & README

**Prompt:**
> "Write a comprehensive README and 4-page technical brief for JobPilot covering: architecture diagram, BAX-423 technique choices with benchmarks (MinHash LSH vs exact string matching), pipeline design, test persona results, and limitations. The brief should cover: executive summary, system architecture, data pipeline, technique justification with complexity analysis, adaptive learning design, test results, and limitations."

**Used for:** `README.md`, technical brief PDF.

---

## Notes on AI Usage

- All prompts were authored by Jacob Renneisen
- Generated code was reviewed, tested, and in many cases modified before use
- No AI-generated content was submitted without understanding and verification
- The MinHash LSH sequential loop design, the inference-only resume constraint, and the Thompson Sampling weight adaptation were all original design decisions specified in the prompts, not suggested by the AI
