"""
ranker.py — Multi-stage job ranking pipeline for JobPilot.

Pipeline:
  Stage 1: Hard filters (dealbreakers, salary, location, seniority, visa)
  Stage 2: Feature scoring (embedding sim + skill match + preference fit)
  Stage 3: Combined weighted score
  Stage 4: MMR re-ranking (diversity)
  Stage 5: "Why ranked here?" explanation generation

BAX-423 Technique: Multi-Stage Re-Ranking
"""

import re
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any

from src.utils import logger, DEFAULT_WEIGHTS, TOP_K_JOBS


# ─── Data classes ─────────────────────────────────────────────────────────────
@dataclass
class RankedJob:
    job_id:             str
    title:              str
    company:            str
    location:           str
    remote:             bool
    salary_min:         float
    salary_max:         float
    salary_midpoint:    float
    description:        str
    skills_extracted:   list[str]
    seniority:          str
    employment_type:    str
    experience_required: int
    visa_possible:      bool
    date_posted:        str
    recency_score:      float
    url:                str
    source:             str

    # Ranking scores
    embedding_score:      float = 0.0
    skill_match_score:    float = 0.0
    title_match_score:    float = 0.0
    location_fit_score:   float = 0.0
    experience_match_score: float = 0.0
    salary_fit_score:     float = 0.0
    recency_weight:       float = 0.0
    final_score:          float = 0.0

    # Explanation fields
    matched_skills:     list[str] = field(default_factory=list)
    missing_skills:     list[str] = field(default_factory=list)
    bonus_skills:       list[str] = field(default_factory=list)
    why_ranked:         str = ""
    rank:               int = 0

    # Feedback
    feedback:           str = ""      # "good", "bad", "save", "skip"


# ─── Stage 1: Hard Filters ────────────────────────────────────────────────────
def apply_hard_filters(df: pd.DataFrame, profile: dict) -> pd.DataFrame:
    """
    Remove jobs that clearly violate user dealbreakers.
    Returns filtered DataFrame (keeps enough for ranking — not over-filtered).
    """
    original = len(df)
    mask = pd.Series(True, index=df.index)

    # 1. Salary filter (only if both salary and preference provided)
    sal_min_pref = profile.get("salary_min", 0) or 0
    if sal_min_pref > 0:
        # Keep if: no salary data OR salary_max >= 70% of user minimum (flexible)
        sal_ok = (df["salary_max"] <= 0) | (df["salary_max"] >= sal_min_pref * 0.7)
        mask &= sal_ok

    # 2. Dealbreaker keyword filter
    dealbreakers = [d.lower() for d in (profile.get("dealbreakers") or [])]
    if dealbreakers:
        job_text = (df["title"].str.lower() + " " + df["company"].str.lower() +
                    " " + df["description"].str.lower().str[:500])
        for db in dealbreakers:
            # Use whole-word matching for short dealbreakers, substring for long
            if len(db) > 8:
                mask &= ~job_text.str.contains(re.escape(db), na=False)
            else:
                mask &= ~job_text.str.contains(r"\b" + re.escape(db) + r"\b", na=False, regex=True)

    # 3. Seniority filter
    seniority_target = profile.get("seniority_target", "mid")
    if seniority_target == "junior":
        mask &= ~df["seniority"].isin(["senior", "staff"])
    elif seniority_target == "senior":
        mask &= ~df["seniority"].isin(["junior"])

    # 4. Remote filter (only if user strictly requires remote)
    if profile.get("remote_required", False):
        mask &= df["remote"] == True

    # 5. Employment type: remove contract-only if user flags it
    if "contract" in dealbreakers or "contract only" in dealbreakers:
        mask &= df["employment_type"] != "Contract"

    # 6. Visa / sponsorship filter
    if profile.get("visa_required", False):
        # Keep jobs that mention sponsorship OR have no explicit denial
        mask &= df["visa_possible"] == True

    # 7. Experience: remove jobs requiring far more than user's experience
    user_exp = profile.get("years_experience", 0) or 0
    if user_exp < 2:
        mask &= df["experience_required"] <= 3   # junior: max 3 years req
    elif user_exp < 5:
        mask &= df["experience_required"] <= 7   # mid: allow up to 7
    # Senior: no upper filter

    filtered = df[mask].copy()
    removed = original - len(filtered)
    logger.info(f"Hard filters: removed {removed:,} ({removed/max(original,1):.1%}), kept {len(filtered):,}")

    # Safety: if too few remain, relax and return top 100 by seniority match
    if len(filtered) < 20:
        logger.warning("Too few after hard filters — relaxing constraints")
        filtered = df.copy()

    return filtered


# ─── Stage 2: Feature Scoring ─────────────────────────────────────────────────
def score_jobs(
    df: pd.DataFrame,
    profile: dict,
    candidate_scores: list[tuple[str, float]],
    weights: dict | None = None,
) -> list[RankedJob]:
    """
    Score each candidate job across 5 dimensions and produce RankedJob objects.
    """
    weights = weights or DEFAULT_WEIGHTS.copy()

    # Build lookup: job_id → embedding_score
    emb_lookup = {jid: score for jid, score in candidate_scores}

    # Build job lookup by job_id
    job_lookup = {row["job_id"]: row for _, row in df.iterrows()}

    profile_skills = set(s.lower() for s in (profile.get("skills") or []))
    target_roles   = [r.lower() for r in (profile.get("target_roles") or [])]
    sal_min_pref   = profile.get("salary_min", 0) or 0
    locations_pref = [l.lower() for l in (profile.get("locations") or [])]
    user_exp       = float(profile.get("years_experience", 0) or 0)

    ranked_jobs = []

    for job_id, emb_score in candidate_scores:
        row = job_lookup.get(job_id)
        if row is None:
            continue

        # --- Skill Match ---
        job_skills = set(s.lower() for s in (row.get("skills_extracted") or []))
        if isinstance(row.get("skills_extracted"), str):
            import ast
            try:
                job_skills = set(s.lower() for s in ast.literal_eval(row["skills_extracted"]))
            except Exception:
                job_skills = set()

        matched  = list(profile_skills & job_skills)
        missing  = list(job_skills - profile_skills)
        bonus    = list(profile_skills - job_skills)  # extra skills user has

        skill_match = len(matched) / max(len(job_skills), 1)

        # --- Title Match ---
        title_lower = str(row.get("title", "")).lower()
        title_match = 1.0 if any(role in title_lower for role in target_roles) \
                      else 0.5 if any(word in title_lower for role in target_roles
                                      for word in role.split()) \
                      else 0.0

        # --- Location / Remote Fit ---
        job_location = str(row.get("location", "")).lower()
        is_remote    = bool(row.get("remote", False))
        loc_fit = 1.0 if is_remote else (
            1.0 if any(loc in job_location for loc in locations_pref) else 0.2
        )

        # --- Salary Fit ---
        sal_mid = float(row.get("salary_midpoint", 0) or 0)
        if sal_mid == 0 or sal_min_pref == 0:
            sal_fit = 0.5  # unknown → neutral
        else:
            ratio    = sal_mid / sal_min_pref
            sal_fit  = min(1.0, max(0.0, ratio))

        # --- Experience Match ---
        job_exp_req = float(row.get("experience_required", 0) or 0)
        if job_exp_req == 0:
            exp_fit = 0.7  # Unknown requirement → neutral-positive
        else:
            gap = user_exp - job_exp_req
            if gap >= 0:
                # User meets or exceeds requirement
                # Slight penalty for being heavily overqualified (gap > 5 years)
                exp_fit = 1.0 if gap <= 5 else max(0.5, 1.0 - (gap - 5) * 0.08)
            else:
                # User is underqualified — penalty scales with gap
                exp_fit = max(0.1, 1.0 + gap * 0.2)  # gap<0, so this subtracts

        # --- Recency ---
        recency = float(row.get("recency_score", 0.5) or 0.5)

        # --- Combined Weighted Score ---
        final = (
            weights.get("embedding_similarity", 0.27) * float(emb_score) +
            weights.get("skill_match",          0.25) * skill_match +
            weights.get("title_match",          0.07) * title_match +
            weights.get("location_fit",         0.18) * loc_fit +
            weights.get("experience_match",     0.08) * exp_fit +
            weights.get("salary_fit",           0.10) * sal_fit +
            weights.get("recency",              0.05) * recency
        )

        # Build explanation
        why = _build_explanation(
            row, emb_score, matched, missing, title_match, loc_fit, sal_fit,
            sal_min_pref, exp_fit, job_exp_req, user_exp
        )

        skills_list = list(row.get("skills_extracted") or [])
        if isinstance(skills_list, str):
            import ast
            try:
                skills_list = ast.literal_eval(skills_list)
            except Exception:
                skills_list = []

        ranked_jobs.append(RankedJob(
            job_id                = str(job_id),
            title                 = str(row.get("title", "")),
            company               = str(row.get("company", "")),
            location              = str(row.get("location", "")),
            remote                = bool(row.get("remote", False)),
            salary_min            = float(row.get("salary_min", 0) or 0),
            salary_max            = float(row.get("salary_max", 0) or 0),
            salary_midpoint       = sal_mid,
            description           = str(row.get("description", ""))[:2000],
            skills_extracted      = skills_list,
            seniority             = str(row.get("seniority", "mid")),
            employment_type       = str(row.get("employment_type", "Full-time")),
            experience_required   = int(row.get("experience_required", 0) or 0),
            visa_possible         = bool(row.get("visa_possible", False)),
            date_posted           = str(row.get("date_posted", "")),
            recency_score         = recency,
            url                   = str(row.get("url", "")),
            source                = str(row.get("source", "")),
            embedding_score       = float(emb_score),
            skill_match_score     = skill_match,
            title_match_score     = title_match,
            location_fit_score    = loc_fit,
            experience_match_score= exp_fit,
            salary_fit_score      = sal_fit,
            recency_weight        = recency,
            final_score           = final,
            matched_skills        = matched[:8],
            missing_skills        = missing[:5],
            bonus_skills          = bonus[:5],
            why_ranked            = why,
        ))

    return ranked_jobs


# ─── Stage 3: MMR Re-ranking ──────────────────────────────────────────────────
def mmr_rerank(
    ranked_jobs: list[RankedJob],
    top_n: int = TOP_K_JOBS,
    lambda_param: float = 0.7,
    max_per_company: int = 2,
) -> list[RankedJob]:
    """
    Maximal Marginal Relevance re-ranking for diversity.
    - Penalizes redundancy (same company, same role cluster).
    - Guarantees at most max_per_company results from one employer.
    """
    # Sort by score first
    candidates = sorted(ranked_jobs, key=lambda j: j.final_score, reverse=True)

    selected      = []
    company_count = {}

    for job in candidates:
        if len(selected) >= top_n:
            break

        # Company cap
        co = job.company.lower()
        if company_count.get(co, 0) >= max_per_company:
            continue

        # Duplicate title check
        existing_titles = [s.title.lower() for s in selected]
        if job.title.lower() in existing_titles and len(selected) > 0:
            continue

        selected.append(job)
        company_count[co] = company_count.get(co, 0) + 1

    # Fill remaining slots if company cap was too aggressive
    if len(selected) < min(top_n, len(candidates)):
        remaining = [j for j in candidates if j not in selected]
        selected.extend(remaining[:top_n - len(selected)])

    # Assign rank numbers
    for i, job in enumerate(selected):
        job.rank = i + 1

    return selected[:top_n]


# ─── Main entry point ─────────────────────────────────────────────────────────
def rank_jobs(
    df: pd.DataFrame,
    profile: dict,
    candidate_scores: list[tuple[str, float]],
    weights: dict | None = None,
    top_n: int = TOP_K_JOBS,
    feedback: dict | None = None,
) -> list[RankedJob]:
    """
    Full ranking pipeline.

    Args:
        df:               Cleaned jobs DataFrame
        profile:          User profile dict
        candidate_scores: (job_id, emb_score) from FAISS retrieval
        weights:          Scoring weights (updated by adaptive learning)
        top_n:            Number of results to return
        feedback:         Dict of {job_id: feedback_type}

    Returns:
        List of RankedJob objects, ranked and re-ranked.
    """
    # Filter to candidate job IDs only
    candidate_ids = {jid for jid, _ in candidate_scores}
    df_candidates = df[df["job_id"].isin(candidate_ids)].copy()

    # Stage 1: Hard filters
    df_filtered = apply_hard_filters(df_candidates, profile)

    # Re-filter candidate scores to match filtered df
    filtered_ids = set(df_filtered["job_id"].tolist())
    filtered_scores = [(jid, s) for jid, s in candidate_scores if jid in filtered_ids]

    # Stage 2: Score
    ranked = score_jobs(df_filtered, profile, filtered_scores, weights)

    # Apply feedback boosts if provided
    if feedback:
        ranked = _apply_feedback_boosts(ranked, feedback)
        ranked.sort(key=lambda j: j.final_score, reverse=True)

    # Stage 3: MMR re-rank
    final = mmr_rerank(ranked, top_n=top_n)

    logger.info(f"Ranking complete: {len(final)} jobs returned from {len(ranked)} candidates")
    return final


# ─── Explanation builder ──────────────────────────────────────────────────────
def _build_explanation(
    row: pd.Series,
    emb_score: float,
    matched: list,
    missing: list,
    title_match: float,
    loc_fit: float,
    sal_fit: float,
    sal_min_pref: float,
    exp_fit: float = 0.7,
    job_exp_req: float = 0.0,
    user_exp: float = 0.0,
) -> str:
    """Build the human-readable 'Why this ranked highly' explanation."""
    parts = []

    # Semantic similarity
    pct = int(emb_score * 100)
    parts.append(f"**{pct}% semantic match** to your profile.")

    # Title match
    if title_match >= 1.0:
        parts.append(f"**Strong title alignment** with your target roles.")
    elif title_match >= 0.5:
        parts.append("Partial title alignment with your target roles.")

    # Skills
    if matched:
        top_matched = matched[:4]
        parts.append(f"**Matched skills:** {', '.join(top_matched)}.")
    if missing:
        parts.append(f"**Skill gaps:** {', '.join(missing[:3])} not in your profile.")

    # Experience
    if job_exp_req > 0 and user_exp > 0:
        gap = user_exp - job_exp_req
        if gap >= 0 and gap <= 2:
            parts.append(f"**Experience match:** {int(job_exp_req)}+ yrs required, you have {int(user_exp)}.")
        elif gap > 2:
            parts.append(f"You exceed this role's {int(job_exp_req)}-yr requirement by {int(gap)} years.")
        else:
            parts.append(f"Requires {int(job_exp_req)} yrs experience (you have {int(user_exp)}).")
    elif job_exp_req == 0:
        parts.append("No specific experience requirement listed.")

    # Location
    is_remote = bool(row.get("remote", False))
    if is_remote:
        parts.append("**Remote** position — matches your preference.")
    elif loc_fit >= 0.8:
        parts.append("Location aligns with your preferences.")

    # Salary
    sal_mid = float(row.get("salary_midpoint", 0) or 0)
    if sal_mid > 0 and sal_min_pref > 0:
        if sal_mid >= sal_min_pref:
            parts.append(f"**Salary** (${sal_mid:,.0f} midpoint) meets your target.")
        else:
            parts.append(f"Salary (${sal_mid:,.0f}) is below your ${sal_min_pref:,.0f} target.")
    elif sal_mid == 0:
        parts.append("Salary not listed in posting.")

    # Visa
    if bool(row.get("visa_possible", False)):
        parts.append("**Visa sponsorship** indicated.")

    return " ".join(parts)


def _apply_feedback_boosts(ranked_jobs: list[RankedJob], feedback: dict) -> list[RankedJob]:
    """Apply score boosts/penalties based on stored user feedback."""
    for job in ranked_jobs:
        fb = feedback.get(job.job_id, "")
        if fb == "good":
            job.final_score = min(1.0, job.final_score * 1.15)
        elif fb == "save":
            job.final_score = min(1.0, job.final_score * 1.10)
        elif fb == "bad":
            job.final_score *= 0.5
        elif fb == "skip":
            job.final_score *= 0.85
    return ranked_jobs


# ─── Evaluation Metrics ───────────────────────────────────────────────────────
def compute_ndcg(ranked_jobs: list[RankedJob], relevant_ids: list[str], k: int = 10) -> float:
    """Compute NDCG@k. relevant_ids are the "ground truth" good matches."""
    rel_set = set(relevant_ids)
    gains   = [1.0 if j.job_id in rel_set else 0.0 for j in ranked_jobs[:k]]
    ideal   = sorted(gains, reverse=True)

    def dcg(scores):
        return sum(s / np.log2(i + 2) for i, s in enumerate(scores))

    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def compute_precision_at_k(ranked_jobs: list[RankedJob], relevant_ids: list[str], k: int = 5) -> float:
    """Compute Precision@k."""
    rel_set = set(relevant_ids)
    hits = sum(1 for j in ranked_jobs[:k] if j.job_id in rel_set)
    return hits / k


def benchmark_ranking(
    df: pd.DataFrame,
    profile: dict,
    emb_candidates: list[tuple[str, float]],
    tfidf_candidates: list[tuple[str, float]],
) -> dict:
    """
    Compare embedding-only vs multi-stage ranking.
    Returns benchmark metrics dict.
    """
    # Embedding only (no re-ranking)
    emb_scored = score_jobs(df, profile, emb_candidates)
    emb_sorted = sorted(emb_scored, key=lambda j: j.embedding_score, reverse=True)[:10]

    # Multi-stage (full pipeline)
    multi_ranked = rank_jobs(df, profile, emb_candidates)

    # TF-IDF baseline
    tfidf_scored = score_jobs(df, profile, tfidf_candidates)
    tfidf_sorted = sorted(tfidf_scored, key=lambda j: j.embedding_score, reverse=True)[:10]

    # Simulated persona fit scores (for demo; replace with real labels in full eval)
    def fit_score(jobs, profile):
        target_roles = [r.lower() for r in profile.get("target_roles", [])]
        hits = sum(1 for j in jobs
                   if any(role in j.title.lower() for role in target_roles))
        return hits / max(len(jobs), 1)

    return {
        "method": ["TF-IDF Only", "Embedding Only", "Multi-Stage (Full Pipeline)"],
        "top10_persona_fit":  [
            round(fit_score(tfidf_sorted, profile), 2),
            round(fit_score(emb_sorted, profile), 2),
            round(fit_score(multi_ranked, profile), 2),
        ],
        "dealbreaker_violations": [
            _count_dealbreaker_violations(tfidf_sorted, profile),
            _count_dealbreaker_violations(emb_sorted, profile),
            _count_dealbreaker_violations(multi_ranked, profile),
        ],
        "avg_match_score": [
            round(np.mean([j.embedding_score for j in tfidf_sorted]), 3),
            round(np.mean([j.embedding_score for j in emb_sorted]), 3),
            round(np.mean([j.final_score for j in multi_ranked]), 3),
        ],
    }


def _count_dealbreaker_violations(jobs: list[RankedJob], profile: dict) -> int:
    dealbreakers = [d.lower() for d in (profile.get("dealbreakers") or [])]
    violations = 0
    for job in jobs:
        text = (job.title + " " + job.company + " " + job.description[:300]).lower()
        if any(db in text for db in dealbreakers):
            violations += 1
    return violations
