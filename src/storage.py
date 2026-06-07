"""
storage.py — Persistent SQLite storage for JobPilot user profiles, feedback,
             bandit state, ranking weights, and generated resumes.

Each user gets their own row in every table keyed by user_id (their chosen name).
Feedback history is replayed on login to restore the adaptive learner's state,
so the model continues improving across sessions.

Database file: data/jobpilot.db  (auto-created on first run)
"""

import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

from src.utils import DATA_DIR, logger

DB_PATH = DATA_DIR / "jobpilot.db"


# ─── Connection ───────────────────────────────────────────────────────────────
@contextmanager
def _get_conn():
    """Thread-safe SQLite connection with WAL mode for concurrent reads."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row        # rows behave like dicts
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ─── Schema ───────────────────────────────────────────────────────────────────
def init_db():
    """
    Create all tables if they don't exist.
    Safe to call on every app startup — existing data is never touched.
    """
    with _get_conn() as conn:
        conn.executescript("""
            -- One row per registered user
            CREATE TABLE IF NOT EXISTS users (
                user_id     TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                created_at  TEXT DEFAULT (datetime('now')),
                last_seen   TEXT DEFAULT (datetime('now'))
            );

            -- Full profile JSON (skills, prefs, resume text, dealbreakers, etc.)
            CREATE TABLE IF NOT EXISTS profiles (
                user_id     TEXT PRIMARY KEY REFERENCES users(user_id),
                profile_json TEXT NOT NULL,
                updated_at  TEXT DEFAULT (datetime('now'))
            );

            -- Every feedback event ever recorded for a user
            -- Used to replay adaptive learning state on login
            CREATE TABLE IF NOT EXISTS feedback (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id         TEXT REFERENCES users(user_id),
                job_id          TEXT NOT NULL,
                job_title       TEXT,
                company         TEXT,
                seniority       TEXT,
                industry        TEXT,
                remote          INTEGER,           -- 0 or 1
                salary_midpoint REAL,
                matched_skills  TEXT,              -- JSON array
                feedback_type   TEXT NOT NULL,     -- good/bad/save/skip
                final_score     REAL,
                recorded_at     TEXT DEFAULT (datetime('now'))
            );

            -- Bandit arm states (alpha/beta per seniority×industry×location cluster)
            -- Persisted so exploration progress survives restarts
            CREATE TABLE IF NOT EXISTS bandit_arms (
                user_id     TEXT REFERENCES users(user_id),
                arm_key     TEXT NOT NULL,
                alpha       REAL DEFAULT 1.0,
                beta        REAL DEFAULT 1.0,
                updated_at  TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (user_id, arm_key)
            );

            -- Ranking formula weights (updated by online weight learner)
            CREATE TABLE IF NOT EXISTS ranking_weights (
                user_id     TEXT PRIMARY KEY REFERENCES users(user_id),
                weights_json TEXT NOT NULL,
                updated_at  TEXT DEFAULT (datetime('now'))
            );

            -- Generated resumes (cached per user × job)
            CREATE TABLE IF NOT EXISTS resumes (
                user_id     TEXT REFERENCES users(user_id),
                job_id      TEXT NOT NULL,
                job_title   TEXT,
                company     TEXT,
                resume_md   TEXT NOT NULL,
                method      TEXT,                  -- 'ai' or 'template'
                generated_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (user_id, job_id)
            );

            -- Key-value store for misc per-user data (job lists, settings, etc.)
            CREATE TABLE IF NOT EXISTS user_metadata (
                user_id     TEXT REFERENCES users(user_id),
                key         TEXT NOT NULL,
                value       TEXT NOT NULL,
                updated_at  TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (user_id, key)
            );

            -- Indexes for common query patterns
            CREATE INDEX IF NOT EXISTS idx_feedback_user
                ON feedback(user_id, recorded_at DESC);
            CREATE INDEX IF NOT EXISTS idx_feedback_job
                ON feedback(user_id, job_id);
        """)
    logger.info(f"Database ready: {DB_PATH}")


# ─── User management ──────────────────────────────────────────────────────────
def create_or_update_user(user_id: str, display_name: str) -> dict:
    """Register a new user or update last_seen for an existing one."""
    with _get_conn() as conn:
        conn.execute("""
            INSERT INTO users (user_id, display_name, created_at, last_seen)
            VALUES (?, ?, datetime('now'), datetime('now'))
            ON CONFLICT(user_id) DO UPDATE SET
                last_seen = datetime('now'),
                display_name = excluded.display_name
        """, (user_id, display_name))
    return get_user(user_id)


def get_user(user_id: str) -> dict | None:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
        return dict(row) if row else None


def list_users() -> list[dict]:
    """Return all registered users ordered by last seen."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM users ORDER BY last_seen DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def delete_user(user_id: str):
    """Delete a user and all their data."""
    with _get_conn() as conn:
        for table in ["resumes", "ranking_weights", "bandit_arms",
                       "feedback", "profiles", "users"]:
            conn.execute(f"DELETE FROM {table} WHERE user_id = ?", (user_id,))
    logger.info(f"Deleted all data for user: {user_id}")


# ─── Profile ──────────────────────────────────────────────────────────────────
def save_profile(user_id: str, profile: dict):
    """Persist the full user profile dict as JSON."""
    # Strip non-serializable fields
    safe = {k: v for k, v in profile.items()
            if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
    with _get_conn() as conn:
        conn.execute("""
            INSERT INTO profiles (user_id, profile_json, updated_at)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(user_id) DO UPDATE SET
                profile_json = excluded.profile_json,
                updated_at   = datetime('now')
        """, (user_id, json.dumps(safe)))
    logger.debug(f"Profile saved for {user_id}")


def load_profile(user_id: str) -> dict | None:
    """Load a user's profile from the database."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT profile_json FROM profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
        return json.loads(row["profile_json"]) if row else None


# ─── Feedback ─────────────────────────────────────────────────────────────────
def save_feedback_event(user_id: str, job, feedback_type: str):
    """
    Persist a single feedback event.
    Called every time a user clicks Good Fit / Not For Me / Save / Skip.
    """
    matched = json.dumps(getattr(job, "matched_skills", []) or [])
    with _get_conn() as conn:
        conn.execute("""
            INSERT INTO feedback
                (user_id, job_id, job_title, company, seniority, industry,
                 remote, salary_midpoint, matched_skills, feedback_type,
                 final_score, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """, (
            user_id,
            job.job_id,
            job.title,
            job.company,
            job.seniority,
            _infer_industry(job.title),
            int(job.remote),
            job.salary_midpoint,
            matched,
            feedback_type,
            job.final_score,
        ))
    logger.debug(f"Feedback saved: {user_id} → {job.job_id} → {feedback_type}")


def load_feedback_history(user_id: str) -> list[dict]:
    """Load all feedback events for a user, newest first."""
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT * FROM feedback
            WHERE user_id = ?
            ORDER BY recorded_at DESC
        """, (user_id,)).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["matched_skills"] = json.loads(d.get("matched_skills") or "[]")
            results.append(d)
        return results


def get_feedback_summary(user_id: str) -> dict:
    """Return aggregated feedback stats for display in the UI."""
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT
                feedback_type,
                COUNT(*) as count,
                AVG(final_score) as avg_score
            FROM feedback
            WHERE user_id = ?
            GROUP BY feedback_type
        """, (user_id,)).fetchall()

        summary = {r["feedback_type"]: {
            "count": r["count"],
            "avg_score": round(r["avg_score"] or 0, 3)
        } for r in rows}

        # Top liked companies
        liked = conn.execute("""
            SELECT company, COUNT(*) as cnt
            FROM feedback
            WHERE user_id = ? AND feedback_type IN ('good','save')
            GROUP BY company ORDER BY cnt DESC LIMIT 5
        """, (user_id,)).fetchall()
        summary["top_liked_companies"] = [r["company"] for r in liked]

        # Top liked job titles
        liked_titles = conn.execute("""
            SELECT job_title, COUNT(*) as cnt
            FROM feedback
            WHERE user_id = ? AND feedback_type IN ('good','save')
            GROUP BY job_title ORDER BY cnt DESC LIMIT 5
        """, (user_id,)).fetchall()
        summary["top_liked_titles"] = [r["job_title"] for r in liked_titles]

        # Most disliked industries/seniority
        disliked = conn.execute("""
            SELECT seniority, industry, COUNT(*) as cnt
            FROM feedback
            WHERE user_id = ? AND feedback_type = 'bad'
            GROUP BY seniority, industry ORDER BY cnt DESC LIMIT 5
        """, (user_id,)).fetchall()
        summary["disliked_patterns"] = [dict(r) for r in disliked]

        summary["total_events"] = sum(
            v["count"] for k, v in summary.items()
            if isinstance(v, dict) and "count" in v
        )
        return summary


# ─── Bandit state ─────────────────────────────────────────────────────────────
def save_bandit_state(user_id: str, bandit):
    """Persist all arm alpha/beta values from the bandit."""
    with _get_conn() as conn:
        for arm_key, arm in bandit.arms.items():
            conn.execute("""
                INSERT INTO bandit_arms (user_id, arm_key, alpha, beta, updated_at)
                VALUES (?, ?, ?, ?, datetime('now'))
                ON CONFLICT(user_id, arm_key) DO UPDATE SET
                    alpha      = excluded.alpha,
                    beta       = excluded.beta,
                    updated_at = datetime('now')
            """, (user_id, arm_key, arm.alpha, arm.beta))
    logger.debug(f"Bandit state saved: {len(bandit.arms)} arms for {user_id}")


def load_bandit_state(user_id: str, bandit):
    """
    Restore a bandit's arm distributions from the database.
    Called on login so the model picks up exactly where it left off.
    """
    from src.adaptive_learning import ArmStats
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT arm_key, alpha, beta FROM bandit_arms WHERE user_id = ?",
            (user_id,)
        ).fetchall()
        for r in rows:
            bandit.arms[r["arm_key"]] = ArmStats(
                alpha=r["alpha"], beta=r["beta"]
            )
    if rows:
        logger.info(f"Bandit restored: {len(rows)} arms for {user_id}")
    return bandit


def replay_feedback_into_bandit(user_id: str, bandit, updater):
    """
    Replay full feedback history through the bandit and weight updater.
    This is called on login to rebuild the learned model from stored events,
    so the adaptive ranker improves across sessions — not just within one.
    """
    history = load_feedback_history(user_id)
    if not history:
        return bandit, updater

    logger.info(f"Replaying {len(history)} feedback events for {user_id}...")

    # Replay in chronological order (history is newest-first, so reverse)
    from src.adaptive_learning import ArmStats
    reward_map = {"good": 1.0, "save": 0.85, "skip": 0.4, "bad": 0.0}

    for event in reversed(history):
        arm_key = f"{event['seniority']}_{event['industry']}_{('remote' if event['remote'] else 'onsite')}"
        reward  = reward_map.get(event["feedback_type"], 0.5)

        arm = bandit.arms.setdefault(arm_key, ArmStats())
        arm.alpha += reward
        arm.beta  += (1.0 - reward)
        bandit.total_interactions += 1

    logger.info(f"Replay complete — bandit has {len(bandit.arms)} active arms")
    return bandit, updater


# ─── Ranking weights ──────────────────────────────────────────────────────────
def save_ranking_weights(user_id: str, weights: dict):
    """Persist the current ranking formula weights."""
    with _get_conn() as conn:
        conn.execute("""
            INSERT INTO ranking_weights (user_id, weights_json, updated_at)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(user_id) DO UPDATE SET
                weights_json = excluded.weights_json,
                updated_at   = datetime('now')
        """, (user_id, json.dumps(weights)))


def load_ranking_weights(user_id: str) -> dict | None:
    """Load persisted ranking weights, or None if not yet set."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT weights_json FROM ranking_weights WHERE user_id = ?",
            (user_id,)
        ).fetchone()
        return json.loads(row["weights_json"]) if row else None


# ─── Resumes ──────────────────────────────────────────────────────────────────
def save_resume(user_id: str, job, resume_result: dict):
    """Cache a generated resume so it doesn't need to be regenerated."""
    with _get_conn() as conn:
        conn.execute("""
            INSERT INTO resumes
                (user_id, job_id, job_title, company, resume_md, method, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(user_id, job_id) DO UPDATE SET
                resume_md    = excluded.resume_md,
                method       = excluded.method,
                generated_at = datetime('now')
        """, (
            user_id, job.job_id, job.title, job.company,
            resume_result.get("markdown", ""),
            resume_result.get("method", "template"),
        ))


def load_resumes(user_id: str) -> dict:
    """Load all cached resumes for a user. Returns {job_id: resume_result}."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM resumes WHERE user_id = ? ORDER BY generated_at DESC",
            (user_id,)
        ).fetchall()
        return {
            r["job_id"]: {
                "markdown": r["resume_md"],
                "method":   r["method"],
                "job_title": r["job_title"],
                "company":   r["company"],
                "warning": (
                    "⚠️ JobPilot tailors wording only — it does not invent credentials. "
                    "Always review before submitting."
                ),
                "matched_skills": [],
                "missing_skills": [],
            }
            for r in rows
        }


# ─── Learning insights (for UI display) ───────────────────────────────────────
def get_learning_insights(user_id: str) -> dict:
    """
    Derive personalised insights from the full feedback history.
    Used to show the user what the model has learned about their preferences.
    """
    history = load_feedback_history(user_id)
    if not history:
        return {}

    liked   = [e for e in history if e["feedback_type"] in ("good", "save")]
    disliked= [e for e in history if e["feedback_type"] == "bad"]

    from collections import Counter

    # Preferred seniority levels
    liked_seniority = Counter(e["seniority"] for e in liked if e["seniority"])

    # Preferred industries
    liked_industry  = Counter(e["industry"]  for e in liked if e["industry"])

    # Avoided patterns
    disliked_seniority = Counter(e["seniority"] for e in disliked if e["seniority"])
    disliked_industry  = Counter(e["industry"]  for e in disliked if e["industry"])

    # Avg score of liked vs disliked
    avg_liked_score    = (sum(e["final_score"] or 0 for e in liked)
                          / max(len(liked), 1))
    avg_disliked_score = (sum(e["final_score"] or 0 for e in disliked)
                          / max(len(disliked), 1))

    # Top matched skills in liked jobs
    from collections import Counter as C
    all_matched = []
    for e in liked:
        all_matched.extend(e.get("matched_skills") or [])
    top_matched_skills = C(all_matched).most_common(8)

    return {
        "total_feedback":       len(history),
        "liked_count":          len(liked),
        "disliked_count":       len(disliked),
        "preferred_seniority":  liked_seniority.most_common(3),
        "preferred_industry":   liked_industry.most_common(3),
        "avoided_seniority":    disliked_seniority.most_common(3),
        "avoided_industry":     disliked_industry.most_common(3),
        "avg_liked_score":      round(avg_liked_score, 3),
        "avg_disliked_score":   round(avg_disliked_score, 3),
        "top_matched_skills":   top_matched_skills,
        "sessions_count":       _count_sessions(user_id),
    }


def _count_sessions(user_id: str) -> int:
    """Approximate number of sessions by counting distinct feedback dates."""
    with _get_conn() as conn:
        row = conn.execute("""
            SELECT COUNT(DISTINCT date(recorded_at)) as days
            FROM feedback WHERE user_id = ?
        """, (user_id,)).fetchone()
        return row["days"] if row else 0


# ─── Job list (saved per profile) ────────────────────────────────────────────
def save_job_list(user_id: str, ranked_jobs: list):
    """
    Persist the top-20 ranked job summaries so a profile can show its job list.
    Stores only lightweight metadata — no heavy embeddings.
    """
    jobs_data = []
    for j in ranked_jobs[:20]:
        jobs_data.append({
            "rank":         getattr(j, "rank", 0),
            "job_id":       getattr(j, "job_id", ""),
            "title":        getattr(j, "title", ""),
            "company":      getattr(j, "company", ""),
            "location":     getattr(j, "location", ""),
            "remote":       getattr(j, "remote", False),
            "seniority":    getattr(j, "seniority", ""),
            "final_score":  round(getattr(j, "final_score", 0.0), 4),
            "salary_min":   getattr(j, "salary_min", 0),
            "salary_max":   getattr(j, "salary_max", 0),
            "url":          getattr(j, "url", ""),
            "source":       getattr(j, "source", ""),
            "date_posted":  getattr(j, "date_posted", ""),
        })
    with _get_conn() as conn:
        conn.execute("""
            INSERT INTO user_metadata (user_id, key, value, updated_at)
            VALUES (?, 'job_list', ?, datetime('now'))
            ON CONFLICT(user_id, key) DO UPDATE SET
                value      = excluded.value,
                updated_at = datetime('now')
        """, (user_id, json.dumps(jobs_data)))
    logger.debug(f"Job list saved: {len(jobs_data)} jobs for {user_id}")


def load_job_list(user_id: str) -> list[dict]:
    """Load the saved ranked job list for a user. Returns [] if none saved."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT value, updated_at FROM user_metadata WHERE user_id = ? AND key = 'job_list'",
            (user_id,)
        ).fetchone()
        if not row:
            return []
        jobs = json.loads(row["value"])
        # Attach the timestamp so the UI can show "last updated at..."
        for j in jobs:
            j["_saved_at"] = row["updated_at"]
        return jobs


# ─── Utility ──────────────────────────────────────────────────────────────────
def _infer_industry(title: str) -> str:
    """Quick title-to-industry mapping (mirrors adaptive_learning.py)."""
    t = title.lower()
    if any(k in t for k in ["machine learning", "ml", "ai ", "deep learning", "nlp",
                              "computer vision", "research scientist", "applied scientist"]):
        return "ml_ai"
    if any(k in t for k in ["data scientist", "data analyst", "data engineer",
                              "bi analyst", "analytics"]):
        return "data"
    if any(k in t for k in ["software engineer", "backend", "frontend",
                              "platform", "devops", "mlops", "sre"]):
        return "swe"
    if any(k in t for k in ["fintech", "quant", "trading", "banking"]):
        return "finance"
    if any(k in t for k in ["healthcare", "health", "clinical", "pharma"]):
        return "healthcare"
    return "other"


def db_size_kb() -> float:
    """Return the database file size in KB."""
    if DB_PATH.exists():
        return round(DB_PATH.stat().st_size / 1024, 1)
    return 0.0
