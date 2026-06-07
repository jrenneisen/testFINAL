"""
utils.py — Shared utilities, constants, and logging for JobPilot.
"""

import os
import re
import logging
import hashlib
from pathlib import Path

# ─── Load env ─────────────────────────────────────────────────────────────────
# python-dotenv is only needed locally — on Streamlit Cloud, secrets are
# injected as real environment variables automatically from the Secrets tab.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Running on Streamlit Cloud or dotenv not installed — that's fine

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).parent.parent
DATA_DIR        = ROOT_DIR / "data"
OUTPUTS_DIR     = ROOT_DIR / "outputs"
RAW_CSV         = DATA_DIR / "raw_jobs.csv"
CLEAN_PARQUET   = DATA_DIR / "jobs_clean.parquet"
SAMPLE_PARQUET  = DATA_DIR / "jobs_sample.parquet"
FAISS_INDEX     = DATA_DIR / "faiss_index.bin"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npy"
JOB_IDS_FILE    = DATA_DIR / "job_ids.npy"
CLUSTERS_FILE   = DATA_DIR / "job_clusters.npz"   # K-Means job family clusters
PERSONAS_FILE   = DATA_DIR / "personas.json"

DATA_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# ─── Env ──────────────────────────────────────────────────────────────────────
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
JSEARCH_API_KEY  = os.getenv("JSEARCH_API_KEY", "")
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OFFLINE_MODE     = os.getenv("OFFLINE_MODE", "false").lower() == "true"
TOP_K_JOBS       = int(os.getenv("TOP_K_JOBS", "20"))
RETRIEVAL_K      = int(os.getenv("RETRIEVAL_K", "200"))

# ─── Ranking weights (default; updated by adaptive learning) ──────────────────
DEFAULT_WEIGHTS = {
    "embedding_similarity": 0.27,  # Reduced slightly — broad JSearch uses embedding for matching
    "skill_match":          0.25,
    "title_match":          0.07,  # Lowered — broad queries don't filter by title keyword
    "location_fit":         0.18,  # Raised — location is a key hard preference signal
    "experience_match":     0.08,  # NEW — rewards appropriate experience fit
    "salary_fit":           0.10,
    "recency":              0.05,
}

# ─── Skill taxonomy ───────────────────────────────────────────────────────────
TECH_SKILLS = [
    # Programming
    "python", "r", "java", "javascript", "typescript", "c++", "c#", "scala",
    "go", "rust", "ruby", "php", "swift", "kotlin", "matlab",
    # Data / ML
    "sql", "nosql", "mongodb", "postgresql", "mysql", "spark", "hadoop",
    "kafka", "airflow", "dbt", "databricks", "snowflake", "redshift",
    "pandas", "numpy", "scikit-learn", "sklearn", "tensorflow", "pytorch",
    "keras", "xgboost", "lightgbm", "hugging face", "transformers", "bert",
    "gpt", "llm", "nlp", "computer vision", "deep learning", "machine learning",
    "reinforcement learning", "mlops", "mlflow", "kubeflow",
    # Cloud / Infra
    "aws", "gcp", "azure", "kubernetes", "docker", "terraform", "ci/cd",
    "git", "github", "gitlab", "jenkins", "linux",
    # BI / Viz
    "tableau", "power bi", "looker", "matplotlib", "plotly", "seaborn", "d3",
    # Analytics
    "a/b testing", "hypothesis testing", "statistics", "regression",
    "time series", "forecasting", "causal inference",
    # Soft / Domain
    "communication", "stakeholder management", "agile", "scrum", "product analytics",
]

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("jobpilot")


# ─── Text helpers ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Remove HTML tags, extra whitespace, and normalize."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)          # strip HTML
    text = re.sub(r"&[a-z]+;", " ", text)          # HTML entities
    text = re.sub(r"[^\w\s.,;:()\-/+#]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_job_id(title: str, company: str, description: str) -> str:
    """Stable SHA-256 based job ID."""
    raw = f"{title}|{company}|{description[:200]}".lower().strip()
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def extract_skills_from_text(text: str) -> list[str]:
    """Extract known tech skills from free text using keyword matching."""
    text_lower = text.lower()
    found = []
    for skill in TECH_SKILLS:
        # Use word-boundary aware matching
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.append(skill)
    return list(dict.fromkeys(found))  # dedupe, preserve order


def detect_seniority(title: str, description: str = "") -> str:
    """Classify job seniority from title and description."""
    text = (title + " " + description[:300]).lower()
    if any(w in text for w in ["staff ", "principal ", "distinguished ", "fellow "]):
        return "staff"
    if any(w in text for w in ["senior ", "sr.", "sr ", "lead ", "director ", "vp ", "head of"]):
        return "senior"
    if any(w in text for w in ["junior ", "jr.", "jr ", "entry level", "entry-level", "associate ", "intern"]):
        return "junior"
    if re.search(r"\bii\b|\biii\b", text):
        return "mid"
    return "mid"


def detect_remote(location: str, description: str = "") -> bool:
    """Detect if a job is remote."""
    combined = (location + " " + description[:500]).lower()
    remote_keywords = ["remote", "work from home", "wfh", "distributed team",
                       "fully remote", "100% remote", "telecommute"]
    return any(kw in combined for kw in remote_keywords)


def detect_experience_required(description: str) -> int:
    """Extract minimum years of experience required."""
    matches = re.findall(r"(\d+)\+?\s*(?:to\s*\d+\s*)?years?\s*(?:of\s*)?(?:experience|exp)", description.lower())
    if matches:
        return min(int(m) for m in matches)
    return 0


def detect_visa_possible(description: str, company: str = "") -> bool:
    """Detect if job likely sponsors visas."""
    text = (description + " " + company).lower()
    positive = ["h-1b", "h1b", "visa sponsorship", "sponsor", "will sponsor",
                "work authorization", "open to international"]
    negative = ["no sponsorship", "citizen only", "us citizen", "security clearance",
                "secret clearance", "top secret", "clearance required"]
    if any(kw in text for kw in negative):
        return False
    if any(kw in text for kw in positive):
        return True
    # Large known sponsors (conservative list)
    known_sponsors = ["google", "microsoft", "amazon", "meta", "apple", "netflix",
                      "uber", "airbnb", "salesforce", "oracle", "ibm", "intel",
                      "nvidia", "qualcomm", "linkedin", "twitter", "stripe"]
    return any(s in text for s in known_sponsors)


def parse_salary(salary_str: str) -> tuple[float, float]:
    """Parse salary string into (min, max) floats. Returns (0, 0) if unparseable."""
    if not isinstance(salary_str, str) or not salary_str.strip():
        return 0.0, 0.0
    # Strip non-numeric except separators
    nums = re.findall(r"[\d,]+(?:\.\d+)?", salary_str.replace(",", ""))
    if not nums:
        return 0.0, 0.0
    floats = [float(n.replace(",", "")) for n in nums if n]
    # Normalize: if values look like hourly (< 500), convert to annual
    floats = [f * 2080 if f < 500 else f for f in floats]
    if len(floats) == 1:
        return floats[0], floats[0]
    return min(floats), max(floats)


def salary_midpoint(s_min: float, s_max: float) -> float:
    if s_min == 0 and s_max == 0:
        return 0.0
    if s_min == 0:
        return s_max
    if s_max == 0:
        return s_min
    return (s_min + s_max) / 2
