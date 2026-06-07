"""
resume_generator.py — AI-powered tailored resume generation using OpenAI GPT-4o-mini.

Takes a user profile + selected job description and generates an ATS-optimized,
tailored resume that emphasizes relevant skills without fabricating credentials.
"""

import os
import json
import logging
from openai import OpenAI

from src.utils import OPENAI_API_KEY, OPENAI_MODEL, logger
from src.ranker import RankedJob

# ─── Prompts (also exported to prompts.md) ────────────────────────────────────

RESUME_SYSTEM_PROMPT = """You are an expert resume editor and career coach specializing in tech and data roles.
Your task is to rewrite a candidate's resume to be tailored for a specific job posting.

═══ WHAT YOU MAY DO ═══
✅ Rephrase existing bullet points using the job description's exact language and keywords.
✅ Infer what a real past role DEMONSTRATES and surface that inference explicitly.
   Example: candidate was a "Statistics TA" → job says "strong quantitative skills"
   → Write: "Developed students' quantitative reasoning skills through weekly instruction
     and applied problem-set feedback" — this is true, specific, and uses the job's language.
✅ Reorder sections and bullets to front-load the most relevant experience.
✅ Add a Professional Summary that mirrors the job's priority keywords.
✅ Quantify achievements where the profile contains supporting numbers.
✅ Expand a thin bullet into 2-3 bullets when the role clearly involved multiple
   activities that each map to something the job description asks for.

═══ WHAT YOU MAY NOT DO ═══
❌ Invent employers, job titles, or date ranges that are not in the profile.
❌ Add degrees, certifications, or courses the candidate did not complete.
❌ Claim proficiency in tools or technologies not mentioned anywhere in the profile
   or reasonably implied by the roles listed.
❌ Change numeric claims (e.g. do not say "improved accuracy by 40%" if no number
   was given — use "improved accuracy" or ask the candidate to add the metric).

The key distinction: inferring WHAT an existing experience demonstrates is allowed and
encouraged. Inventing experiences that never happened is never allowed.

FORMATTING:
- Keep to one page for <5 years experience, two pages for 5+ years.
- Produce output in clean Markdown.
- Every experience bullet must start with a strong action verb."""

RESUME_USER_TEMPLATE = """
## CANDIDATE PROFILE
{profile_json}

## TARGET JOB
**Title:** {job_title}
**Company:** {company}
**Location:** {location}
**Experience Required:** {experience_required} years

**Job Description:**
{job_description}

**Key phrases the job uses (mirror these in bullets):**
{key_phrases}

**Required Skills from Posting:** {required_skills}
**Skills Candidate HAS (matched):** {matched_skills}
**Skills Candidate is Missing:** {missing_skills}

## YOUR TASK
Generate a complete, ATS-optimized resume tailored to this specific role.

For EVERY past role in the candidate's experience:
1. Read what that role actually involved.
2. Identify which of the job's key phrases that role genuinely demonstrates.
3. Rewrite the bullet(s) for that role using the job's language — as long as the
   underlying activity really happened. Do not skip any role.

Structure the resume exactly as:
# [Candidate Name]
[email] | [location] | [LinkedIn if available]

## Professional Summary
(3 sentences — use the job's exact priority keywords; make it specific to THIS role)

## Skills
(Lead with skills that appear in the job description; group by category)

## Experience
### [Job Title] | [Company] | [Dates]
(2-4 bullets per role, each reframed using job description language where applicable)

## Education

## Projects (if any in profile)

---
After the resume, add:
## What Was Changed & Why
(3-5 bullets — explain each key reframing decision, e.g.
 "Statistics TA → reframed as 'quantitative skills' because the job specifically requires
  data-driven reasoning and the TA role involved exactly that.")
"""

SKILL_EXTRACTION_PROMPT = """Extract the required technical skills from this job description.
Return a JSON array of strings. Include only specific technical skills, tools, and technologies.
Limit to 15 items. Do not include soft skills.

Job Description:
{description}

Return only valid JSON like: ["Python", "SQL", "AWS"]"""

KEY_PHRASES_PROMPT = """Read this job description and extract 8-12 short phrases that describe
what the employer most wants to see demonstrated. Focus on capability phrases (not tool names),
things like "strong quantitative skills", "cross-functional collaboration",
"data-driven decision making", "stakeholder communication", "process improvement".

Return a JSON array of short phrases (3-7 words each).

Job Description:
{description}

Return only valid JSON like: ["strong quantitative skills", "data-driven decision making"]"""

EXPLANATION_PROMPT = """You are a career advisor. Explain in 2-3 friendly, specific sentences why this job
is a good match for this candidate. Be honest about both strengths and gaps.

Candidate skills: {candidate_skills}
Job title: {job_title}
Matched skills: {matched_skills}
Missing skills: {missing_skills}
Semantic similarity: {similarity_pct}%

Write a 2-3 sentence explanation starting with "This role matches your profile because..." """


# ─── OpenAI client ────────────────────────────────────────────────────────────
def _get_client() -> OpenAI | None:
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set — LLM features disabled")
        return None
    return OpenAI(api_key=OPENAI_API_KEY)


def _call_llm(
    system: str,
    user: str,
    model: str = OPENAI_MODEL,
    max_tokens: int = 1800,
    temperature: float = 0.3,
) -> str | None:
    """Call OpenAI API with error handling and rate limit awareness."""
    client = _get_client()
    if client is None:
        return None
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return None


# ─── Key phrase extraction ────────────────────────────────────────────────────
def _extract_key_phrases(job_description: str) -> list[str]:
    """
    Extract 8-12 capability phrases from a job description (e.g. "strong quantitative skills").
    These are passed to the resume LLM so it can mirror the employer's language in bullets.
    Falls back to a regex heuristic if no API key is available.
    """
    prompt = KEY_PHRASES_PROMPT.format(description=job_description[:1500])
    result = _call_llm(
        "You are a job description analyst. Return only valid JSON arrays.",
        prompt,
        max_tokens=300,
        temperature=0.1,
    )
    if result:
        try:
            import re as _re
            match = _re.search(r'\[.*?\]', result, _re.DOTALL)
            if match:
                phrases = json.loads(match.group())
                return [p for p in phrases if isinstance(p, str)][:12]
        except Exception:
            pass

    # Fallback: pull noun phrases that follow "strong", "experience with",
    # "ability to", "proven", "demonstrated" — common JD patterns
    import re as _re
    patterns = [
        r'(?:strong|excellent|exceptional|demonstrated|proven)\s+[\w\s]{3,30}?(?=[\.,;])',
        r'experience (?:with|in)\s+[\w\s]{3,30}?(?=[\.,;])',
        r'ability to\s+[\w\s]{3,30}?(?=[\.,;])',
    ]
    phrases = []
    for pat in patterns:
        for m in _re.findall(pat, job_description[:2000], _re.IGNORECASE):
            cleaned = m.strip().lower()
            if 5 < len(cleaned) < 60:
                phrases.append(cleaned)
    return list(dict.fromkeys(phrases))[:10]  # dedupe, keep order


# ─── Resume generation ────────────────────────────────────────────────────────
def generate_resume(profile: dict, job: "RankedJob") -> dict:
    """
    Generate a tailored resume for a selected job.

    Returns:
        dict with keys:
          - 'markdown': the full resume as markdown string
          - 'method':   'ai' or 'template'
          - 'matched_skills': list of matched skills
          - 'missing_skills': list of missing skills
          - 'warning': fabrication warning string
    """
    matched = job.matched_skills or []
    missing = job.missing_skills or []

    # Extract capability key phrases from the job description so the LLM
    # can mirror them in each experience bullet (the core of inference-based rewriting)
    key_phrases = _extract_key_phrases(job.description)

    user_prompt = RESUME_USER_TEMPLATE.format(
        profile_json        = json.dumps({
            k: v for k, v in profile.items()
            if k not in ["id", "pass_criteria"]
        }, indent=2),
        job_title           = job.title,
        company             = job.company,
        location            = job.location,
        experience_required = job.experience_required or "Not specified",
        job_description     = job.description[:2500],   # more context for better inference
        key_phrases         = "\n".join(f"• {p}" for p in key_phrases),
        required_skills     = ", ".join(job.skills_extracted[:12]),
        matched_skills      = ", ".join(matched[:8]),
        missing_skills      = ", ".join(missing[:5]),
    )

    markdown = _call_llm(RESUME_SYSTEM_PROMPT, user_prompt, max_tokens=2500)  # larger budget

    if markdown is None:
        markdown = _template_resume(profile, job)
        method = "template"
    else:
        method = "ai"

    return {
        "markdown":      markdown,
        "method":        method,
        "matched_skills": matched,
        "missing_skills": missing,
        "warning": (
            "⚠️ JobPilot tailors wording and prioritization only. "
            "It does not invent degrees, certifications, employers, or experience. "
            "Always review before submitting."
        ),
    }


def generate_match_explanation(profile: dict, job: "RankedJob") -> str:
    """Generate a friendly 2-3 sentence explanation of why this job matches."""
    prompt = EXPLANATION_PROMPT.format(
        candidate_skills = ", ".join(profile.get("skills", [])[:10]),
        job_title        = job.title,
        matched_skills   = ", ".join(job.matched_skills[:5]),
        missing_skills   = ", ".join(job.missing_skills[:3]),
        similarity_pct   = int(job.embedding_score * 100),
    )

    result = _call_llm(
        "You are a helpful career advisor. Be concise and specific.",
        prompt,
        max_tokens=200,
        temperature=0.5,
    )

    if result:
        return result.strip()

    # Fallback
    return (
        f"This role matches your profile because you share {len(job.matched_skills)} key skills "
        f"including {', '.join(job.matched_skills[:3])}. "
        f"The {int(job.embedding_score * 100)}% semantic similarity score indicates strong overall alignment. "
        + (f"You'll want to build skills in {', '.join(job.missing_skills[:2])} to be fully competitive."
           if job.missing_skills else "Your skill set covers most requirements well.")
    )


def extract_skills_with_llm(job_description: str) -> list[str]:
    """Use LLM to extract required skills from a job description."""
    prompt = SKILL_EXTRACTION_PROMPT.format(description=job_description[:1200])
    result = _call_llm(
        "You are a skill extractor. Return only valid JSON arrays.",
        prompt,
        max_tokens=200,
        temperature=0.1,
    )
    if result:
        try:
            import re
            json_match = re.search(r'\[.*?\]', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception:
            pass
    return []


# ─── Template fallback (when no API key) ─────────────────────────────────────
def _template_resume(profile: dict, job: "RankedJob") -> str:
    """Generate a structured resume template without LLM (fallback)."""
    name     = profile.get("name", "Candidate")
    title    = profile.get("current_title", "Professional")
    skills   = profile.get("skills", [])
    resume   = profile.get("resume_text", "")
    edu      = profile.get("education", "")
    exp      = profile.get("years_experience", 0)
    target   = job.title
    company  = job.company

    # Prioritize matched skills first
    matched  = job.matched_skills or []
    all_skills = matched + [s for s in skills if s not in matched]

    return f"""# {name}
{profile.get('location_preference', 'United States')}

---

## Professional Summary
Results-driven {title} with {exp}+ years of experience seeking a {target} role at {company}.
Bringing expertise in {', '.join(all_skills[:4])} with a track record of delivering data-driven insights.
Eager to apply technical skills to {job.description[:100].split('.')[0].lower()}.

---

## Skills

**Core Technical:** {' • '.join(all_skills[:10])}

**Additional:** {' • '.join(all_skills[10:16])}

---

## Experience
*(Based on your profile — tailor bullet points to match {company}'s job description)*

**{title}**
Previous Employer | [Duration]
- Applied {', '.join(matched[:3])} to solve business problems
- Delivered data analysis projects with measurable impact
- Collaborated cross-functionally to drive analytical initiatives

---

## Education
{edu}

---

## Notes
*This resume was generated from your profile using JobPilot's template engine (no AI). *
*For best results, add your OpenAI API key to enable AI-powered resume tailoring.*
*Missing skills to consider highlighting: {', '.join(job.missing_skills[:4])}*
"""
