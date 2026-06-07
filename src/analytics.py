"""
analytics.py — Batch market analytics and benchmark computation for JobPilot.

Generates aggregate insights from the full job corpus and benchmark tables
for the Technical Brief and in-app Benchmarks page.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

from src.utils import logger
from src.ranker import RankedJob


# ─── Market Analytics ─────────────────────────────────────────────────────────
def get_top_skills(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Count skill frequency across all job postings."""
    skill_counts = Counter()
    for skills in df["skills_extracted"].dropna():
        if isinstance(skills, list):
            skill_counts.update(s.lower() for s in skills)
        elif isinstance(skills, str) and skills.startswith("["):
            import ast
            try:
                skill_counts.update(s.lower() for s in ast.literal_eval(skills))
            except Exception:
                pass

    df_skills = pd.DataFrame(skill_counts.most_common(top_n),
                              columns=["skill", "count"])
    df_skills["skill"] = df_skills["skill"].str.title()
    return df_skills


def get_salary_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return salary midpoints for jobs that have salary data."""
    sal_df = df[df["salary_midpoint"] > 10000].copy()
    sal_df["salary_k"] = sal_df["salary_midpoint"] / 1000
    return sal_df[["title", "seniority", "salary_k", "remote"]].dropna()


def get_remote_distribution(df: pd.DataFrame) -> dict:
    """Count remote vs onsite vs hybrid."""
    remote_count  = int(df["remote"].sum())
    onsite_count  = len(df) - remote_count
    return {
        "Remote":   remote_count,
        "On-site":  onsite_count,
    }


def get_top_companies(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Most frequently hiring companies."""
    counts = df["company"].value_counts().head(top_n).reset_index()
    counts.columns = ["company", "job_count"]
    return counts


def get_top_job_titles(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Most common job titles."""
    counts = df["title"].value_counts().head(top_n).reset_index()
    counts.columns = ["title", "count"]
    return counts


def get_location_distribution(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Most common cities/locations."""
    counts = df["city"].value_counts().head(top_n).reset_index()
    counts.columns = ["city", "count"]
    return counts[counts["city"].str.strip() != ""]


def get_skill_gaps(profile: dict, df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Find skills frequently required in jobs matching the user's target roles
    that the user does NOT have.
    """
    user_skills = set(s.lower() for s in profile.get("skills", []))
    target_roles = [r.lower() for r in profile.get("target_roles", [])]

    # Filter to relevant jobs
    if target_roles:
        mask = df["title"].str.lower().apply(
            lambda t: any(role in t for role in target_roles)
        )
        relevant = df[mask]
    else:
        relevant = df

    # Count skills in relevant jobs
    gap_counter = Counter()
    for skills in relevant["skills_extracted"].dropna():
        if isinstance(skills, list):
            for s in skills:
                if s.lower() not in user_skills:
                    gap_counter[s.lower()] += 1
        elif isinstance(skills, str) and skills.startswith("["):
            import ast
            try:
                for s in ast.literal_eval(skills):
                    if s.lower() not in user_skills:
                        gap_counter[s.lower()] += 1
            except Exception:
                pass

    gaps_df = pd.DataFrame(gap_counter.most_common(top_n),
                            columns=["skill", "frequency"])
    gaps_df["skill"] = gaps_df["skill"].str.title()
    return gaps_df


def get_seniority_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["seniority"].value_counts().reset_index()
    counts.columns = ["seniority", "count"]
    counts["seniority"] = counts["seniority"].str.title()
    return counts


# ─── Plotly chart builders ────────────────────────────────────────────────────
BRAND_COLORS = ["#1F4E79", "#2E75B6", "#5BA3D0", "#9EC5E8", "#C6DDF0",
                "#27AE60", "#F39C12", "#E74C3C", "#8E44AD", "#16A085"]

def plot_top_skills(skills_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        skills_df.sort_values("count"),
        x="count", y="skill", orientation="h",
        title="Top In-Demand Skills",
        color="count",
        color_continuous_scale=["#C6DDF0", "#1F4E79"],
        labels={"count": "Job Postings", "skill": ""},
    )
    fig.update_layout(
        showlegend=False, coloraxis_showscale=False,
        height=500, plot_bgcolor="white",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def plot_salary_distribution(sal_df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        sal_df, x="seniority", y="salary_k",
        title="Salary Distribution by Seniority (USD, thousands)",
        color="seniority",
        color_discrete_sequence=BRAND_COLORS,
        labels={"salary_k": "Salary ($K)", "seniority": "Seniority Level"},
        category_orders={"seniority": ["Junior", "Mid", "Senior", "Staff"]},
    )
    fig.update_layout(
        showlegend=False, plot_bgcolor="white",
        height=400, margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def plot_remote_pie(remote_dist: dict) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=list(remote_dist.keys()),
        values=list(remote_dist.values()),
        marker_colors=[BRAND_COLORS[0], BRAND_COLORS[3]],
        hole=0.4,
        textinfo="label+percent",
    ))
    fig.update_layout(
        title="Remote vs On-site Split",
        height=350,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def plot_top_companies(companies_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        companies_df.sort_values("job_count"),
        x="job_count", y="company", orientation="h",
        title="Top Hiring Companies",
        color="job_count",
        color_continuous_scale=["#C6DDF0", "#1F4E79"],
        labels={"job_count": "Open Positions", "company": ""},
    )
    fig.update_layout(
        showlegend=False, coloraxis_showscale=False,
        height=450, plot_bgcolor="white",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def plot_skill_gaps(gaps_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        gaps_df.sort_values("frequency"),
        x="frequency", y="skill", orientation="h",
        title="Your Top Skill Gaps (in Target Roles)",
        color="frequency",
        color_continuous_scale=["#F9E4B7", "#E74C3C"],
        labels={"frequency": "Jobs Requiring This Skill", "skill": ""},
    )
    fig.update_layout(
        showlegend=False, coloraxis_showscale=False,
        height=400, plot_bgcolor="white",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def plot_adaptive_learning_curve(precision_history: list[float]) -> go.Figure:
    rounds = list(range(len(precision_history)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rounds, y=precision_history,
        mode="lines+markers+text",
        name="Precision@5",
        line=dict(color="#1F4E79", width=3),
        marker=dict(size=10, color="#2E75B6"),
        text=[f"{p:.0%}" for p in precision_history],
        textposition="top center",
    ))
    fig.update_layout(
        title="Adaptive Learning: Precision@5 Improvement Over Feedback Rounds",
        xaxis_title="Feedback Round",
        yaxis_title="Precision@5",
        yaxis=dict(range=[0, 1.05], tickformat=".0%"),
        plot_bgcolor="white",
        height=350,
        margin=dict(l=10, r=10, t=60, b=40),
    )
    return fig


def plot_weight_evolution(weight_history: list[dict]) -> go.Figure:
    if len(weight_history) < 2:
        return go.Figure()
    df = pd.DataFrame(weight_history)
    df["round"] = range(len(df))
    fig = go.Figure()
    for col in df.columns:
        if col == "round":
            continue
        fig.add_trace(go.Scatter(
            x=df["round"], y=df[col], mode="lines+markers",
            name=col.replace("_", " ").title(),
        ))
    fig.update_layout(
        title="Ranking Weight Evolution (Adaptive Learning)",
        xaxis_title="Update Round",
        yaxis_title="Weight",
        plot_bgcolor="white",
        height=350,
        margin=dict(l=10, r=10, t=60, b=40),
    )
    return fig


def plot_benchmark_table(benchmark_data: dict) -> go.Figure:
    """Render a benchmark comparison as a styled Plotly table."""
    if not benchmark_data:
        return go.Figure()

    headers = list(benchmark_data.keys())
    rows    = [benchmark_data[h] for h in headers]

    fig = go.Figure(go.Table(
        header=dict(
            values=[h.replace("_", " ").title() for h in headers],
            fill_color="#1F4E79",
            font=dict(color="white", size=13),
            align="center",
            height=35,
        ),
        cells=dict(
            values=rows,
            fill_color=[["#F0F6FC" if i % 2 == 0 else "white" for i in range(len(rows[0]))]],
            font=dict(size=12),
            align="center",
            height=30,
        ),
    ))
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=250)
    return fig


# ─── Full analytics bundle ────────────────────────────────────────────────────
def get_full_analytics(df: pd.DataFrame, profile: dict) -> dict:
    """Compute all analytics metrics in one call."""
    return {
        "total_jobs":         len(df),
        "with_salary":        int((df["salary_midpoint"] > 0).sum()),
        "remote_count":       int(df["remote"].sum()),
        "top_skills":         get_top_skills(df),
        "salary_dist":        get_salary_distribution(df),
        "remote_dist":        get_remote_distribution(df),
        "top_companies":      get_top_companies(df),
        "top_titles":         get_top_job_titles(df),
        "location_dist":      get_location_distribution(df),
        "skill_gaps":         get_skill_gaps(profile, df),
        "seniority_dist":     get_seniority_distribution(df),
    }
