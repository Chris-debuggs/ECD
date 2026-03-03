"""
trajectory_engine.py — Developmental Trajectory Engine.

Computes:
  - Developmental coefficient (z-score vs age-cohort mean risk score)
  - Risk transition detection between screening time points
  - Per-child trajectory records for dashboard display
"""

import json
import uuid
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

from schema import TIER_LABELS


def compute_developmental_coefficient(
    risk_prob: float,
    domain: str,
    age_months: int,
    cohort_stats: dict,
) -> float:
    """
    Developmental coefficient = standardised risk score vs age-cohort mean.
    Negative = better than cohort mean. Positive = worse than cohort mean.

    Args:
        risk_prob:     P(High) + P(Critical) for this domain assessment.
        domain:        'cognitive' | 'motor' | 'socio_emotional'
        age_months:    Child's age at assessment.
        cohort_stats:  Dict from model artifact: {domain: {mean, std}}.

    Returns:
        z-score (float). Negative is good (lower risk than cohort average).
    """
    stats = cohort_stats.get(domain, {"mean": 0.3, "std": 0.2})
    z = (risk_prob - stats["mean"]) / stats["std"]
    return round(float(z), 4)


def detect_risk_transition(
    prev_tier: Optional[int],
    curr_tier: int,
) -> Optional[str]:
    """
    Returns transition string if tier changed, else None.
    e.g. "Medium→Critical" or "High→Low"
    """
    if prev_tier is None:
        return None
    if prev_tier == curr_tier:
        return None
    return f"{TIER_LABELS[prev_tier]}→{TIER_LABELS[curr_tier]}"


def build_child_trajectory(
    child_id: str,
    assessments_with_results: list[dict],
    cohort_stats: dict,
) -> list[dict]:
    """
    Build a list of trajectory records for one child across all time points.

    Args:
        child_id:               Child's ID.
        assessments_with_results: List of dicts, each with:
            {assessment_id, assessment_date, age_at_assessment_months,
             cognitive_tier, motor_tier, socio_tier, overall_tier,
             cognitive_prob, motor_prob, socio_prob, overall_prob}
            Sorted oldest-first.
        cohort_stats:           From model artifact.

    Returns:
        List of trajectory record dicts (one per assessment).
    """
    records = []
    prev_overall_tier = None

    # Sort by date
    sorted_assessments = sorted(
        assessments_with_results,
        key=lambda a: a.get("assessment_date", "")
    )

    for i, asmnt in enumerate(sorted_assessments):
        curr_tier = asmnt.get("overall_tier", 0)
        transition = detect_risk_transition(prev_overall_tier, curr_tier)

        # Compute developmental coefficients per domain
        # Use P(High) + P(Critical) as risk score; fallback to tier/3 if prob unavailable
        def _risk_score(prob, tier):
            if prob is not None and prob > 0:
                return float(prob)
            return float(tier) / 3.0  # normalised tier as fallback

        cog_score = _risk_score(asmnt.get("cognitive_prob"), asmnt.get("cognitive_tier", 0))
        mot_score = _risk_score(asmnt.get("motor_prob"), asmnt.get("motor_tier", 0))
        se_score  = _risk_score(asmnt.get("socio_prob"), asmnt.get("socio_tier", 0))
        ov_score  = _risk_score(asmnt.get("overall_prob"), curr_tier)

        cog_coeff = compute_developmental_coefficient(cog_score, "cognitive", asmnt.get("age_at_assessment_months", 24), cohort_stats)
        mot_coeff = compute_developmental_coefficient(mot_score, "motor", asmnt.get("age_at_assessment_months", 24), cohort_stats)
        se_coeff  = compute_developmental_coefficient(se_score, "socio_emotional", asmnt.get("age_at_assessment_months", 24), cohort_stats)
        ov_coeff  = round((cog_coeff + mot_coeff + se_coeff) / 3.0, 4)

        record = {
            "trajectory_id":   str(uuid.uuid4()),
            "child_id":        child_id,
            "assessment_id":   asmnt.get("assessment_id", ""),
            "time_point":      asmnt.get("assessment_date", ""),
            "age_months":      asmnt.get("age_at_assessment_months", 0),
            "cognitive_coeff": cog_coeff,
            "motor_coeff":     mot_coeff,
            "socio_coeff":     se_coeff,
            "overall_coeff":   ov_coeff,
            "risk_transition": transition,
            "overall_tier":    curr_tier,
            "computed_at":     datetime.now().isoformat(),
        }
        records.append(record)
        prev_overall_tier = curr_tier

    return records


def compute_all_trajectories(
    children_df: pd.DataFrame,
    assessments_df: pd.DataFrame,
    risk_results_df: pd.DataFrame,
    cohort_stats: dict,
) -> pd.DataFrame:
    """
    Compute trajectories for all children.

    Args:
        children_df:     DataFrame with child records.
        assessments_df:  DataFrame with assessments.
        risk_results_df: DataFrame with risk_results (joined to assessments).
        cohort_stats:    From model artifact.

    Returns:
        DataFrame of trajectory records.
    """
    # Join assessments + risk results
    merged = assessments_df.merge(
        risk_results_df[["assessment_id", "cognitive_tier", "motor_tier",
                         "socio_tier", "overall_tier",
                         "cognitive_prob", "motor_prob", "socio_prob", "overall_prob"]],
        on="assessment_id",
        how="left"
    )

    all_records = []
    for child_id in children_df["child_id"]:
        child_assessments = merged[merged["child_id"] == child_id].to_dict("records")
        if not child_assessments:
            continue
        records = build_child_trajectory(child_id, child_assessments, cohort_stats)
        all_records.extend(records)

    return pd.DataFrame(all_records) if all_records else pd.DataFrame()


def trajectory_summary(trajectories_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a per-child trajectory summary for the dashboard.

    Returns DataFrame with columns:
        child_id, n_screenings, first_tier, latest_tier,
        latest_overall_coeff, has_transition, transition_label,
        trajectory_direction  (IMPROVING / STABLE / DETERIORATING)
    """
    if trajectories_df.empty:
        return pd.DataFrame()

    summaries = []
    for child_id, grp in trajectories_df.groupby("child_id"):
        grp_sorted = grp.sort_values("time_point")
        first = grp_sorted.iloc[0]
        latest = grp_sorted.iloc[-1]
        transitions = grp_sorted["risk_transition"].dropna().tolist()

        first_tier  = int(first["overall_tier"])
        latest_tier = int(latest["overall_tier"])

        if latest_tier < first_tier:
            direction = "IMPROVING"
        elif latest_tier > first_tier:
            direction = "DETERIORATING"
        else:
            direction = "STABLE"

        summaries.append({
            "child_id":              child_id,
            "n_screenings":          len(grp_sorted),
            "first_tier":            first_tier,
            "latest_tier":           latest_tier,
            "latest_overall_coeff":  round(float(latest["overall_coeff"]), 3),
            "has_transition":        len(transitions) > 0,
            "transition_label":      " | ".join(transitions) if transitions else "—",
            "trajectory_direction":  direction,
        })

    return pd.DataFrame(summaries)


# --- Metrics for demo narrative ---

def impact_metrics(trajectories_df: pd.DataFrame) -> dict:
    """
    Compute programme-level impact metrics for governor/CDPO view.
    """
    if trajectories_df.empty:
        return {}

    summary = trajectory_summary(trajectories_df)

    n_total = len(summary)
    n_deteriorating = (summary["trajectory_direction"] == "DETERIORATING").sum()
    n_improving     = (summary["trajectory_direction"] == "IMPROVING").sum()
    n_stable        = (summary["trajectory_direction"] == "STABLE").sum()
    n_high_critical = (summary["latest_tier"] >= 2).sum()
    n_transitions   = summary["has_transition"].sum()

    mean_coeff = summary["latest_overall_coeff"].mean()

    return {
        "n_children_tracked":      n_total,
        "n_high_critical_now":     int(n_high_critical),
        "pct_high_critical":       round(100 * n_high_critical / n_total, 1) if n_total else 0,
        "n_deteriorating":         int(n_deteriorating),
        "n_improving":             int(n_improving),
        "n_stable":                int(n_stable),
        "n_transitions_detected":  int(n_transitions),
        "mean_overall_coeff":      round(float(mean_coeff), 3),
    }


if __name__ == "__main__":
    # Quick self-test
    from synthetic_data_generator import generate_children_and_assessments, DEMO_OVERRIDES

    print("[trajectory_engine] Self-test with synthetic data...")
    children_df, assessments_df = generate_children_and_assessments(
        n_children=12, n_time_points=2, demo_overrides=DEMO_OVERRIDES
    )

    # Mock risk results using true_tier as proxy
    assessments_df["cognitive_tier"] = assessments_df["true_tier"]
    assessments_df["motor_tier"]     = assessments_df["true_tier"]
    assessments_df["socio_tier"]     = assessments_df["true_tier"]
    assessments_df["overall_tier"]   = assessments_df["true_tier"]
    assessments_df["cognitive_prob"] = assessments_df["true_tier"] / 4.0
    assessments_df["motor_prob"]     = assessments_df["true_tier"] / 4.0
    assessments_df["socio_prob"]     = assessments_df["true_tier"] / 4.0
    assessments_df["overall_prob"]   = assessments_df["true_tier"] / 4.0
    assessments_df["result_id"]      = [str(uuid.uuid4()) for _ in range(len(assessments_df))]

    mock_cohort_stats = {d: {"mean": 0.3, "std": 0.2}
                         for d in ["cognitive", "motor", "socio_emotional"]}

    traj_df = compute_all_trajectories(
        children_df, assessments_df, assessments_df, mock_cohort_stats
    )
    print(f"Trajectories computed: {len(traj_df)} records for {traj_df['child_id'].nunique()} children")

    summary = trajectory_summary(traj_df)
    print("\nTrajectory Summary:")
    print(summary[["child_id", "first_tier", "latest_tier", "trajectory_direction", "transition_label"]].to_string())

    metrics = impact_metrics(traj_df)
    print("\nImpact Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
