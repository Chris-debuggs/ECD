"""
cohort_aggregator.py — Block and district-level cohort impact metrics.

Groups children by block/anganwadi and computes:
- Coverage rate (screened vs registered)
- Tier distribution
- Mean developmental coefficient
- High-risk count and trends
"""

import pandas as pd
import numpy as np
from typing import Optional

from schema import TIER_LABELS


def block_summary(
    children_df: pd.DataFrame,
    risk_results_df: pd.DataFrame,
    assessments_df: pd.DataFrame,
    trajectory_summary_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute block-level summary metrics.

    Returns:
        DataFrame with one row per block:
        block_id, n_registered, n_screened, coverage_pct,
        n_low, n_medium, n_high, n_critical,
        pct_high_critical, n_deteriorating, mean_coeff
    """
    # Latest risk result per child
    if risk_results_df.empty:
        return pd.DataFrame()

    latest_risk = (
        risk_results_df
        .sort_values("computed_at", ascending=False)
        .drop_duplicates(subset="child_id", keep="first")
        [["child_id", "overall_tier", "overall_prob"]]
    )

    # Merge with children to get block info
    child_block = children_df[["child_id", "block_id", "anganwadi_id"]].copy()
    merged = child_block.merge(latest_risk, on="child_id", how="left")

    # Merge trajectory summary if provided
    if trajectory_summary_df is not None and not trajectory_summary_df.empty:
        merged = merged.merge(
            trajectory_summary_df[["child_id", "trajectory_direction", "latest_overall_coeff"]],
            on="child_id",
            how="left"
        )
    else:
        merged["trajectory_direction"] = None
        merged["latest_overall_coeff"] = None

    rows = []
    for block_id, grp in merged.groupby("block_id"):
        n_registered = len(grp)
        n_screened   = grp["overall_tier"].notna().sum()
        screened_grp = grp[grp["overall_tier"].notna()]

        tier_counts = screened_grp["overall_tier"].value_counts()

        n_deteriorating = (
            screened_grp["trajectory_direction"] == "DETERIORATING"
        ).sum() if "trajectory_direction" in screened_grp.columns else 0

        mean_coeff = (
            screened_grp["latest_overall_coeff"].mean()
            if "latest_overall_coeff" in screened_grp.columns
            else None
        )

        n_high_critical = int(tier_counts.get(2, 0) + tier_counts.get(3, 0))

        rows.append({
            "block_id":         block_id,
            "n_registered":     n_registered,
            "n_screened":       int(n_screened),
            "coverage_pct":     round(100 * n_screened / n_registered, 1) if n_registered > 0 else 0,
            "n_low":            int(tier_counts.get(0, 0)),
            "n_medium":         int(tier_counts.get(1, 0)),
            "n_high":           int(tier_counts.get(2, 0)),
            "n_critical":       int(tier_counts.get(3, 0)),
            "n_high_critical":  n_high_critical,
            "pct_high_critical": round(100 * n_high_critical / n_screened, 1) if n_screened > 0 else 0,
            "n_deteriorating":  int(n_deteriorating),
            "mean_coeff":       round(float(mean_coeff), 3) if mean_coeff is not None and not np.isnan(mean_coeff) else None,
        })

    return pd.DataFrame(rows).sort_values("pct_high_critical", ascending=False)


def district_rollup(block_summary_df: pd.DataFrame) -> dict:
    """
    Aggregate block summary to district level.
    Returns a flat dict of district-level metrics.
    """
    if block_summary_df.empty:
        return {}

    total_registered = block_summary_df["n_registered"].sum()
    total_screened   = block_summary_df["n_screened"].sum()

    return {
        "n_blocks":           len(block_summary_df),
        "total_registered":   int(total_registered),
        "total_screened":     int(total_screened),
        "coverage_pct":       round(100 * total_screened / total_registered, 1) if total_registered > 0 else 0,
        "n_low":              int(block_summary_df["n_low"].sum()),
        "n_medium":           int(block_summary_df["n_medium"].sum()),
        "n_high":             int(block_summary_df["n_high"].sum()),
        "n_critical":         int(block_summary_df["n_critical"].sum()),
        "n_high_critical":    int(block_summary_df["n_high_critical"].sum()),
        "pct_high_critical":  round(100 * block_summary_df["n_high_critical"].sum() / total_screened, 1) if total_screened > 0 else 0,
        "n_deteriorating":    int(block_summary_df["n_deteriorating"].sum()),
        "highest_risk_block": str(block_summary_df.iloc[0]["block_id"]) if len(block_summary_df) > 0 else "—",
    }


def child_list_for_supervisor(
    children_df: pd.DataFrame,
    risk_results_df: pd.DataFrame,
    trajectory_summary_df: Optional[pd.DataFrame],
    block_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Returns a child-level table for the Supervisor dashboard.
    Optionally filtered to a specific block.
    """
    latest_risk = (
        risk_results_df
        .sort_values("computed_at", ascending=False)
        .drop_duplicates(subset="child_id", keep="first")
        [["child_id", "overall_tier", "cognitive_tier", "motor_tier", "socio_tier"]]
    ) if not risk_results_df.empty else pd.DataFrame(
        columns=["child_id", "overall_tier", "cognitive_tier", "motor_tier", "socio_tier"]
    )

    merged = children_df[["child_id", "name", "age_months", "gender", "block_id",
                           "anganwadi_id"]].merge(
        latest_risk, on="child_id", how="left"
    )

    if trajectory_summary_df is not None and not trajectory_summary_df.empty:
        merged = merged.merge(
            trajectory_summary_df[["child_id", "trajectory_direction",
                                   "latest_overall_coeff", "transition_label", "n_screenings"]],
            on="child_id",
            how="left"
        )
    else:
        merged["trajectory_direction"] = "NO DATA"
        merged["latest_overall_coeff"] = None
        merged["transition_label"] = "—"
        merged["n_screenings"] = 0

    if block_id:
        merged = merged[merged["block_id"] == block_id]

    # Add display columns
    merged["tier_label"] = merged["overall_tier"].map(
        lambda t: TIER_LABELS.get(int(t), "Unknown") if pd.notna(t) else "Not Screened"
    )
    merged["needs_referral"] = merged["overall_tier"].map(
        lambda t: t >= 2 if pd.notna(t) else False
    )

    return merged.sort_values(
        "overall_tier", ascending=False, na_position="last"
    ).reset_index(drop=True)


if __name__ == "__main__":
    print("[cohort_aggregator] Self-test...")
    from synthetic_data_generator import generate_children_and_assessments, DEMO_OVERRIDES
    import uuid

    children_df, assessments_df = generate_children_and_assessments(
        n_children=60, n_time_points=2, demo_overrides=DEMO_OVERRIDES
    )

    # Mock risk results
    risk_rows = []
    for _, row in assessments_df.iterrows():
        tier = row.get("true_tier", 0)
        risk_rows.append({
            "result_id":      str(uuid.uuid4()),
            "assessment_id":  row["assessment_id"],
            "child_id":       row["child_id"],
            "overall_tier":   tier,
            "cognitive_tier": tier,
            "motor_tier":     tier,
            "socio_tier":     tier,
            "overall_prob":   tier / 4.0,
            "cognitive_prob": tier / 4.0,
            "motor_prob":     tier / 4.0,
            "socio_prob":     tier / 4.0,
            "computed_at":    row["assessment_date"],
            "model_version":  "v1.0-test",
        })
    risk_df = pd.DataFrame(risk_rows)

    summary_df = block_summary(children_df, risk_df, assessments_df)
    print("Block Summary:")
    print(summary_df.to_string(index=False))

    district = district_rollup(summary_df)
    print("\nDistrict Rollup:")
    for k, v in district.items():
        print(f"  {k}: {v}")
