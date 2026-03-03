"""
seed_demo_db.py — Populate demo.db with synthetic data and pre-run risk + trajectory.

Run once: python seed_demo_db.py
Resets and re-seeds the database.

Demo narrative:
  ECD-0001: Stable Low risk  (normal trajectory — shows system works)
  ECD-0002: High→Medium      (improving — shows positive trajectory)
  ECD-0003: Medium→Critical  (KEY DEMO — system catches deterioration)
"""

import os
import uuid
import json
import sqlite3
from datetime import datetime, date, timedelta

print("[seed] Starting database seed...")

from schema import create_schema, get_connection, DB_PATH, TIER_LABELS
from synthetic_data_generator import (
    generate_children_and_assessments, DEMO_OVERRIDES
)
from risk_engine import train_and_save, predict, load_model, MODEL_PATH
from trajectory_engine import (
    build_child_trajectory, compute_all_trajectories, trajectory_summary
)
from decision_trace import log_screening, log_consent


# ── Step 1: Reset database ──────────────────────────────────────────────────

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
    print(f"[seed] Removed existing {DB_PATH}")

create_schema(DB_PATH)
print(f"[seed] Schema created at {DB_PATH}")


# ── Step 2: Train or load model ─────────────────────────────────────────────

if not os.path.exists(MODEL_PATH):
    print("[seed] Training risk model (first run)...")
    artifact = train_and_save(force_retrain=False)
else:
    print("[seed] Loading existing model...")
    artifact = load_model()

cohort_stats = artifact["cohort_stats"]
print(f"[seed] Model version: {artifact['model_version']}")


# ── Step 3: Generate synthetic children + assessments ───────────────────────

print("[seed] Generating 60 children, 2 time points each...")
children_df, assessments_df = generate_children_and_assessments(
    n_children=60,
    n_time_points=2,
    demo_overrides=DEMO_OVERRIDES,
)
print(f"[seed] Generated {len(children_df)} children, {len(assessments_df)} assessments")


# ── Step 4: Insert children ─────────────────────────────────────────────────

conn = get_connection(DB_PATH)

for _, child in children_df.iterrows():
    conn.execute(
        """INSERT OR IGNORE INTO children
           (child_id, name, dob, age_months, gender, block_id, anganwadi_id, created_at)
           VALUES (?,?,?,?,?,?,?,?)""",
        (
            child["child_id"], child["name"], child["dob"],
            int(child["age_months"]), child["gender"],
            child["block_id"], child["anganwadi_id"], child["created_at"]
        )
    )

# Add consent records for all children
for _, child in children_df.iterrows():
    conn.execute(
        """INSERT OR IGNORE INTO consents
           (consent_id, child_id, guardian_name, consent_date,
            consent_method, aww_witness, created_at)
           VALUES (?,?,?,?,?,?,?)""",
        (
            str(uuid.uuid4()), child["child_id"],
            f"Guardian of {child['name']}",
            child["created_at"][:10],
            "VERBAL_WITNESSED",
            "AWW Meena Sharma",
            child["created_at"]
        )
    )

conn.commit()
print(f"[seed] Inserted {len(children_df)} children and consent records")


# ── Step 5: Insert assessments + run risk engine ────────────────────────────

assessment_ids_written = []
risk_results_written = []

for _, row in assessments_df.iterrows():
    assessment_id = row["assessment_id"]
    child_id = row["child_id"]

    # Insert assessment
    conn.execute(
        """INSERT OR IGNORE INTO assessments
           (assessment_id, child_id, assessment_date, age_at_assessment_months,
            cog_lang_milestone, cog_memory_recall, cog_problem_solving,
            cog_attention_span, cog_learning_adapt,
            mot_gross_motor, mot_fine_motor, mot_balance, mot_hand_eye, mot_body_aware,
            se_social_play, se_emotion_reg, se_peer_interact, se_attachment, se_self_care,
            is_offline, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            assessment_id, child_id,
            row["assessment_date"], int(row["age_at_assessment_months"]),
            int(row["cog_lang_milestone"]), int(row["cog_memory_recall"]),
            int(row["cog_problem_solving"]), int(row["cog_attention_span"]),
            int(row["cog_learning_adapt"]),
            int(row["mot_gross_motor"]), int(row["mot_fine_motor"]),
            int(row["mot_balance"]), int(row["mot_hand_eye"]),
            int(row["mot_body_aware"]),
            int(row["se_social_play"]), int(row["se_emotion_reg"]),
            int(row["se_peer_interact"]), int(row["se_attachment"]),
            int(row["se_self_care"]),
            int(row.get("is_offline", 0)),
            row["assessment_date"]
        )
    )

    # Run risk engine
    result = predict(artifact, row.to_dict(), is_provisional=False)
    result_id = str(uuid.uuid4())

    conn.execute(
        """INSERT OR IGNORE INTO risk_results
           (result_id, assessment_id, child_id, model_version,
            cognitive_tier, motor_tier, socio_tier, overall_tier,
            cognitive_prob, motor_prob, socio_prob, overall_prob,
            shap_values_json, top_factors_json, is_provisional, computed_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            result_id, assessment_id, child_id, result["model_version"],
            int(result["cognitive_tier"]), int(result["motor_tier"]),
            int(result["socio_tier"]), int(result["overall_tier"]),
            float(result["cognitive_prob"] or 0),
            float(result["motor_prob"] or 0),
            float(result["socio_prob"] or 0),
            float(result["overall_prob"] or 0),
            result["shap_values_json"], result["top_factors_json"],
            int(result["is_provisional"]),
            datetime.now().isoformat()
        )
    )

    risk_results_written.append({
        "result_id":      result_id,
        "assessment_id":  assessment_id,
        "child_id":       child_id,
        "overall_tier":   result["overall_tier"],
        "cognitive_tier": result["cognitive_tier"],
        "motor_tier":     result["motor_tier"],
        "socio_tier":     result["socio_tier"],
        "overall_prob":   result["overall_prob"] or 0,
        "cognitive_prob": result["cognitive_prob"] or 0,
        "motor_prob":     result["motor_prob"] or 0,
        "socio_prob":     result["socio_prob"] or 0,
        "computed_at":    datetime.now().isoformat(),
        "model_version":  result["model_version"],
    })

    # Log to decision trace
    log_screening(
        role="SYSTEM_SEED",
        child_id=child_id,
        assessment_id=assessment_id,
        overall_tier=result["overall_tier"],
        top_factors=result.get("top_factors", []),
        is_provisional=False,
        db_path=DB_PATH,
    )
    assessment_ids_written.append(assessment_id)

conn.commit()
print(f"[seed] Inserted {len(assessment_ids_written)} assessments + risk results")


# ── Step 6: Compute trajectories ────────────────────────────────────────────

import pandas as pd

risk_df = pd.DataFrame(risk_results_written)
traj_df = compute_all_trajectories(
    children_df,
    assessments_df,
    risk_df,
    cohort_stats,
)

for _, traj in traj_df.iterrows():
    conn.execute(
        """INSERT OR IGNORE INTO developmental_trajectories
           (trajectory_id, child_id, assessment_id, time_point, age_months,
            cognitive_coeff, motor_coeff, socio_coeff, overall_coeff,
            risk_transition, overall_tier, computed_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            traj["trajectory_id"], traj["child_id"], traj["assessment_id"],
            traj["time_point"], int(traj["age_months"] or 0),
            float(traj["cognitive_coeff"] or 0),
            float(traj["motor_coeff"] or 0),
            float(traj["socio_coeff"] or 0),
            float(traj["overall_coeff"] or 0),
            traj.get("risk_transition"),
            int(traj["overall_tier"] or 0),
            traj["computed_at"]
        )
    )

conn.commit()
conn.close()
print(f"[seed] Inserted {len(traj_df)} trajectory records")


# ── Step 7: Verify demo children ────────────────────────────────────────────

conn = get_connection(DB_PATH)
print("\n[seed] === Demo Children Verification ===")
for demo_child in ["ECD-0001", "ECD-0002", "ECD-0003"]:
    traj_rows = conn.execute(
        """SELECT t.time_point, t.overall_tier, t.overall_coeff, t.risk_transition
           FROM developmental_trajectories t
           WHERE t.child_id = ?
           ORDER BY t.time_point""",
        (demo_child,)
    ).fetchall()
    child_row = conn.execute(
        "SELECT name, block_id FROM children WHERE child_id=?", (demo_child,)
    ).fetchone()
    name = child_row["name"] if child_row else "Unknown"
    print(f"\n  {demo_child} ({name}):")
    for r in traj_rows:
        tier_label = TIER_LABELS.get(r["overall_tier"], "?")
        transition = f" ← TRANSITION: {r['risk_transition']}" if r["risk_transition"] else ""
        print(f"    {r['time_point']}  Tier={tier_label}  Coeff={r['overall_coeff']:.3f}{transition}")
conn.close()

print("\n[seed] ✅ Database seeded successfully.")
print(f"[seed] Run: streamlit run app.py")
