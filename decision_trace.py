"""
decision_trace.py — Governance decision trace logger.

Logs every risk computation, referral action, and consent event
to the decision_trace table. Provides read-back for the Audit Trail view.
"""

import uuid
import sqlite3
from datetime import datetime
from typing import Optional

from schema import get_connection, DB_PATH, TIER_LABELS


def log_event(
    action: str,
    role: str,
    child_id: Optional[str] = None,
    assessment_id: Optional[str] = None,
    overall_tier: Optional[int] = None,
    top_factor: Optional[str] = None,
    shap_magnitude: Optional[float] = None,
    is_provisional: bool = False,
    notes: Optional[str] = None,
    db_path: str = DB_PATH,
) -> str:
    """
    Write a governance trace entry.

    Returns: trace_id (str)
    """
    trace_id = str(uuid.uuid4())
    conn = get_connection(db_path)
    conn.execute(
        """INSERT INTO decision_trace
           (trace_id, timestamp, role, child_id, assessment_id,
            action, overall_tier, top_factor, shap_magnitude,
            is_provisional, notes)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (
            trace_id,
            datetime.now().isoformat(),
            role,
            child_id,
            assessment_id,
            action,
            overall_tier,
            top_factor,
            shap_magnitude,
            int(is_provisional),
            notes,
        )
    )
    conn.commit()
    conn.close()
    return trace_id


def log_screening(
    role: str,
    child_id: str,
    assessment_id: str,
    overall_tier: int,
    top_factors: list,
    is_provisional: bool = False,
    db_path: str = DB_PATH,
) -> str:
    """Convenience: log a completed risk screening event."""
    top_factor_label = top_factors[0]["label"] if top_factors else None
    shap_mag = top_factors[0]["magnitude"] if top_factors else None
    return log_event(
        action="SCREEN",
        role=role,
        child_id=child_id,
        assessment_id=assessment_id,
        overall_tier=overall_tier,
        top_factor=top_factor_label,
        shap_magnitude=shap_mag,
        is_provisional=is_provisional,
        notes=f"Risk tier: {TIER_LABELS.get(overall_tier, '?')}",
        db_path=db_path,
    )


def log_referral(
    role: str,
    child_id: str,
    assessment_id: str,
    action_type: str,
    overall_tier: int,
    db_path: str = DB_PATH,
) -> str:
    """Convenience: log a referral action."""
    return log_event(
        action="REFERRAL_CREATE",
        role=role,
        child_id=child_id,
        assessment_id=assessment_id,
        overall_tier=overall_tier,
        notes=f"Referral type: {action_type}",
        db_path=db_path,
    )


def log_consent(
    role: str,
    child_id: str,
    guardian_name: str,
    db_path: str = DB_PATH,
) -> str:
    """Convenience: log a consent capture event."""
    return log_event(
        action="CONSENT_CAPTURE",
        role=role,
        child_id=child_id,
        notes=f"Consent captured for guardian: {guardian_name}",
        db_path=db_path,
    )


def get_trace_log(
    limit: int = 200,
    db_path: str = DB_PATH,
) -> list[dict]:
    """Return the most recent trace log entries as a list of dicts."""
    conn = get_connection(db_path)
    rows = conn.execute(
        """SELECT trace_id, timestamp, role, child_id,
                  action, overall_tier, top_factor, shap_magnitude,
                  is_provisional, notes
           FROM decision_trace
           ORDER BY timestamp DESC
           LIMIT ?""",
        (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_trace_as_display_df(db_path: str = DB_PATH):
    """Return trace log as a pandas DataFrame ready for display."""
    import pandas as pd
    rows = get_trace_log(db_path=db_path)
    if not rows:
        return pd.DataFrame(columns=[
            "Timestamp", "Role", "Action", "Child ID",
            "Tier", "Top Risk Factor", "Provisional", "Notes"
        ])
    df = pd.DataFrame(rows)
    df["Timestamp"]      = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    df["Role"]           = df["role"]
    df["Action"]         = df["action"]
    df["Child ID"]       = df["child_id"].fillna("—")
    df["Tier"]           = df["overall_tier"].map(lambda t: TIER_LABELS.get(int(t), "—") if t is not None else "—")
    df["Top Risk Factor"] = df["top_factor"].fillna("—")
    df["Provisional"]    = df["is_provisional"].map(lambda x: "Yes" if x else "No")
    df["Notes"]          = df["notes"].fillna("—")
    return df[["Timestamp", "Role", "Action", "Child ID", "Tier", "Top Risk Factor", "Provisional", "Notes"]]
