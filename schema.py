"""
schema.py — SQLite schema creation for ECD Decision Support Layer MVP
Run this before seeding. Tables match the production PostgreSQL schema structure.
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "demo.db")


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def create_schema(db_path: str = DB_PATH) -> None:
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.executescript("""
    -- Children master table
    CREATE TABLE IF NOT EXISTS children (
        child_id        TEXT PRIMARY KEY,
        name            TEXT NOT NULL,
        dob             TEXT NOT NULL,
        age_months      INTEGER NOT NULL,
        gender          TEXT CHECK(gender IN ('M','F','O')) NOT NULL,
        block_id        TEXT NOT NULL,
        anganwadi_id    TEXT NOT NULL,
        created_at      TEXT NOT NULL
    );

    -- Screening assessments (one per visit)
    CREATE TABLE IF NOT EXISTS assessments (
        assessment_id           TEXT PRIMARY KEY,
        child_id                TEXT NOT NULL REFERENCES children(child_id),
        assessment_date         TEXT NOT NULL,
        age_at_assessment_months INTEGER NOT NULL,
        -- Cognitive domain indicators (0/1 or 1-4)
        cog_lang_milestone      INTEGER,   -- Language milestone achieved (0/1)
        cog_memory_recall       INTEGER,   -- Simple memory task score (1-4)
        cog_problem_solving     INTEGER,   -- Problem solving score (1-4)
        cog_attention_span      INTEGER,   -- Attention task completion (0/1)
        cog_learning_adapt      INTEGER,   -- Adapts to new task (0/1)
        -- Motor domain indicators
        mot_gross_motor         INTEGER,   -- Gross motor milestone (0/1)
        mot_fine_motor          INTEGER,   -- Fine motor skill (1-4)
        mot_balance             INTEGER,   -- Balance task (0/1)
        mot_hand_eye            INTEGER,   -- Hand-eye coordination (1-4)
        mot_body_aware          INTEGER,   -- Body awareness (0/1)
        -- Socio-Emotional domain indicators
        se_social_play          INTEGER,   -- Age-appropriate play (0/1)
        se_emotion_reg          INTEGER,   -- Emotion regulation (1-4)
        se_peer_interact        INTEGER,   -- Peer interaction quality (1-4)
        se_attachment           INTEGER,   -- Secure attachment (0/1)
        se_self_care            INTEGER,   -- Self-care skills (0/1)
        -- Metadata
        is_offline              INTEGER DEFAULT 0,
        created_at              TEXT NOT NULL
    );

    -- Risk results (one per assessment, per domain)
    CREATE TABLE IF NOT EXISTS risk_results (
        result_id           TEXT PRIMARY KEY,
        assessment_id       TEXT NOT NULL REFERENCES assessments(assessment_id),
        child_id            TEXT NOT NULL REFERENCES children(child_id),
        model_version       TEXT NOT NULL,
        -- Tier: 0=Low, 1=Medium, 2=High, 3=Critical
        cognitive_tier      INTEGER,
        motor_tier          INTEGER,
        socio_tier          INTEGER,
        overall_tier        INTEGER,
        -- Calibrated probabilities (for highest-risk class)
        cognitive_prob      REAL,
        motor_prob          REAL,
        socio_prob          REAL,
        overall_prob        REAL,
        -- Explainability (stored as JSON strings)
        shap_values_json    TEXT,
        top_factors_json    TEXT,   -- [{"feature": "...", "label": "...", "direction": "+/-", "value": float}]
        is_provisional      INTEGER DEFAULT 0,
        computed_at         TEXT NOT NULL
    );

    -- Developmental trajectories (per child, per time point)
    CREATE TABLE IF NOT EXISTS developmental_trajectories (
        trajectory_id       TEXT PRIMARY KEY,
        child_id            TEXT NOT NULL REFERENCES children(child_id),
        assessment_id       TEXT NOT NULL REFERENCES assessments(assessment_id),
        time_point          TEXT NOT NULL,   -- ISO date of assessment
        age_months          INTEGER,
        cognitive_coeff     REAL,    -- z-score vs age-cohort mean
        motor_coeff         REAL,
        socio_coeff         REAL,
        overall_coeff       REAL,
        risk_transition     TEXT,    -- e.g. "Medium→High" or NULL if first point
        overall_tier        INTEGER,
        computed_at         TEXT NOT NULL
    );

    -- Consent records
    CREATE TABLE IF NOT EXISTS consents (
        consent_id          TEXT PRIMARY KEY,
        child_id            TEXT NOT NULL REFERENCES children(child_id),
        guardian_name       TEXT NOT NULL,
        consent_date        TEXT NOT NULL,
        consent_method      TEXT CHECK(consent_method IN ('VERBAL_WITNESSED','THUMBPRINT','WRITTEN')) NOT NULL,
        aww_witness         TEXT NOT NULL,
        created_at          TEXT NOT NULL
    );

    -- Decision trace log (governance)
    CREATE TABLE IF NOT EXISTS decision_trace (
        trace_id        TEXT PRIMARY KEY,
        timestamp       TEXT NOT NULL,
        role            TEXT NOT NULL,       -- AWW / SUPERVISOR / SYSTEM
        child_id        TEXT,
        assessment_id   TEXT,
        action          TEXT NOT NULL,       -- SCREEN / REFERRAL_CREATE / CONSENT_CAPTURE / SYNC / OVERRIDE
        overall_tier    INTEGER,
        top_factor      TEXT,
        shap_magnitude  REAL,
        is_provisional  INTEGER DEFAULT 0,
        notes           TEXT
    );

    -- Referrals
    CREATE TABLE IF NOT EXISTS referrals (
        referral_id     TEXT PRIMARY KEY,
        child_id        TEXT NOT NULL REFERENCES children(child_id),
        assessment_id   TEXT NOT NULL REFERENCES assessments(assessment_id),
        created_at      TEXT NOT NULL,
        action_type     TEXT NOT NULL,   -- HOME_VISIT / PHC_REFERRAL / NRC_REFERRAL
        notes           TEXT,
        outcome_code    INTEGER,         -- 0=pending, 1=attended-no concern, 2=mild, 3=moderate, 4=significant
        outcome_date    TEXT
    );

    -- Indexes for common query patterns
    CREATE INDEX IF NOT EXISTS idx_assessments_child   ON assessments(child_id, assessment_date);
    CREATE INDEX IF NOT EXISTS idx_risk_child          ON risk_results(child_id, computed_at);
    CREATE INDEX IF NOT EXISTS idx_trajectory_child    ON developmental_trajectories(child_id, time_point);
    CREATE INDEX IF NOT EXISTS idx_trace_timestamp     ON decision_trace(timestamp);
    """)

    conn.commit()
    conn.close()
    print(f"[schema] Database schema created at {db_path}")


TIER_LABELS = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}
TIER_COLOURS = {0: "#27AE60", 1: "#F39C12", 2: "#E74C3C", 3: "#8E44AD"}

FEATURE_LABELS = {
    "cog_lang_milestone":  "Language milestone",
    "cog_memory_recall":   "Memory recall",
    "cog_problem_solving": "Problem solving",
    "cog_attention_span":  "Attention span",
    "cog_learning_adapt":  "Learning adaptation",
    "mot_gross_motor":     "Gross motor skills",
    "mot_fine_motor":      "Fine motor skills",
    "mot_balance":         "Balance & coordination",
    "mot_hand_eye":        "Hand-eye coordination",
    "mot_body_aware":      "Body awareness",
    "se_social_play":      "Social play",
    "se_emotion_reg":      "Emotion regulation",
    "se_peer_interact":    "Peer interaction",
    "se_attachment":       "Caregiver attachment",
    "se_self_care":        "Self-care skills",
}

COGNITIVE_FEATURES = [
    "cog_lang_milestone", "cog_memory_recall", "cog_problem_solving",
    "cog_attention_span", "cog_learning_adapt"
]
MOTOR_FEATURES = [
    "mot_gross_motor", "mot_fine_motor", "mot_balance",
    "mot_hand_eye", "mot_body_aware"
]
SOCIO_FEATURES = [
    "se_social_play", "se_emotion_reg", "se_peer_interact",
    "se_attachment", "se_self_care"
]
ALL_FEATURES = COGNITIVE_FEATURES + MOTOR_FEATURES + SOCIO_FEATURES


if __name__ == "__main__":
    create_schema()
