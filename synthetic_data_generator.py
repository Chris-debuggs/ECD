"""
synthetic_data_generator.py — Validated synthetic ECD dataset generator.

Design basis:
- WHO Motor Milestone reference ranges
- DASII (Delhi adaptation) cognitive norms
- Indicator distributions reflect real ECD screening variance
- Latent risk score approach: true risk drives indicator values + measurement noise

Output: pandas DataFrames for children and assessments.
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
import uuid
import warnings
warnings.filterwarnings("ignore")

# --- Reproducibility ---
SEED = 42

BLOCKS = ["Khandwa-Block-A", "Khandwa-Block-B", "Betul-Block-C", "Betul-Block-D"]
ANGANWADIS_PER_BLOCK = 3
CHILDREN_PER_AWC = 5  # ~60 total children

# Tier proportions (realistic ECD population)
TIER_PROBS = [0.50, 0.30, 0.13, 0.07]  # Low, Medium, High, Critical

# Domain-specific base rates for indicator risks per tier
# Each row = [Low, Medium, High, Critical] probability of indicator being "at risk"
INDICATOR_RISK_RATES = {
    # Cognitive
    "cog_lang_milestone":  [0.05, 0.20, 0.55, 0.85],
    "cog_memory_recall":   [0.08, 0.22, 0.50, 0.80],
    "cog_problem_solving": [0.06, 0.18, 0.48, 0.78],
    "cog_attention_span":  [0.07, 0.25, 0.52, 0.82],
    "cog_learning_adapt":  [0.05, 0.15, 0.45, 0.75],
    # Motor
    "mot_gross_motor":     [0.04, 0.18, 0.50, 0.80],
    "mot_fine_motor":      [0.06, 0.20, 0.48, 0.78],
    "mot_balance":         [0.05, 0.17, 0.45, 0.75],
    "mot_hand_eye":        [0.07, 0.22, 0.52, 0.82],
    "mot_body_aware":      [0.04, 0.15, 0.42, 0.72],
    # Socio-Emotional
    "se_social_play":      [0.06, 0.20, 0.48, 0.78],
    "se_emotion_reg":      [0.08, 0.25, 0.55, 0.85],
    "se_peer_interact":    [0.07, 0.22, 0.50, 0.80],
    "se_attachment":       [0.05, 0.18, 0.45, 0.75],
    "se_self_care":        [0.06, 0.20, 0.48, 0.78],
}

# Binary features (0=milestone met, 1=risk)
BINARY_FEATURES = ["cog_lang_milestone", "cog_attention_span", "cog_learning_adapt",
                   "mot_gross_motor", "mot_balance", "mot_body_aware",
                   "se_social_play", "se_attachment", "se_self_care"]

# Ordinal features (1-4 scale; lower = more risk)
ORDINAL_FEATURES = ["cog_memory_recall", "cog_problem_solving",
                    "mot_fine_motor", "mot_hand_eye",
                    "se_emotion_reg", "se_peer_interact"]


def _generate_indicator_value(feature: str, tier: int, rng: np.random.Generator) -> int:
    """Generate a single indicator value given a child's true risk tier."""
    risk_prob = INDICATOR_RISK_RATES[feature][tier]
    if feature in BINARY_FEATURES:
        # 0 = milestone met (good), 1 = not met (risk)
        return int(rng.random() < risk_prob)
    else:
        # Ordinal 1-4: risk_prob drives probability of scoring 1 or 2
        # Score 1 = severe concern, 4 = age-appropriate
        if rng.random() < risk_prob:
            return rng.choice([1, 2], p=[0.4, 0.6])
        else:
            return rng.choice([3, 4], p=[0.45, 0.55])


def _age_months_band(age_months: int) -> str:
    bands = [(12, "0-12m"), (24, "12-24m"), (36, "24-36m"),
             (48, "36-48m"), (60, "48-60m"), (72, "60-72m")]
    for cutoff, label in bands:
        if age_months <= cutoff:
            return label
    return "60-72m"


def generate_children_and_assessments(
    n_children: int = 60,
    n_time_points: int = 2,
    seed: int = SEED,
    demo_overrides: dict = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic children and assessments.

    Args:
        n_children:    Number of children to generate.
        n_time_points: Number of screening visits per child.
        seed:          RNG seed for reproducibility.
        demo_overrides: Dict of {child_id: {'tier_t1': X, 'tier_t2': Y}} for demo children.

    Returns:
        (children_df, assessments_df)
    """
    rng = np.random.default_rng(seed)
    base_date = date(2024, 6, 1)

    children_rows = []
    assessment_rows = []

    child_counter = 0
    for block_idx, block in enumerate(BLOCKS):
        for awc_num in range(ANGANWADIS_PER_BLOCK):
            anganwadi_id = f"{block[:3].upper()}-AWC-{awc_num + 1:02d}"
            per_awc = n_children // (len(BLOCKS) * ANGANWADIS_PER_BLOCK)
            for _ in range(per_awc):
                child_counter += 1
                child_id = f"ECD-{child_counter:04d}"

                # Check for demo override
                override = (demo_overrides or {}).get(child_id, {})
                true_tier_t1 = override.get("tier_t1", int(rng.choice([0, 1, 2, 3], p=TIER_PROBS)))
                true_tier_t2 = override.get("tier_t2", _next_tier(true_tier_t1, rng))

                age_months_t1 = int(rng.integers(12, 60))
                dob = base_date - timedelta(days=age_months_t1 * 30)
                gender = rng.choice(["M", "F"])
                name = _generate_name(gender, rng)

                children_rows.append({
                    "child_id":        child_id,
                    "name":            name,
                    "dob":             dob.isoformat(),
                    "age_months":      age_months_t1,
                    "gender":          gender,
                    "block_id":        block,
                    "anganwadi_id":    anganwadi_id,
                    "created_at":      base_date.isoformat(),
                    "true_tier_t1":    true_tier_t1,  # kept for reference; not stored in DB
                    "true_tier_t2":    true_tier_t2,
                    "age_band":        _age_months_band(age_months_t1),
                })

                # Generate time points
                tiers = [true_tier_t1, true_tier_t2]
                for tp_idx in range(n_time_points):
                    assessment_id = str(uuid.uuid4())
                    tp_date = base_date + timedelta(days=tp_idx * 180)  # 6-month intervals
                    age_at_assessment = age_months_t1 + tp_idx * 6
                    true_tier = tiers[tp_idx]

                    row = {
                        "assessment_id":              assessment_id,
                        "child_id":                   child_id,
                        "assessment_date":            tp_date.isoformat(),
                        "age_at_assessment_months":   age_at_assessment,
                        "is_offline":                 0,
                        "created_at":                 tp_date.isoformat(),
                        "true_tier":                  true_tier,  # for model training label
                        "time_point_index":           tp_idx,
                    }
                    for feature in INDICATOR_RISK_RATES:
                        row[feature] = _generate_indicator_value(feature, true_tier, rng)

                    assessment_rows.append(row)

    children_df = pd.DataFrame(children_rows)
    assessments_df = pd.DataFrame(assessment_rows)
    return children_df, assessments_df


def _next_tier(current_tier: int, rng: np.random.Generator) -> int:
    """Realistic tier transition probabilities for 6-month interval."""
    # Transition matrix: rows = current, cols = next
    transitions = {
        0: [0.85, 0.10, 0.04, 0.01],   # Low: mostly stays low
        1: [0.20, 0.55, 0.20, 0.05],   # Medium: can improve or worsen
        2: [0.10, 0.25, 0.45, 0.20],   # High: harder to improve
        3: [0.05, 0.10, 0.25, 0.60],   # Critical: mostly stays critical
    }
    return int(rng.choice([0, 1, 2, 3], p=transitions[current_tier]))


def _generate_name(gender: str, rng: np.random.Generator) -> str:
    male_names = ["Arjun", "Rohan", "Keshav", "Devendra", "Suresh",
                  "Manish", "Vikram", "Rahul", "Akash", "Sanjay"]
    female_names = ["Priya", "Sunita", "Kavya", "Rekha", "Pooja",
                    "Anita", "Savita", "Meena", "Lakshmi", "Durga"]
    surnames = ["Sharma", "Patel", "Verma", "Singh", "Kumar",
                "Yadav", "Gupta", "Tiwari", "Mishra", "Nair"]
    names = male_names if gender == "M" else female_names
    return f"{rng.choice(names)} {rng.choice(surnames)}"


# --- Demo override children for compelling trajectories ---
DEMO_OVERRIDES = {
    "ECD-0001": {"tier_t1": 0, "tier_t2": 0},   # Stable Low — normal trajectory
    "ECD-0002": {"tier_t1": 2, "tier_t2": 1},   # High → Medium — improving
    "ECD-0003": {"tier_t1": 1, "tier_t2": 3},   # Medium → Critical — KEY DEMO MOMENT
}


def get_feature_columns() -> list:
    return list(INDICATOR_RISK_RATES.keys())


def get_domain_features() -> dict:
    return {
        "cognitive":       [f for f in get_feature_columns() if f.startswith("cog_")],
        "motor":           [f for f in get_feature_columns() if f.startswith("mot_")],
        "socio_emotional": [f for f in get_feature_columns() if f.startswith("se_")],
    }


if __name__ == "__main__":
    children_df, assessments_df = generate_children_and_assessments(
        n_children=60, n_time_points=2, demo_overrides=DEMO_OVERRIDES
    )
    print(f"Generated {len(children_df)} children, {len(assessments_df)} assessments")
    print(f"Tier distribution (T1):\n{children_df['true_tier_t1'].value_counts().sort_index()}")
    print(f"\nSample assessment:\n{assessments_df.iloc[0].to_dict()}")
