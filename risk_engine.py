"""
risk_engine.py — ECD multi-domain risk stratification engine.

Trains XGBoost on synthetic data, applies isotonic calibration,
computes SHAP feature contributions, enforces asymmetric threshold
for High/Critical sensitivity, and saves model artifacts to disk.

Tier mapping: 0=Low, 1=Medium, 2=High, 3=Critical
"""

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, classification_report,
    confusion_matrix, brier_score_loss
)
from sklearn.preprocessing import label_binarize
import xgboost as xgb
import shap

from schema import (
    COGNITIVE_FEATURES, MOTOR_FEATURES, SOCIO_FEATURES, ALL_FEATURES,
    FEATURE_LABELS, TIER_LABELS
)
from synthetic_data_generator import (
    generate_children_and_assessments, get_domain_features, DEMO_OVERRIDES
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "risk_model.joblib")
MODEL_VERSION = "v1.0-synthetic"

# Sensitivity-first thresholds per overall tier
# Calibrated probability must exceed these to be assigned this tier
# Lower threshold = higher recall (sensitivity) for High/Critical
TIER_THRESHOLDS = {
    3: 0.30,   # Critical — very sensitive; accept some false positives
    2: 0.38,   # High
    1: 0.50,   # Medium
    0: 0.0,    # Low — always assigned if no higher tier triggered
}


def _train_domain_model(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
) -> CalibratedClassifierCV:
    """Train XGBoost with isotonic calibration for one domain."""
    base_clf = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=random_state,
        verbosity=0,
        n_jobs=-1
    )
    calibrated = CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
    calibrated.fit(X_train, y_train)
    return calibrated


def _compute_shap(model: CalibratedClassifierCV, X: pd.DataFrame) -> np.ndarray:
    """
    Extract SHAP values from the underlying XGBoost estimator.
    Returns array of shape (n_samples, n_features, n_classes).
    """
    # Access base estimator inside isotonic calibration
    base_estimator = model.calibrated_classifiers_[0].estimator
    explainer = shap.TreeExplainer(base_estimator)
    shap_values = explainer.shap_values(X)
    return shap_values  # list of arrays, one per class


def _top3_factors(shap_values_for_instance, feature_names: list, tier: int) -> list:
    """
    Return top-3 SHAP contributors for the predicted tier.

    XGBoost TreeExplainer returns a list of arrays (one per class),
    each of shape (n_samples, n_features). For a single-row X we get
    shape (1, n_features), so we take [0] to get the 1-D vector.
    """
    if isinstance(shap_values_for_instance, list):
        # Multi-class: list[ class ] -> array (n_samples, n_features)
        raw = shap_values_for_instance[tier] if tier < len(shap_values_for_instance) \
              else shap_values_for_instance[-1]
    else:
        raw = shap_values_for_instance

    # Unwrap single-sample dimension if present
    sv = raw[0] if raw.ndim == 2 else raw   # -> shape (n_features,)

    abs_vals = np.abs(sv)
    top_indices = np.argsort(abs_vals)[::-1][:3]
    factors = []
    for idx in top_indices:
        i = int(idx)          # cast numpy integer → Python int for list indexing
        feat = feature_names[i]
        val = float(sv[i])
        factors.append({
            "feature":   feat,
            "label":     FEATURE_LABELS.get(feat, feat),
            "direction": "increases risk" if val > 0 else "reduces risk",
            "magnitude": round(abs(val), 4),
        })
    return factors


def _assign_tier_from_probs(probs: np.ndarray) -> int:
    """
    Assign tier using asymmetric sensitivity-first thresholds.
    Checks Critical first, then High, then Medium, then defaults to Low.
    """
    for tier in [3, 2, 1]:
        if probs[tier] >= TIER_THRESHOLDS[tier]:
            return tier
    return 0


def train_and_save(force_retrain: bool = False) -> dict:
    """
    Train the risk model on synthetic data and save to disk.
    Returns training metrics dict.
    """
    if os.path.exists(MODEL_PATH) and not force_retrain:
        print(f"[risk_engine] Model already exists at {MODEL_PATH}. Use force_retrain=True to retrain.")
        return load_model()

    os.makedirs(MODEL_DIR, exist_ok=True)
    print("[risk_engine] Generating synthetic training data...")
    children_df, assessments_df = generate_children_and_assessments(
        n_children=600, n_time_points=1, demo_overrides=DEMO_OVERRIDES
    )

    domain_map = get_domain_features()
    models = {}
    shap_explainers = {}
    metrics = {}
    cohort_stats = {}

    for domain, features in domain_map.items():
        print(f"[risk_engine] Training {domain} model...")
        X = assessments_df[features].copy()
        y = assessments_df["true_tier"].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = _train_domain_model(X_train, y_train)
        models[domain] = model

        # SHAP explainer (store on base estimator)
        base_est = model.calibrated_classifiers_[0].estimator
        explainer = shap.TreeExplainer(base_est)
        shap_explainers[domain] = explainer

        # Metrics
        probs = model.predict_proba(X_test)
        preds = np.array([_assign_tier_from_probs(p) for p in probs])

        y_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
        auc = roc_auc_score(y_bin, probs, multi_class="ovr", average="macro")
        bal_acc = balanced_accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)

        # FNR for High+Critical
        cm = confusion_matrix(y_test, preds, labels=[0, 1, 2, 3])
        fnr_high = 1 - (cm[2, 2] / cm[2].sum()) if cm[2].sum() > 0 else 0
        fnr_critical = 1 - (cm[3, 3] / cm[3].sum()) if cm[3].sum() > 0 else 0

        metrics[domain] = {
            "auc_roc_macro":    round(auc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "fnr_high":          round(fnr_high, 4),
            "fnr_critical":      round(fnr_critical, 4),
            "per_class_recall":  {
                str(k): round(v["recall"], 4)
                for k, v in report.items() if k.isdigit()
            },
            "n_test":            len(y_test),
        }
        print(f"  AUC={auc:.3f}  BalAcc={bal_acc:.3f}  FNR_Critical={fnr_critical:.3f}")

        # Cohort stats for z-score normalisation (full dataset)
        X_all = assessments_df[features]
        probs_all = model.predict_proba(X_all)
        risk_scores = probs_all[:, 2] + probs_all[:, 3]  # P(High) + P(Critical)
        cohort_stats[domain] = {
            "mean": float(np.mean(risk_scores)),
            "std":  float(np.std(risk_scores) + 1e-8),
        }

    artifact = {
        "models":         models,
        "shap_explainers": shap_explainers,
        "metrics":        metrics,
        "cohort_stats":   cohort_stats,
        "model_version":  MODEL_VERSION,
        "trained_at":     datetime.now().isoformat(),
        "feature_map": {
            "cognitive":       COGNITIVE_FEATURES,
            "motor":           MOTOR_FEATURES,
            "socio_emotional": SOCIO_FEATURES,
        },
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"[risk_engine] Model saved to {MODEL_PATH}")
    return artifact


def load_model() -> dict:
    """Load model artifact from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train_and_save() first "
            "or execute: python risk_engine.py"
        )
    return joblib.load(MODEL_PATH)


def predict(
    artifact: dict,
    assessment_row: dict,
    is_provisional: bool = False
) -> dict:
    """
    Predict risk tier and SHAP explanation for a single assessment.

    Args:
        artifact:        Loaded model artifact from load_model().
        assessment_row:  Dict of feature values (15 indicators + metadata).
        is_provisional:  True if computed offline (no calibration available).

    Returns:
        {
            cognitive_tier, motor_tier, socio_tier, overall_tier,
            cognitive_prob, motor_prob, socio_prob, overall_prob,
            top_factors,          # list of top-3 contributors (overall)
            domain_top_factors,   # dict per domain
            shap_values_json,
            is_provisional,
            model_version,
        }
    """
    result = {"is_provisional": is_provisional, "model_version": MODEL_VERSION}
    domain_tiers = {}
    domain_probs = {}
    doamin_top_factors = {}
    all_shap = {}

    domain_feature_map = {
        "cognitive":       COGNITIVE_FEATURES,
        "motor":           MOTOR_FEATURES,
        "socio_emotional": SOCIO_FEATURES,
    }

    for domain, features in domain_feature_map.items():
        model = artifact["models"][domain]
        explainer = artifact["shap_explainers"][domain]

        X = pd.DataFrame([{f: assessment_row.get(f, 0) for f in features}])
        probs = model.predict_proba(X)[0]
        tier = _assign_tier_from_probs(probs)

        # SHAP
        shap_vals = explainer.shap_values(X)
        factors = _top3_factors(shap_vals, features, tier)

        domain_tiers[domain] = tier
        domain_probs[domain] = float(probs[tier])
        doamin_top_factors[domain] = factors
        # Unwrap (n_samples, n_features) -> (n_features,) for single-row X
        def _sv_for_class(sv_list, cls):
            arr = sv_list[cls] if isinstance(sv_list, list) else sv_list
            return arr[0] if arr.ndim == 2 else arr

        all_shap[domain] = {
            feat: float(_sv_for_class(shap_vals, tier)[i])
            for i, feat in enumerate(features)
        }

    # Overall tier = max across domains (worst domain drives overall)
    overall_tier = max(domain_tiers.values())
    # Overall prob = probability from the domain driving the overall tier
    worst_domain = max(domain_tiers, key=lambda d: domain_tiers[d])
    overall_prob = domain_probs[worst_domain]

    # Top 3 overall factors = from worst domain
    top_factors = doamin_top_factors[worst_domain]

    result.update({
        "cognitive_tier":    domain_tiers["cognitive"],
        "motor_tier":        domain_tiers["motor"],
        "socio_tier":        domain_tiers["socio_emotional"],
        "overall_tier":      overall_tier,
        "cognitive_prob":    domain_probs["cognitive"],
        "motor_prob":        domain_probs["motor"],
        "socio_prob":        domain_probs["socio_emotional"],
        "overall_prob":      overall_prob,
        "top_factors":       top_factors,
        "domain_top_factors": doamin_top_factors,
        "shap_values_json":  json.dumps(all_shap),
        "top_factors_json":  json.dumps(top_factors),
    })
    return result


def offline_provisional_predict(assessment_row: dict) -> dict:
    """
    Rule-based provisional risk tier for offline mode (no model required).
    Simple threshold on sum of at-risk indicators per domain.
    Clearly labelled as PROVISIONAL.
    """
    def _domain_score(features, binary_feats):
        score = 0
        for f in features:
            v = assessment_row.get(f, 0)
            if f in binary_feats:
                score += v  # 1 = at risk
            else:
                score += max(0, (3 - v))  # ordinal: lower score = more risk
        return score

    from schema import BINARY_FEATURES

    cog_score = _domain_score(COGNITIVE_FEATURES, BINARY_FEATURES)
    mot_score = _domain_score(MOTOR_FEATURES, BINARY_FEATURES)
    se_score  = _domain_score(SOCIO_FEATURES, BINARY_FEATURES)

    def _score_to_tier(score, max_score=7):
        ratio = score / max_score
        if ratio >= 0.70: return 3
        if ratio >= 0.45: return 2
        if ratio >= 0.20: return 1
        return 0

    cog_tier = _score_to_tier(cog_score)
    mot_tier = _score_to_tier(mot_score)
    se_tier  = _score_to_tier(se_score)
    overall  = max(cog_tier, mot_tier, se_tier)

    return {
        "cognitive_tier": cog_tier,
        "motor_tier":     mot_tier,
        "socio_tier":     se_tier,
        "overall_tier":   overall,
        "cognitive_prob": None,
        "motor_prob":     None,
        "socio_prob":     None,
        "overall_prob":   None,
        "top_factors":    [],
        "top_factors_json": "[]",
        "shap_values_json": "{}",
        "is_provisional": True,
        "model_version":  "offline-rule-based",
    }


def get_training_metrics() -> dict:
    """Load and return training metrics from saved artifact."""
    artifact = load_model()
    return artifact.get("metrics", {})


if __name__ == "__main__":
    print("Training ECD Risk Engine...")
    artifact = train_and_save(force_retrain=True)
    print("\n=== Training Metrics ===")
    for domain, m in artifact["metrics"].items():
        print(f"\n{domain.upper()}")
        print(f"  AUC-ROC (macro):    {m['auc_roc_macro']}")
        print(f"  Balanced Accuracy:  {m['balanced_accuracy']}")
        print(f"  FNR High tier:      {m['fnr_high']}")
        print(f"  FNR Critical tier:  {m['fnr_critical']}")
        print(f"  Per-class Recall:   {m['per_class_recall']}")
