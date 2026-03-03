"""
tests/test_risk_engine.py — Unit tests for the risk engine.

Run: pytest tests/ -v
"""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from risk_engine import (
    _assign_tier_from_probs, offline_provisional_predict, TIER_THRESHOLDS
)
from schema import TIER_LABELS


class TestTierAssignment:
    def test_critical_tier_assigned_at_low_threshold(self):
        """Critical tier should be assigned at probability >= 0.30"""
        probs = np.array([0.05, 0.10, 0.15, 0.70])
        assert _assign_tier_from_probs(probs) == 3

    def test_high_tier_assigned_when_critical_not_met(self):
        probs = np.array([0.10, 0.20, 0.50, 0.20])
        tier = _assign_tier_from_probs(probs)
        assert tier == 2

    def test_medium_tier_assigned(self):
        probs = np.array([0.20, 0.60, 0.15, 0.05])
        tier = _assign_tier_from_probs(probs)
        assert tier == 1

    def test_low_tier_when_all_below_threshold(self):
        probs = np.array([0.80, 0.10, 0.05, 0.05])
        tier = _assign_tier_from_probs(probs)
        assert tier == 0

    def test_tier_labels_cover_all_tiers(self):
        for t in [0, 1, 2, 3]:
            assert t in TIER_LABELS


class TestOfflineProvisionalPredict:
    def _make_row(self, risk_level="high"):
        """Create a synthetic indicator row at given risk level."""
        if risk_level == "high":
            return {
                "cog_lang_milestone": 1, "cog_memory_recall": 1,
                "cog_problem_solving": 1, "cog_attention_span": 1, "cog_learning_adapt": 1,
                "mot_gross_motor": 1, "mot_fine_motor": 1, "mot_balance": 1,
                "mot_hand_eye": 1, "mot_body_aware": 1,
                "se_social_play": 1, "se_emotion_reg": 1, "se_peer_interact": 1,
                "se_attachment": 1, "se_self_care": 1,
            }
        else:  # low risk
            return {
                "cog_lang_milestone": 0, "cog_memory_recall": 4,
                "cog_problem_solving": 4, "cog_attention_span": 0, "cog_learning_adapt": 0,
                "mot_gross_motor": 0, "mot_fine_motor": 4, "mot_balance": 0,
                "mot_hand_eye": 4, "mot_body_aware": 0,
                "se_social_play": 0, "se_emotion_reg": 4, "se_peer_interact": 4,
                "se_attachment": 0, "se_self_care": 0,
            }

    def test_high_indicators_produce_high_or_critical(self):
        result = offline_provisional_predict(self._make_row("high"))
        assert result["overall_tier"] >= 2, "All-at-risk indicators should produce High or Critical"

    def test_low_indicators_produce_low(self):
        result = offline_provisional_predict(self._make_row("low"))
        assert result["overall_tier"] == 0, "All-met indicators should produce Low risk"

    def test_is_provisional_always_true(self):
        result = offline_provisional_predict(self._make_row("high"))
        assert result["is_provisional"] is True

    def test_result_has_required_keys(self):
        result = offline_provisional_predict(self._make_row("low"))
        required = ["overall_tier", "cognitive_tier", "motor_tier", "socio_tier", "is_provisional"]
        for key in required:
            assert key in result, f"Missing key: {key}"
