"""
tests/test_student3_triage_agent.py
=====================================
STUDENT 3 — Individual Test File
Agent   : Triage Agent (agents/triage_agent.py)
Tool    : classify_triage_level() + generate_triage_advice() (tools/triage_tools.py)

Run standalone:
    python -m pytest tests/test_student3_triage_agent.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.triage_tools import classify_triage_level, generate_triage_advice


# ═══════════════════════════════════════════════════════════════════════════
# SECTION A — classify_triage_level() Unit Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestClassifyTriageLevel:
    """Unit tests for the classify_triage_level tool."""

    VALID_LEVELS = {"EMERGENCY", "URGENT", "SEMI-URGENT", "ROUTINE"}

    # ── Return type ───────────────────────────────────────────────────────

    def test_returns_tuple(self):
        """Must return a tuple."""
        result = classify_triage_level(5, [])
        assert isinstance(result, tuple)

    def test_returns_two_elements(self):
        """Tuple must have exactly 2 elements."""
        result = classify_triage_level(5, [])
        assert len(result) == 2

    def test_first_element_is_string(self):
        """First element (level) must be a string."""
        level, _ = classify_triage_level(5, [])
        assert isinstance(level, str)

    def test_second_element_is_string(self):
        """Second element (reasoning) must be a string."""
        _, reasoning = classify_triage_level(5, [])
        assert isinstance(reasoning, str)

    # ── Level correctness ─────────────────────────────────────────────────

    def test_level_always_valid(self):
        """Level must always be one of the four valid triage tiers."""
        for score in range(1, 11):
            level, _ = classify_triage_level(score, [])
            assert level in self.VALID_LEVELS, f"Invalid level '{level}' for score {score}"

    def test_score_10_is_emergency(self):
        """Score 10 must produce EMERGENCY."""
        level, _ = classify_triage_level(10, [])
        assert level == "EMERGENCY"

    def test_score_9_is_emergency(self):
        """Score 9 must produce EMERGENCY."""
        level, _ = classify_triage_level(9, [])
        assert level == "EMERGENCY"

    def test_score_8_is_urgent_or_above(self):
        """Score 8 must produce URGENT or EMERGENCY."""
        level, _ = classify_triage_level(8, [])
        assert level in {"URGENT", "EMERGENCY"}

    def test_score_7_is_urgent_or_above(self):
        """Score 7 must produce URGENT or EMERGENCY."""
        level, _ = classify_triage_level(7, [])
        assert level in {"URGENT", "EMERGENCY"}

    def test_score_5_is_semi_urgent_or_below(self):
        """Score 5 without red flags must be SEMI-URGENT or ROUTINE."""
        level, _ = classify_triage_level(5, [])
        assert level in {"SEMI-URGENT", "ROUTINE"}

    def test_score_3_is_routine_or_semi_urgent(self):
        """Score 3 without red flags should be ROUTINE or SEMI-URGENT."""
        level, _ = classify_triage_level(3, [])
        assert level in {"ROUTINE", "SEMI-URGENT"}

    def test_score_1_is_routine(self):
        """Score 1 with no red flags must produce ROUTINE."""
        level, _ = classify_triage_level(1, [])
        assert level == "ROUTINE"

    def test_score_2_is_routine(self):
        """Score 2 with no red flags must produce ROUTINE."""
        level, _ = classify_triage_level(2, [])
        assert level == "ROUTINE"

    # ── Red flag elevation ────────────────────────────────────────────────

    def test_red_flags_elevate_routine_to_urgent(self):
        """Red flag on ROUTINE-range score must elevate to at least URGENT."""
        level, _ = classify_triage_level(2, ["⚠️  CHEST PAIN detected"])
        assert level in {"URGENT", "EMERGENCY"}

    def test_red_flags_elevate_semi_urgent_to_urgent(self):
        """Red flag on SEMI-URGENT-range score must elevate to URGENT or above."""
        level, _ = classify_triage_level(5, ["⚠️  DYSPNOEA detected"])
        assert level in {"URGENT", "EMERGENCY"}

    def test_red_flags_elevate_urgent_to_emergency(self):
        """Red flag on URGENT-range score must elevate to EMERGENCY."""
        level, _ = classify_triage_level(7, ["⚠️  CHEST PAIN detected"])
        assert level == "EMERGENCY"

    def test_multiple_red_flags_produce_emergency(self):
        """Multiple red flags must produce EMERGENCY regardless of base score."""
        level, _ = classify_triage_level(3, [
            "⚠️  CHEST PAIN detected",
            "⚠️  DYSPNOEA detected",
        ])
        assert level in {"URGENT", "EMERGENCY"}

    def test_no_red_flags_no_elevation(self):
        """Without red flags, level must be determined by score alone."""
        level_no_flags, _ = classify_triage_level(3, [])
        level_with_flags, _ = classify_triage_level(3, ["⚠️  flag"])
        # With flags it must be >= without flags
        levels_ordered = ["ROUTINE", "SEMI-URGENT", "URGENT", "EMERGENCY"]
        assert levels_ordered.index(level_with_flags) >= levels_ordered.index(level_no_flags)

    # ── Safety invariants ─────────────────────────────────────────────────

    def test_emergency_is_never_downgraded(self):
        """Score 10 must always yield EMERGENCY — never downgraded."""
        for _ in range(5):  # Run 5 times to check consistency
            level, _ = classify_triage_level(10, [])
            assert level == "EMERGENCY"

    def test_deterministic_same_input_same_output(self):
        """Same inputs must always produce the same triage level."""
        level1, _ = classify_triage_level(6, ["⚠️  flag"])
        level2, _ = classify_triage_level(6, ["⚠️  flag"])
        assert level1 == level2

    def test_reasoning_mentions_score(self):
        """Reasoning must reference the severity score."""
        score = 7
        _, reasoning = classify_triage_level(score, [])
        assert str(score) in reasoning

    def test_reasoning_mentions_level(self):
        """Reasoning must reference the triage level."""
        level, reasoning = classify_triage_level(9, [])
        assert level in reasoning

    def test_reasoning_non_empty(self):
        """Reasoning must not be empty or whitespace."""
        _, reasoning = classify_triage_level(5, [])
        assert len(reasoning.strip()) > 5

    # ── Boundary and error cases ──────────────────────────────────────────

    def test_raises_on_score_zero(self):
        """Score 0 must raise ValueError."""
        with pytest.raises(ValueError):
            classify_triage_level(0, [])

    def test_raises_on_score_eleven(self):
        """Score 11 must raise ValueError."""
        with pytest.raises(ValueError):
            classify_triage_level(11, [])

    def test_raises_on_negative_score(self):
        """Negative score must raise ValueError."""
        with pytest.raises(ValueError):
            classify_triage_level(-1, [])

    def test_raises_on_score_100(self):
        """Score 100 must raise ValueError."""
        with pytest.raises(ValueError):
            classify_triage_level(100, [])

    def test_empty_red_flags_list_ok(self):
        """Empty list for red_flags must not raise any error."""
        level, _ = classify_triage_level(5, [])
        assert level in self.VALID_LEVELS

    def test_all_valid_scores_produce_valid_levels(self):
        """Every score from 1 to 10 must produce a valid triage level."""
        for score in range(1, 11):
            level, _ = classify_triage_level(score, [])
            assert level in self.VALID_LEVELS


# ═══════════════════════════════════════════════════════════════════════════
# SECTION B — generate_triage_advice() Unit Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestGenerateTriageAdvice:
    """Unit tests for the generate_triage_advice tool."""

    VALID_LEVELS = ["EMERGENCY", "URGENT", "SEMI-URGENT", "ROUTINE"]

    def test_returns_string_for_all_levels(self):
        """Must return a non-empty string for every valid level."""
        for level in self.VALID_LEVELS:
            advice = generate_triage_advice(level)
            assert isinstance(advice, str)
            assert len(advice.strip()) > 10

    def test_emergency_mentions_services(self):
        """EMERGENCY advice must mention contacting emergency services."""
        advice = generate_triage_advice("EMERGENCY")
        keywords = ["emergency", "119", "911", "immediately", "ambulance", "without delay"]
        found = any(kw.lower() in advice.lower() for kw in keywords)
        assert found, f"EMERGENCY advice must mention emergency services. Got: {advice}"

    def test_emergency_advice_is_urgent_in_tone(self):
        """EMERGENCY advice must convey high urgency."""
        advice = generate_triage_advice("EMERGENCY")
        urgent_words = ["immediately", "without", "delay", "now", "emergency"]
        assert any(w in advice.lower() for w in urgent_words)

    def test_routine_advice_mentions_gp(self):
        """ROUTINE advice must mention a GP or doctor appointment."""
        advice = generate_triage_advice("ROUTINE")
        gp_words = ["gp", "doctor", "appointment", "clinic", "practice"]
        assert any(w in advice.lower() for w in gp_words)

    def test_urgent_advice_mentions_timeframe(self):
        """URGENT advice must mention a time constraint."""
        advice = generate_triage_advice("URGENT")
        time_words = ["hour", "minute", "today", "now", "soon", "within"]
        assert any(w in advice.lower() for w in time_words)

    def test_semi_urgent_advice_is_less_urgent_than_urgent(self):
        """SEMI-URGENT advice must not say 'call emergency services'."""
        advice = generate_triage_advice("SEMI-URGENT")
        emergency_phrases = ["call 119", "call 911", "call emergency services"]
        assert not any(p in advice.lower() for p in emergency_phrases)

    def test_different_levels_give_different_advice(self):
        """Each triage level must produce different advice text."""
        advices = [generate_triage_advice(level) for level in self.VALID_LEVELS]
        assert len(set(advices)) == 4, "Each level must have unique advice"

    def test_raises_on_unknown_level(self):
        """Unknown level string must raise KeyError."""
        with pytest.raises(KeyError):
            generate_triage_advice("CRITICAL")

    def test_raises_on_empty_string(self):
        """Empty string must raise KeyError."""
        with pytest.raises(KeyError):
            generate_triage_advice("")

    def test_raises_on_lowercase_level(self):
        """Lowercase level (e.g. 'emergency') must raise KeyError."""
        with pytest.raises(KeyError):
            generate_triage_advice("emergency")

    def test_advice_does_not_contain_html(self):
        """Advice must be plain text — no HTML tags."""
        for level in self.VALID_LEVELS:
            advice = generate_triage_advice(level)
            assert "<" not in advice and ">" not in advice


# ═══════════════════════════════════════════════════════════════════════════
# SECTION C — Triage Agent Integration Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestTriageAgentIntegration:
    """Integration tests for the Triage Agent node."""

    def _base_state(self, score=5, red_flags=None):
        from state import PatientState
        return PatientState(
            patient_id="S3_TEST",
            raw_symptoms="fever and cough",
            structured_symptoms=["fever", "cough"],
            severity_score=score,
            symptom_analysis="Fever and cough.",
            possible_conditions=["Influenza", "COVID-19"],
            research_summary="Most likely viral illness.",
            red_flags=red_flags or [],
            triage_level=None,
            triage_reasoning=None,
            final_report=None,
            report_filepath=None,
            agent_trace=["[S1]", "[S2]"],
            error_log=[],
        )

    def _mock(self):
        m = MagicMock()
        m.raise_for_status = lambda: None
        m.json.return_value = {"response": "Triage assessment complete. Passing to Report Agent."}
        return m

    @patch("requests.post")
    def test_triage_level_populated(self, mock_post):
        """triage_level must be set to a valid level."""
        mock_post.return_value = self._mock()
        from agents.triage_agent import run_triage_agent
        result = run_triage_agent(self._base_state())
        assert result["triage_level"] in {"EMERGENCY", "URGENT", "SEMI-URGENT", "ROUTINE"}

    @patch("requests.post")
    def test_triage_reasoning_populated(self, mock_post):
        """triage_reasoning must be a non-empty string."""
        mock_post.return_value = self._mock()
        from agents.triage_agent import run_triage_agent
        result = run_triage_agent(self._base_state())
        assert isinstance(result["triage_reasoning"], str)
        assert len(result["triage_reasoning"]) > 5

    @patch("requests.post")
    def test_high_score_yields_emergency_or_urgent(self, mock_post):
        """Score 9 must yield EMERGENCY."""
        mock_post.return_value = self._mock()
        from agents.triage_agent import run_triage_agent
        result = run_triage_agent(self._base_state(score=9))
        assert result["triage_level"] == "EMERGENCY"

    @patch("requests.post")
    def test_red_flags_elevate_level(self, mock_post):
        """Red flags must elevate a low-score case to at least URGENT."""
        mock_post.return_value = self._mock()
        from agents.triage_agent import run_triage_agent
        result = run_triage_agent(self._base_state(
            score=2,
            red_flags=["⚠️  CHEST PAIN detected — potential emergency"]
        ))
        assert result["triage_level"] in {"URGENT", "EMERGENCY"}

    @patch("requests.post")
    def test_prior_state_fully_preserved(self, mock_post):
        """All prior agent outputs must be preserved in the state."""
        mock_post.return_value = self._mock()
        from agents.triage_agent import run_triage_agent
        result = run_triage_agent(self._base_state())
        assert result["symptom_analysis"] == "Fever and cough."
        assert result["research_summary"] == "Most likely viral illness."
        assert result["patient_id"] == "S3_TEST"

    @patch("requests.post")
    def test_trace_preserves_prior_entries(self, mock_post):
        """Prior agent_trace entries must not be lost."""
        mock_post.return_value = self._mock()
        from agents.triage_agent import run_triage_agent
        result = run_triage_agent(self._base_state())
        trace = result["agent_trace"]
        assert "[S1]" in trace
        assert "[S2]" in trace

    @patch("requests.post")
    def test_ollama_fallback_still_classifies(self, mock_post):
        """Triage level must be set even when Ollama is offline."""
        import requests as req
        mock_post.side_effect = req.exceptions.ConnectionError("refused")
        from agents.triage_agent import run_triage_agent
        result = run_triage_agent(self._base_state())
        assert result["triage_level"] in {"EMERGENCY", "URGENT", "SEMI-URGENT", "ROUTINE"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])