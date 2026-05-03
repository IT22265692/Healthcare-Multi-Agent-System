"""
tests/test_student1_symptom_agent.py
=====================================
STUDENT 1 — Individual Test File
Agent   : Symptom Analysis Agent (agents/symptom_agent.py)
Tool    : normalize_symptoms() + score_severity() (tools/symptom_tools.py)

This file contains ALL test cases written by Student 1 to validate their
own agent and tool. It is part of the unified test harness but can also
be run standalone:

    python -m pytest tests/test_student1_symptom_agent.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.symptom_tools import normalize_symptoms, score_severity


# ═══════════════════════════════════════════════════════════════════════════
# SECTION A — normalize_symptoms() Unit Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestNormalizeSymptoms:
    """Unit tests for the normalize_symptoms tool."""

    # ── Return type and structure ─────────────────────────────────────────

    def test_returns_list(self):
        """Must always return a Python list."""
        result = normalize_symptoms("fever")
        assert isinstance(result, list)

    def test_returns_non_empty_list_for_valid_input(self):
        """Non-empty input must produce a non-empty list."""
        result = normalize_symptoms("headache")
        assert len(result) > 0

    def test_each_item_is_string(self):
        """Every element in the result must be a string."""
        result = normalize_symptoms("fever and cough")
        assert all(isinstance(s, str) for s in result)

    def test_each_item_non_empty(self):
        """No empty strings in the result."""
        result = normalize_symptoms("fever, cough, headache")
        assert all(len(s.strip()) > 0 for s in result)

    # ── Input cleaning ─────────────────────────────────────────────────────

    def test_lowercases_output(self):
        """Output symptom strings must be lowercase."""
        result = normalize_symptoms("FEVER AND COUGH")
        for item in result:
            assert item == item.lower(), f"Expected lowercase, got: {item}"

    def test_strips_filler_phrases(self):
        """Common filler phrases like 'I have' and 'I feel' are removed."""
        result = normalize_symptoms("I have a fever and I feel really tired")
        assert "fever" in result
        assert "fatigue" in result

    def test_comma_delimited_input(self):
        """Comma-separated list should produce multiple items."""
        result = normalize_symptoms("fever, cough, headache")
        assert len(result) >= 3

    def test_and_delimiter(self):
        """'and' keyword should split symptoms correctly."""
        result = normalize_symptoms("fever and cough")
        assert "fever" in result
        assert "cough" in result

    def test_semicolon_delimiter(self):
        """Semicolons should act as symptom delimiters."""
        result = normalize_symptoms("fever; headache; sore throat")
        assert len(result) >= 2

    def test_newline_delimiter(self):
        """Newlines should act as symptom delimiters."""
        result = normalize_symptoms("fever\ncough\nheadache")
        assert len(result) >= 2

    # ── Synonym mapping ────────────────────────────────────────────────────

    def test_running_nose_maps_to_runny_nose(self):
        """'running nose' → 'runny nose'"""
        result = normalize_symptoms("running nose")
        assert "runny nose" in result

    def test_sob_maps_to_dyspnoea(self):
        """'SOB' abbreviation → 'dyspnoea'"""
        result = normalize_symptoms("SOB")
        assert "dyspnoea" in result

    def test_shortness_of_breath_maps_to_dyspnoea(self):
        """'shortness of breath' → 'dyspnoea'"""
        result = normalize_symptoms("shortness of breath")
        assert "dyspnoea" in result

    def test_difficulty_breathing_maps_to_dyspnoea(self):
        """'difficulty breathing' → 'dyspnoea'"""
        result = normalize_symptoms("difficulty breathing")
        assert "dyspnoea" in result

    def test_tired_maps_to_fatigue(self):
        """'tired' → 'fatigue'"""
        result = normalize_symptoms("I feel tired")
        assert "fatigue" in result

    def test_exhausted_maps_to_fatigue(self):
        """'exhausted' → 'fatigue'"""
        result = normalize_symptoms("exhausted")
        assert "fatigue" in result

    def test_stomach_ache_maps_to_abdominal_pain(self):
        """'stomach ache' → 'abdominal pain'"""
        result = normalize_symptoms("stomach ache")
        assert "abdominal pain" in result

    def test_throwing_up_maps_to_vomiting(self):
        """'throwing up' → 'vomiting'"""
        result = normalize_symptoms("throwing up")
        assert "vomiting" in result

    def test_dizzy_maps_to_dizziness(self):
        """'dizzy' → 'dizziness'"""
        result = normalize_symptoms("dizzy")
        assert "dizziness" in result

    def test_temp_maps_to_fever(self):
        """'temp' abbreviation → 'fever'"""
        result = normalize_symptoms("high temp")
        assert "fever" in result

    # ── Deduplication ─────────────────────────────────────────────────────

    def test_deduplication_exact(self):
        """Exact duplicate symptoms must be collapsed to one."""
        result = normalize_symptoms("fever, fever, fever")
        assert result.count("fever") == 1

    def test_deduplication_synonym(self):
        """Synonyms mapping to the same canonical term must be deduplicated."""
        result = normalize_symptoms("fever, temperature, pyrexia")
        assert result.count("fever") == 1

    def test_deduplication_preserves_order(self):
        """First occurrence of a symptom must appear before later ones."""
        result = normalize_symptoms("fever, cough, headache")
        assert result.index("fever") < result.index("cough")

    # ── Error handling ─────────────────────────────────────────────────────

    def test_raises_on_empty_string(self):
        """Empty string must raise ValueError."""
        with pytest.raises(ValueError):
            normalize_symptoms("")

    def test_raises_on_whitespace_only(self):
        """Whitespace-only string must raise ValueError."""
        with pytest.raises(ValueError):
            normalize_symptoms("     ")

    def test_raises_on_tab_only(self):
        """Tab-only string must raise ValueError."""
        with pytest.raises(ValueError):
            normalize_symptoms("\t\t")

    def test_handles_single_word(self):
        """Single-word input must not crash."""
        result = normalize_symptoms("cough")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_handles_very_long_input(self):
        """Very long input must not crash."""
        long_input = ", ".join(["fever"] * 50)
        result = normalize_symptoms(long_input)
        assert isinstance(result, list)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION B — score_severity() Unit Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestScoreSeverity:
    """Unit tests for the score_severity tool."""

    # ── Return type ───────────────────────────────────────────────────────

    def test_returns_tuple(self):
        """Must return a tuple."""
        result = score_severity(["fever"])
        assert isinstance(result, tuple)

    def test_returns_two_elements(self):
        """Tuple must have exactly 2 elements."""
        result = score_severity(["fever"])
        assert len(result) == 2

    def test_first_element_is_int(self):
        """First element (score) must be an integer."""
        score, _ = score_severity(["fever"])
        assert isinstance(score, int)

    def test_second_element_is_string(self):
        """Second element (rationale) must be a string."""
        _, rationale = score_severity(["fever"])
        assert isinstance(rationale, str)

    # ── Score range ───────────────────────────────────────────────────────

    def test_score_minimum_is_1(self):
        """Score must never be less than 1."""
        score, _ = score_severity(["runny nose"])
        assert score >= 1

    def test_score_maximum_is_10(self):
        """Score must never exceed 10."""
        score, _ = score_severity(
            ["chest pain", "dyspnoea", "seizure", "haemorrhage", "syncope"]
        )
        assert score <= 10

    def test_score_always_in_range(self):
        """For any valid input the score must be in [1, 10]."""
        test_cases = [
            ["fever"],
            ["cough", "runny nose"],
            ["chest pain"],
            ["fatigue", "dizziness", "headache"],
            ["chest pain", "dyspnoea", "seizure"],
        ]
        for symptoms in test_cases:
            score, _ = score_severity(symptoms)
            assert 1 <= score <= 10, f"Score {score} out of range for {symptoms}"

    # ── Severity logic ────────────────────────────────────────────────────

    def test_high_severity_symptoms_produce_high_score(self):
        """Chest pain + dyspnoea must produce score >= 7."""
        score, _ = score_severity(["chest pain", "dyspnoea"])
        assert score >= 7

    def test_mild_symptoms_produce_low_score(self):
        """Single mild symptom must produce score < 5."""
        score, _ = score_severity(["runny nose"])
        assert score < 5

    def test_more_symptoms_increases_score(self):
        """More symptoms should result in equal or higher score."""
        score_one, _ = score_severity(["fever"])
        score_many, _ = score_severity(["fever", "cough", "headache", "fatigue"])
        assert score_many >= score_one

    def test_chest_pain_alone_scores_high(self):
        """Chest pain alone must produce score >= 4 (base 1 + high-severity +3)."""
        score, _ = score_severity(["chest pain"])
        assert score >= 4

    def test_single_symptom_scores_at_least_1(self):
        """Any single symptom must score at least 1."""
        score, _ = score_severity(["runny nose"])
        assert score >= 1

    def test_capped_at_10_with_many_high_severity(self):
        """Score must be capped at 10 even with 5 high-severity symptoms."""
        score, _ = score_severity(
            ["chest pain", "dyspnoea", "seizure", "haemorrhage", "syncope", "paralysis"]
        )
        assert score == 10

    # ── Rationale quality ─────────────────────────────────────────────────

    def test_rationale_is_non_empty(self):
        """Rationale must not be empty."""
        _, rationale = score_severity(["fever"])
        assert len(rationale.strip()) > 10

    def test_rationale_mentions_score(self):
        """Rationale must reference the numeric score."""
        score, rationale = score_severity(["fever"])
        assert str(score) in rationale

    def test_rationale_mentions_high_severity_when_present(self):
        """When high-severity symptom present, rationale must mention it."""
        _, rationale = score_severity(["chest pain"])
        assert "chest pain" in rationale.lower() or "high" in rationale.lower()

    # ── Error handling ────────────────────────────────────────────────────

    def test_raises_on_empty_list(self):
        """Empty list must raise ValueError."""
        with pytest.raises(ValueError):
            score_severity([])

    def test_handles_unknown_symptoms_gracefully(self):
        """Unknown symptoms should not crash the tool."""
        score, rationale = score_severity(["xyzzy_unknown"])
        assert isinstance(score, int)
        assert 1 <= score <= 10


# ═══════════════════════════════════════════════════════════════════════════
# SECTION C — Symptom Agent Integration Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSymptomAgentIntegration:
    """Integration tests for the Symptom Analysis Agent node."""

    def _base_state(self, symptoms: str = "fever and cough"):
        from state import PatientState
        return PatientState(
            patient_id="S1_TEST",
            raw_symptoms=symptoms,
            structured_symptoms=None,
            severity_score=None,
            symptom_analysis=None,
            possible_conditions=None,
            research_summary=None,
            red_flags=None,
            triage_level=None,
            triage_reasoning=None,
            final_report=None,
            report_filepath=None,
            agent_trace=[],
            error_log=[],
        )

    def _mock_response(self):
        m = MagicMock()
        m.raise_for_status = lambda: None
        m.json.return_value = {"response": "Clinical intake complete. Passing to Research Agent."}
        return m

    @patch("requests.post")
    def test_agent_returns_dict(self, mock_post):
        """Agent must return a dictionary."""
        mock_post.return_value = self._mock_response()
        from agents.symptom_agent import run_symptom_agent
        result = run_symptom_agent(self._base_state())
        assert isinstance(result, dict)

    @patch("requests.post")
    def test_structured_symptoms_populated(self, mock_post):
        """structured_symptoms must be set and be a list."""
        mock_post.return_value = self._mock_response()
        from agents.symptom_agent import run_symptom_agent
        result = run_symptom_agent(self._base_state("fever, cough, runny nose"))
        assert result["structured_symptoms"] is not None
        assert isinstance(result["structured_symptoms"], list)

    @patch("requests.post")
    def test_severity_score_populated(self, mock_post):
        """severity_score must be an integer in [1, 10]."""
        mock_post.return_value = self._mock_response()
        from agents.symptom_agent import run_symptom_agent
        result = run_symptom_agent(self._base_state())
        score = result["severity_score"]
        assert isinstance(score, int)
        assert 1 <= score <= 10

    @patch("requests.post")
    def test_symptom_analysis_populated(self, mock_post):
        """symptom_analysis must be a non-empty string."""
        mock_post.return_value = self._mock_response()
        from agents.symptom_agent import run_symptom_agent
        result = run_symptom_agent(self._base_state())
        assert isinstance(result["symptom_analysis"], str)
        assert len(result["symptom_analysis"]) > 5

    @patch("requests.post")
    def test_patient_id_preserved(self, mock_post):
        """patient_id must not be changed by the agent."""
        mock_post.return_value = self._mock_response()
        from agents.symptom_agent import run_symptom_agent
        result = run_symptom_agent(self._base_state())
        assert result["patient_id"] == "S1_TEST"

    @patch("requests.post")
    def test_raw_symptoms_preserved(self, mock_post):
        """raw_symptoms must be preserved unchanged."""
        mock_post.return_value = self._mock_response()
        from agents.symptom_agent import run_symptom_agent
        state = self._base_state("fever and cough")
        result = run_symptom_agent(state)
        assert result["raw_symptoms"] == "fever and cough"

    @patch("requests.post")
    def test_agent_trace_has_entries(self, mock_post):
        """agent_trace must contain at least 2 entries after running."""
        mock_post.return_value = self._mock_response()
        from agents.symptom_agent import run_symptom_agent
        result = run_symptom_agent(self._base_state())
        assert len(result["agent_trace"]) >= 2

    @patch("requests.post")
    def test_error_log_empty_for_clean_run(self, mock_post):
        """error_log must be empty when everything works correctly."""
        mock_post.return_value = self._mock_response()
        from agents.symptom_agent import run_symptom_agent
        result = run_symptom_agent(self._base_state())
        assert result["error_log"] == []

    @patch("requests.post")
    def test_high_severity_symptoms_detected(self, mock_post):
        """Chest pain input must produce severity_score >= 7."""
        mock_post.return_value = self._mock_response()
        from agents.symptom_agent import run_symptom_agent
        result = run_symptom_agent(self._base_state("chest pain and difficulty breathing"))
        assert result["severity_score"] >= 7

    @patch("requests.post")
    def test_fallback_when_ollama_offline(self, mock_post):
        """Agent must not crash when Ollama connection is refused."""
        import requests as req
        mock_post.side_effect = req.exceptions.ConnectionError("refused")
        from agents.symptom_agent import run_symptom_agent
        result = run_symptom_agent(self._base_state("fever"))
        # Should still have outputs — graceful degradation
        assert result["structured_symptoms"] is not None
        assert result["severity_score"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])