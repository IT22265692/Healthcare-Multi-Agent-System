"""
tests/test_student2_research_agent.py
======================================
STUDENT 2 — Individual Test File
Agent   : Research Agent (agents/research_agent.py)
Tool    : query_disease_api() + extract_red_flags() (tools/research_tools.py)

Run standalone:
    python -m pytest tests/test_student2_research_agent.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.research_tools import query_disease_api, extract_red_flags


# ═══════════════════════════════════════════════════════════════════════════
# SECTION A — query_disease_api() Unit Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestQueryDiseaseApi:
    """Unit tests for the query_disease_api tool."""

    # ── Return type and structure ─────────────────────────────────────────

    def test_returns_dict(self):
        """Must return a dictionary."""
        result = query_disease_api(["fever"])
        assert isinstance(result, dict)

    def test_has_conditions_key(self):
        """Result must contain 'conditions' key."""
        result = query_disease_api(["fever"])
        assert "conditions" in result

    def test_has_notes_key(self):
        """Result must contain 'notes' key."""
        result = query_disease_api(["fever"])
        assert "notes" in result

    def test_has_coverage_key(self):
        """Result must contain 'coverage' key."""
        result = query_disease_api(["fever"])
        assert "coverage" in result

    def test_conditions_is_list(self):
        """'conditions' value must be a list."""
        result = query_disease_api(["fever"])
        assert isinstance(result["conditions"], list)

    def test_notes_is_list(self):
        """'notes' value must be a list."""
        result = query_disease_api(["fever"])
        assert isinstance(result["notes"], list)

    def test_coverage_is_int(self):
        """'coverage' value must be an integer."""
        result = query_disease_api(["fever"])
        assert isinstance(result["coverage"], int)

    # ── Content correctness ───────────────────────────────────────────────

    def test_fever_returns_known_conditions(self):
        """'fever' must return Influenza and COVID-19 as top conditions."""
        result = query_disease_api(["fever"])
        assert "Influenza" in result["conditions"] or "COVID-19" in result["conditions"]

    def test_cough_returns_conditions(self):
        """'cough' must return at least one respiratory condition."""
        result = query_disease_api(["cough"])
        respiratory = {"Bronchitis", "Asthma", "COVID-19", "Upper Respiratory Tract Infection"}
        assert any(c in respiratory for c in result["conditions"])

    def test_chest_pain_surfaces_acs(self):
        """'chest pain' must surface Acute Coronary Syndrome."""
        result = query_disease_api(["chest pain"])
        assert "Acute Coronary Syndrome" in result["conditions"]

    def test_dyspnoea_surfaces_serious_conditions(self):
        """'dyspnoea' must surface at least one serious condition."""
        result = query_disease_api(["dyspnoea"])
        serious = {"Heart Failure", "Pulmonary Embolism", "COPD Exacerbation", "Asthma"}
        assert any(c in serious for c in result["conditions"])

    def test_multiple_symptoms_combines_conditions(self):
        """Multiple symptoms must produce a combined condition list."""
        result_one = query_disease_api(["fever"])
        result_many = query_disease_api(["fever", "cough", "headache"])
        # Multiple symptoms should produce at least as many conditions
        assert len(result_many["conditions"]) >= len(result_one["conditions"])

    def test_co_occurrence_ranking(self):
        """Conditions appearing across multiple symptoms should rank higher."""
        # Both fever and cough map to Influenza and COVID-19
        result = query_disease_api(["fever", "cough"])
        top_two = result["conditions"][:2]
        assert "Influenza" in top_two or "COVID-19" in top_two

    # ── Limits ────────────────────────────────────────────────────────────

    def test_max_five_conditions_returned(self):
        """No more than 5 conditions should ever be returned."""
        result = query_disease_api(["fever", "cough", "headache", "dizziness", "fatigue"])
        assert len(result["conditions"]) <= 5

    def test_coverage_does_not_exceed_symptom_count(self):
        """Coverage cannot exceed the number of symptoms provided."""
        symptoms = ["fever", "cough"]
        result = query_disease_api(symptoms)
        assert result["coverage"] <= len(symptoms)

    def test_coverage_positive_for_known_symptoms(self):
        """Coverage must be >= 1 for known symptoms."""
        result = query_disease_api(["fever"])
        assert result["coverage"] >= 1

    # ── Edge cases ────────────────────────────────────────────────────────

    def test_unknown_symptom_returns_fallback(self):
        """Unknown symptoms must not crash — return fallback condition."""
        result = query_disease_api(["xyzzy_totally_unknown_symptom"])
        assert isinstance(result["conditions"], list)
        assert len(result["conditions"]) >= 1

    def test_raises_on_empty_list(self):
        """Empty list must raise ValueError."""
        with pytest.raises(ValueError):
            query_disease_api([])

    def test_conditions_are_strings(self):
        """All returned conditions must be strings."""
        result = query_disease_api(["fever", "cough"])
        assert all(isinstance(c, str) for c in result["conditions"])

    def test_notes_are_strings(self):
        """All returned notes must be strings."""
        result = query_disease_api(["fever"])
        assert all(isinstance(n, str) for n in result["notes"])

    def test_notes_contain_symptom_name(self):
        """Notes for a matched symptom must reference the symptom."""
        result = query_disease_api(["fever"])
        combined = " ".join(result["notes"]).upper()
        assert "FEVER" in combined


# ═══════════════════════════════════════════════════════════════════════════
# SECTION B — extract_red_flags() Unit Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestExtractRedFlags:
    """Unit tests for the extract_red_flags tool."""

    # ── Return type ───────────────────────────────────────────────────────

    def test_returns_list(self):
        """Must return a list."""
        result = extract_red_flags([], [])
        assert isinstance(result, list)

    def test_empty_inputs_return_empty_list(self):
        """No symptoms and no conditions must return an empty list."""
        result = extract_red_flags([], [])
        assert result == []

    def test_each_flag_is_string(self):
        """Every flag in the result must be a string."""
        result = extract_red_flags(["chest pain"], [])
        assert all(isinstance(f, str) for f in result)

    # ── Red flag symptom detection ────────────────────────────────────────

    def test_chest_pain_triggers_flag(self):
        """'chest pain' must trigger a red flag."""
        flags = extract_red_flags(["chest pain"], [])
        assert len(flags) > 0

    def test_chest_pain_flag_message_contains_symptom(self):
        """Red flag message must reference 'CHEST PAIN'."""
        flags = extract_red_flags(["chest pain"], [])
        assert any("CHEST PAIN" in f.upper() for f in flags)

    def test_dyspnoea_triggers_flag(self):
        """'dyspnoea' must trigger a red flag."""
        flags = extract_red_flags(["dyspnoea"], [])
        assert len(flags) > 0

    def test_seizure_triggers_flag(self):
        """'seizure' must trigger a red flag."""
        flags = extract_red_flags(["seizure"], [])
        assert len(flags) > 0

    def test_syncope_triggers_flag(self):
        """'syncope' must trigger a red flag."""
        flags = extract_red_flags(["syncope"], [])
        assert len(flags) > 0

    def test_haemorrhage_triggers_flag(self):
        """'haemorrhage' must trigger a red flag."""
        flags = extract_red_flags(["haemorrhage"], [])
        assert len(flags) > 0

    # ── Red flag condition detection ──────────────────────────────────────

    def test_acs_condition_triggers_flag(self):
        """'Acute Coronary Syndrome' condition must trigger a red flag."""
        flags = extract_red_flags([], ["Acute Coronary Syndrome"])
        assert len(flags) > 0

    def test_pulmonary_embolism_triggers_flag(self):
        """'Pulmonary Embolism' condition must trigger a red flag."""
        flags = extract_red_flags([], ["Pulmonary Embolism"])
        assert len(flags) > 0

    def test_meningococcaemia_triggers_flag(self):
        """'Meningococcaemia' condition must trigger a red flag."""
        flags = extract_red_flags([], ["Meningococcaemia"])
        assert len(flags) > 0

    # ── No false positives ────────────────────────────────────────────────

    def test_benign_symptoms_produce_no_flags(self):
        """Common cold symptoms must not trigger any red flags."""
        flags = extract_red_flags(["runny nose", "sore throat", "cough"], ["Common Cold"])
        assert len(flags) == 0

    def test_routine_conditions_produce_no_flags(self):
        """Routine conditions must not trigger red flags."""
        flags = extract_red_flags([], ["Viral Pharyngitis", "Common Cold"])
        assert len(flags) == 0

    def test_fatigue_alone_no_flag(self):
        """Fatigue alone must not trigger a red flag."""
        flags = extract_red_flags(["fatigue"], [])
        assert len(flags) == 0

    def test_headache_alone_no_flag(self):
        """Regular headache alone must not trigger a red flag."""
        flags = extract_red_flags(["headache"], [])
        assert len(flags) == 0

    # ── Multiple flags ────────────────────────────────────────────────────

    def test_multiple_red_flag_symptoms_produce_multiple_flags(self):
        """Two red-flag symptoms must produce at least two flags."""
        flags = extract_red_flags(["chest pain", "dyspnoea"], [])
        assert len(flags) >= 2

    def test_symptom_and_condition_flags_combined(self):
        """Red flags from both symptoms and conditions must be combined."""
        flags = extract_red_flags(["chest pain"], ["Acute Coronary Syndrome"])
        assert len(flags) >= 2

    # ── Error handling ────────────────────────────────────────────────────

    def test_raises_on_string_symptoms(self):
        """Passing a string instead of list must raise TypeError."""
        with pytest.raises(TypeError):
            extract_red_flags("chest pain", [])

    def test_raises_on_string_conditions(self):
        """Passing a string for conditions must raise TypeError."""
        with pytest.raises(TypeError):
            extract_red_flags([], "Influenza")

    def test_raises_on_none_symptoms(self):
        """None for symptoms must raise TypeError."""
        with pytest.raises(TypeError):
            extract_red_flags(None, [])


# ═══════════════════════════════════════════════════════════════════════════
# SECTION C — Research Agent Integration Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestResearchAgentIntegration:
    """Integration tests for the Research Agent node."""

    def _base_state(self, symptoms=None):
        from state import PatientState
        return PatientState(
            patient_id="S2_TEST",
            raw_symptoms="fever and cough",
            structured_symptoms=symptoms or ["fever", "cough"],
            severity_score=5,
            symptom_analysis="Fever and cough detected.",
            possible_conditions=None,
            research_summary=None,
            red_flags=None,
            triage_level=None,
            triage_reasoning=None,
            final_report=None,
            report_filepath=None,
            agent_trace=["[S1] done"],
            error_log=[],
        )

    def _mock(self):
        m = MagicMock()
        m.raise_for_status = lambda: None
        m.json.return_value = {"response": "Research complete. Passing findings to Triage Agent."}
        return m

    @patch("requests.post")
    def test_possible_conditions_populated(self, mock_post):
        """possible_conditions must be a non-empty list."""
        mock_post.return_value = self._mock()
        from agents.research_agent import run_research_agent
        result = run_research_agent(self._base_state())
        assert result["possible_conditions"] is not None
        assert len(result["possible_conditions"]) > 0

    @patch("requests.post")
    def test_research_summary_populated(self, mock_post):
        """research_summary must be a non-empty string."""
        mock_post.return_value = self._mock()
        from agents.research_agent import run_research_agent
        result = run_research_agent(self._base_state())
        assert isinstance(result["research_summary"], str)
        assert len(result["research_summary"]) > 5

    @patch("requests.post")
    def test_red_flags_is_list(self, mock_post):
        """red_flags must be a list (can be empty)."""
        mock_post.return_value = self._mock()
        from agents.research_agent import run_research_agent
        result = run_research_agent(self._base_state())
        assert isinstance(result["red_flags"], list)

    @patch("requests.post")
    def test_chest_pain_produces_red_flags(self, mock_post):
        """chest pain + dyspnoea input must produce at least one red flag."""
        mock_post.return_value = self._mock()
        from agents.research_agent import run_research_agent
        result = run_research_agent(self._base_state(["chest pain", "dyspnoea"]))
        assert len(result["red_flags"]) > 0

    @patch("requests.post")
    def test_prior_state_preserved(self, mock_post):
        """symptom_analysis set by Agent 1 must still be present."""
        mock_post.return_value = self._mock()
        from agents.research_agent import run_research_agent
        result = run_research_agent(self._base_state())
        assert result["symptom_analysis"] == "Fever and cough detected."

    @patch("requests.post")
    def test_trace_appended(self, mock_post):
        """agent_trace must grow — prior entries must be preserved."""
        mock_post.return_value = self._mock()
        from agents.research_agent import run_research_agent
        result = run_research_agent(self._base_state())
        assert len(result["agent_trace"]) > 1  # had 1, must be more

    @patch("requests.post")
    def test_ollama_fallback_still_produces_conditions(self, mock_post):
        """Even when Ollama is offline, conditions must be returned via tools."""
        import requests as req
        mock_post.side_effect = req.exceptions.ConnectionError("refused")
        from agents.research_agent import run_research_agent
        result = run_research_agent(self._base_state())
        assert result["possible_conditions"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])