"""
agents/symptom_agent.py — Symptom Analysis Agent (Agent 1 of 4).

Persona: "Dr. Intake" — a meticulous clinical intake specialist whose sole
role is to extract, normalise, and severity-score the patient's symptoms
before passing a clean structured record to the Research Agent.

Responsibilities:
    - Parse free-text symptoms into a canonical list.
    - Assign a 1–10 severity score using evidence-based heuristics.
    - Produce an LLM-reasoned narrative about the symptom presentation.
    - Populate PatientState fields: structured_symptoms, severity_score,
      symptom_analysis, agent_trace.

Constraints:
    - MUST use the normalize_symptoms and score_severity tools.
    - MUST NOT attempt diagnosis (that is the Research Agent's role).
    - MUST flag if input is too vague to process.
    - MUST keep the narrative factual and free of speculative diagnoses.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import requests

from state import PatientState
from observability import (
    log_event, log_llm_call, log_llm_response,
    log_tool_call, log_tool_result, log_error,
)
from tools.symptom_tools import normalize_symptoms, score_severity


_OLLAMA_URL = "http://localhost:11434/api/generate"
_MODEL = "llama3:8b"

_SYSTEM_PROMPT = """You are Dr. Intake, a precise and empathetic clinical intake specialist AI.

Your ONLY role is symptom extraction and severity assessment. You do NOT diagnose.

RULES YOU MUST FOLLOW:
1. Report ONLY what the patient explicitly states — never invent symptoms.
2. Do NOT speculate about diagnoses or conditions.
3. If symptoms are vague, state that clearly and ask what was provided to you.
4. Write in clear, professional clinical language.
5. Keep your response concise: 3–5 sentences maximum.
6. Never recommend specific medications.

OUTPUT FORMAT:
Write a clinical intake note that:
- Confirms which symptoms were identified
- Comments on their apparent severity
- Notes any concerning combinations
- Ends with: "Passing to Research Agent for differential diagnosis."
"""


def _call_ollama(prompt: str, patient_id: str) -> str:
    """
    Call the local Ollama LLM endpoint.

    Args:
        prompt:     The user prompt to send (system prompt is pre-configured).
        patient_id: For trace correlation.

    Returns:
        The LLM's response string.

    Raises:
        RuntimeError: If Ollama is unreachable or returns an error.
    """
    log_llm_call("SymptomAgent", prompt, patient_id=patient_id)

    payload = {
        "model": _MODEL,
        "system": _SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,   # Low temp → less hallucination
            "num_predict": 250,
        },
    }

    try:
        resp = requests.post(_OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        response_text: str = resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        # Fallback when Ollama is not running (for testing / demo)
        response_text = (
            f"[FALLBACK — Ollama offline] Symptom intake completed. "
            f"Identified symptoms processed by rule-based engine. "
            "Passing to Research Agent for differential diagnosis."
        )
    except Exception as exc:
        raise RuntimeError(f"Ollama call failed: {exc}") from exc

    log_llm_response("SymptomAgent", response_text, patient_id=patient_id)
    return response_text


def run_symptom_agent(state: PatientState) -> PatientState:
    """
    Execute the Symptom Analysis Agent node.

    Reads `raw_symptoms` from state, applies normalisation and scoring tools,
    calls the LLM for a clinical narrative, and writes results back to state.

    Args:
        state: The current PatientState from LangGraph.

    Returns:
        Updated PatientState with symptom fields populated.
    """
    pid = state.get("patient_id", "UNKNOWN")
    trace: list[str] = list(state.get("agent_trace") or [])
    errors: list[str] = list(state.get("error_log") or [])

    trace.append(log_event("AGENT_START", "SymptomAgent", {}, patient_id=pid))

    raw = state.get("raw_symptoms", "")

    # ── Tool 1: Normalize symptoms ─────────────────────────────────────────
    try:
        log_tool_call("SymptomAgent", "normalize_symptoms", {"raw_text": raw[:80]}, patient_id=pid)
        symptoms = normalize_symptoms(raw)
        trace.append(
            log_tool_result("SymptomAgent", "normalize_symptoms",
                            str(symptoms), patient_id=pid)
        )
    except Exception as exc:
        err_msg = log_error("SymptomAgent", exc, patient_id=pid)
        errors.append(err_msg)
        symptoms = [raw.strip().lower()]  # Graceful degradation

    # ── Tool 2: Score severity ─────────────────────────────────────────────
    try:
        log_tool_call("SymptomAgent", "score_severity", {"symptoms": symptoms}, patient_id=pid)
        score, score_rationale = score_severity(symptoms)
        trace.append(
            log_tool_result("SymptomAgent", "score_severity",
                            f"score={score}", patient_id=pid)
        )
    except Exception as exc:
        err_msg = log_error("SymptomAgent", exc, patient_id=pid)
        errors.append(err_msg)
        score, score_rationale = 5, "Default score applied due to tool error."

    # ── LLM Reasoning ─────────────────────────────────────────────────────
    llm_prompt = (
        f"Patient reported: \"{raw}\"\n\n"
        f"After normalisation, identified symptoms: {', '.join(symptoms)}.\n"
        f"Severity scoring: {score_rationale}\n\n"
        "Write a brief clinical intake note."
    )

    try:
        analysis = _call_ollama(llm_prompt, pid)
    except RuntimeError as exc:
        err_msg = log_error("SymptomAgent", exc, patient_id=pid)
        errors.append(err_msg)
        analysis = f"Symptom processing complete. Symptoms: {', '.join(symptoms)}. Severity: {score}/10."

    trace.append(log_event("AGENT_END", "SymptomAgent",
                           {"symptoms_count": len(symptoms), "score": score},
                           patient_id=pid))

    return {
        **state,
        "structured_symptoms": symptoms,
        "severity_score": score,
        "symptom_analysis": analysis,
        "agent_trace": trace,
        "error_log": errors,
    }