"""
agents/research_agent.py — Medical Research Agent (Agent 2 of 4).

Persona: "Dr. Evidence" — a clinical decision-support specialist who applies
evidence-based medicine to generate a ranked differential diagnosis list and
flag any emergency red signs.

Responsibilities:
    - Query the medical knowledge base for each normalised symptom.
    - Produce a ranked differential diagnosis list.
    - Identify red-flag emergency warnings.
    - Write an LLM-reasoned evidence summary.
    - Populate PatientState: possible_conditions, research_summary, red_flags.

Constraints:
    - MUST use query_disease_api and extract_red_flags tools.
    - MUST prefix speculative conditions with "Possible" or "Consider".
    - MUST always list most serious conditions at the top.
    - MUST NOT recommend specific treatments or medications.
    - MUST pass ALL context forward — no information loss.
"""

from __future__ import annotations

import requests

from state import PatientState
from observability import (
    log_event, log_llm_call, log_llm_response,
    log_tool_call, log_tool_result, log_error,
)
from tools.research_tools import query_disease_api, extract_red_flags


_OLLAMA_URL = "http://localhost:11434/api/generate"
_MODEL = "llama3:8b"

_SYSTEM_PROMPT = """You are Dr. Evidence, a clinical decision-support AI specialising in evidence-based differential diagnosis.

Your ONLY role is to synthesise research findings into a clear, ranked differential diagnosis summary.

STRICT RULES:
1. Base your summary ONLY on the symptom data and research results provided — do not invent conditions.
2. List conditions in order from most to least serious.
3. Use qualifiers: "Most likely", "Consider", "Exclude urgently".
4. Do NOT prescribe treatments or medications.
5. Always mention if any red flags were detected.
6. Keep your response to 4–6 sentences.
7. End with: "Passing findings to Triage Agent."

NEVER fabricate medical statistics or studies. State uncertainty when present.
"""


def _call_ollama(prompt: str, patient_id: str) -> str:
    """
    Call local Ollama LLM for evidence-based reasoning.

    Args:
        prompt:     Formatted prompt with research data.
        patient_id: For trace correlation.

    Returns:
        LLM research summary string.

    Raises:
        RuntimeError: If the LLM call fails completely.
    """
    log_llm_call("ResearchAgent", prompt, patient_id=patient_id)

    payload = {
        "model": _MODEL,
        "system": _SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 300,
        },
    }

    try:
        resp = requests.post(_OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        response_text: str = resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        response_text = (
            "[FALLBACK — Ollama offline] Research complete. Conditions ranked by "
            "symptom co-occurrence frequency. Red flags assessed. "
            "Passing findings to Triage Agent."
        )
    except Exception as exc:
        raise RuntimeError(f"Ollama call failed: {exc}") from exc

    log_llm_response("ResearchAgent", response_text, patient_id=patient_id)
    return response_text


def run_research_agent(state: PatientState) -> PatientState:
    """
    Execute the Research Agent node.

    Reads structured_symptoms from state, queries the medical knowledge base,
    extracts red flags, generates an LLM evidence summary, and updates state.

    Args:
        state: Current PatientState (must contain structured_symptoms).

    Returns:
        Updated PatientState with research fields populated.
    """
    pid = state.get("patient_id", "UNKNOWN")
    trace: list[str] = list(state.get("agent_trace") or [])
    errors: list[str] = list(state.get("error_log") or [])
    symptoms = state.get("structured_symptoms") or []

    trace.append(log_event("AGENT_START", "ResearchAgent", {}, patient_id=pid))

    # ── Tool 1: Query disease knowledge base ───────────────────────────────
    try:
        log_tool_call("ResearchAgent", "query_disease_api",
                      {"symptoms": symptoms}, patient_id=pid)
        research = query_disease_api(symptoms)
        conditions: list[str] = research["conditions"]
        notes: list[str] = research["notes"]
        trace.append(
            log_tool_result("ResearchAgent", "query_disease_api",
                            f"conditions={conditions[:3]}", patient_id=pid)
        )
    except Exception as exc:
        err_msg = log_error("ResearchAgent", exc, patient_id=pid)
        errors.append(err_msg)
        conditions = ["Unable to retrieve conditions — manual review required"]
        notes = []

    # ── Tool 2: Extract red flags ──────────────────────────────────────────
    try:
        log_tool_call("ResearchAgent", "extract_red_flags",
                      {"symptoms": symptoms, "conditions": conditions}, patient_id=pid)
        red_flags = extract_red_flags(symptoms, conditions)
        trace.append(
            log_tool_result("ResearchAgent", "extract_red_flags",
                            f"{len(red_flags)} red flags", patient_id=pid)
        )
    except Exception as exc:
        err_msg = log_error("ResearchAgent", exc, patient_id=pid)
        errors.append(err_msg)
        red_flags = []

    # ── LLM Reasoning ─────────────────────────────────────────────────────
    notes_text = "\n".join(notes) if notes else "No specific clinical notes."
    flags_text = "\n".join(red_flags) if red_flags else "No red flags detected."

    llm_prompt = (
        f"Patient symptoms: {', '.join(symptoms)}\n"
        f"Severity score: {state.get('severity_score', 'N/A')}/10\n\n"
        f"Research results — possible conditions (ranked):\n"
        + "\n".join(f"  {i+1}. {c}" for i, c in enumerate(conditions))
        + f"\n\nClinical notes:\n{notes_text}\n\n"
        f"Red flags:\n{flags_text}\n\n"
        "Write a concise evidence-based differential diagnosis summary."
    )

    try:
        summary = _call_ollama(llm_prompt, pid)
    except RuntimeError as exc:
        err_msg = log_error("ResearchAgent", exc, patient_id=pid)
        errors.append(err_msg)
        summary = (
            f"Research complete. Top conditions: {', '.join(conditions[:3])}. "
            f"{len(red_flags)} red flag(s) identified."
        )

    trace.append(log_event("AGENT_END", "ResearchAgent",
                           {"conditions_found": len(conditions),
                            "red_flags": len(red_flags)},
                           patient_id=pid))

    return {
        **state,
        "possible_conditions": conditions,
        "research_summary": summary,
        "red_flags": red_flags,
        "agent_trace": trace,
        "error_log": errors,
    }