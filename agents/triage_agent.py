"""
agents/triage_agent.py — Clinical Triage Agent (Agent 3 of 4).

Persona: "Nurse Triage" — an experienced emergency triage nurse whose role
is to apply the Manchester Triage System and determine the urgency level
for the patient, along with clear actionable advice.

Responsibilities:
    - Apply severity score and red flags to classify triage urgency.
    - Generate patient-facing actionable next-step advice.
    - Produce an LLM-reasoned triage justification.
    - Populate PatientState: triage_level, triage_reasoning.

Constraints:
    - MUST use classify_triage_level and generate_triage_advice tools.
    - MUST always err on the side of caution (escalate, never downgrade risk).
    - MUST provide clear, jargon-free advice for the patient.
    - MUST explicitly state if the situation requires emergency services.
    - MUST NOT overrule an EMERGENCY classification.
"""

from __future__ import annotations

import requests

from state import PatientState
from observability import (
    log_event, log_llm_call, log_llm_response,
    log_tool_call, log_tool_result, log_error,
)
from tools.triage_tools import classify_triage_level, generate_triage_advice


_OLLAMA_URL = "http://localhost:11434/api/generate"
_MODEL = "llama3:8b"

_SYSTEM_PROMPT = """You are Nurse Triage, an experienced emergency department triage nurse AI.

Your ONLY role is to justify the triage classification and provide clear patient guidance.

STRICT RULES:
1. NEVER downgrade a triage level — always err on the side of caution.
2. If triage is EMERGENCY, state this prominently and advise calling emergency services.
3. Write in plain English the patient can understand — no unexplained medical jargon.
4. Reference the specific symptoms and conditions that drove the decision.
5. Keep your response to 3–4 sentences.
6. End with: "Triage assessment complete. Passing to Report Agent."

You are not diagnosing — you are determining urgency and safe next steps only.
"""


def _call_ollama(prompt: str, patient_id: str) -> str:
    """
    Call local Ollama LLM for triage reasoning.

    Args:
        prompt:     Formatted triage context prompt.
        patient_id: For trace correlation.

    Returns:
        LLM triage reasoning string.

    Raises:
        RuntimeError: If the LLM call fails completely.
    """
    log_llm_call("TriageAgent", prompt, patient_id=patient_id)

    payload = {
        "model": _MODEL,
        "system": _SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.05,  # Very low — triage decisions must be consistent
            "num_predict": 200,
        },
    }

    try:
        resp = requests.post(_OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        response_text: str = resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        response_text = (
            "[FALLBACK — Ollama offline] Triage classification applied based on "
            "severity score and red flag analysis. "
            "Triage assessment complete. Passing to Report Agent."
        )
    except Exception as exc:
        raise RuntimeError(f"Ollama call failed: {exc}") from exc

    log_llm_response("TriageAgent", response_text, patient_id=patient_id)
    return response_text


def run_triage_agent(state: PatientState) -> PatientState:
    """
    Execute the Triage Agent node.

    Reads severity_score and red_flags from state, applies Manchester Triage
    System logic, generates patient advice, and updates state.

    Args:
        state: Current PatientState (must contain severity_score, red_flags).

    Returns:
        Updated PatientState with triage_level and triage_reasoning populated.
    """
    pid = state.get("patient_id", "UNKNOWN")
    trace: list[str] = list(state.get("agent_trace") or [])
    errors: list[str] = list(state.get("error_log") or [])
    severity = state.get("severity_score", 5)
    red_flags = state.get("red_flags") or []
    conditions = state.get("possible_conditions") or []
    symptoms = state.get("structured_symptoms") or []

    trace.append(log_event("AGENT_START", "TriageAgent", {}, patient_id=pid))

    # ── Tool 1: Classify triage level ─────────────────────────────────────
    try:
        log_tool_call("TriageAgent", "classify_triage_level",
                      {"severity_score": severity, "red_flags_count": len(red_flags)},
                      patient_id=pid)
        level, tool_reasoning = classify_triage_level(severity, red_flags)
        trace.append(
            log_tool_result("TriageAgent", "classify_triage_level",
                            f"level={level}", patient_id=pid)
        )
    except Exception as exc:
        err_msg = log_error("TriageAgent", exc, patient_id=pid)
        errors.append(err_msg)
        level, tool_reasoning = "URGENT", "Default URGENT level applied due to tool error."

    # ── Tool 2: Generate actionable advice ────────────────────────────────
    try:
        log_tool_call("TriageAgent", "generate_triage_advice",
                      {"level": level}, patient_id=pid)
        advice = generate_triage_advice(level)
        trace.append(
            log_tool_result("TriageAgent", "generate_triage_advice",
                            advice[:80], patient_id=pid)
        )
    except Exception as exc:
        err_msg = log_error("TriageAgent", exc, patient_id=pid)
        errors.append(err_msg)
        advice = "Please seek medical attention promptly."

    # ── LLM Reasoning ─────────────────────────────────────────────────────
    llm_prompt = (
        f"Patient symptoms: {', '.join(symptoms)}\n"
        f"Severity score: {severity}/10\n"
        f"Possible conditions: {', '.join(conditions[:3])}\n"
        f"Red flags detected: {len(red_flags)}\n"
        f"Triage classification: {level}\n"
        f"Tool reasoning: {tool_reasoning}\n\n"
        f"Write a patient-facing triage justification and advice."
    )

    try:
        reasoning = _call_ollama(llm_prompt, pid)
    except RuntimeError as exc:
        err_msg = log_error("TriageAgent", exc, patient_id=pid)
        errors.append(err_msg)
        reasoning = f"{tool_reasoning} {advice}"

    trace.append(log_event("AGENT_END", "TriageAgent",
                           {"triage_level": level}, patient_id=pid))

    return {
        **state,
        "triage_level": level,
        "triage_reasoning": reasoning,
        "agent_trace": trace,
        "error_log": errors,
    }