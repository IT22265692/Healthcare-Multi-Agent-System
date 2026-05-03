"""
agents/report_agent.py — Report Generation Agent (Agent 4 of 4).

Persona: "Dr. Scribe" — a meticulous medical documentation specialist whose
role is to consolidate all prior agent outputs into a clear, professional,
and actionable patient triage report, then persist it to disk.

Responsibilities:
    - Compile a structured report from the complete PatientState.
    - Save the report as a timestamped .txt file.
    - Populate PatientState: final_report, report_filepath.

Constraints:
    - MUST use compile_report and save_report tools.
    - MUST include all agent reasoning — no information loss.
    - MUST add the safety disclaimer.
    - MUST confirm file save success in the trace log.
    - MUST NOT alter any clinical findings from prior agents.
"""

from __future__ import annotations

import requests

from state import PatientState
from observability import (
    log_event, log_llm_call, log_llm_response,
    log_tool_call, log_tool_result, log_error,
)
from tools.report_tools import compile_report, save_report


_OLLAMA_URL = "http://localhost:11434/api/generate"
_MODEL = "llama3:8b"

_SYSTEM_PROMPT = """You are Dr. Scribe, a precise medical documentation AI.

Your ONLY role is to write a brief executive summary paragraph (3-4 sentences) 
that synthesises the patient's case for the top of their report.

RULES:
1. Be factual — reference only what the previous agents found.
2. Mention the triage level prominently.
3. Write in third-person clinical style ("The patient presents with...").
4. Keep it to exactly 3-4 sentences.
5. Do NOT add new diagnoses or recommendations.
"""


def _call_ollama(prompt: str, patient_id: str) -> str:
    """
    Call local Ollama LLM to generate an executive summary paragraph.

    Args:
        prompt:     Formatted prompt with full case data.
        patient_id: For trace correlation.

    Returns:
        Executive summary string.

    Raises:
        RuntimeError: If the LLM call fails completely.
    """
    log_llm_call("ReportAgent", prompt, patient_id=patient_id)

    payload = {
        "model": _MODEL,
        "system": _SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 200,
        },
    }

    try:
        resp = requests.post(_OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        response_text: str = resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        response_text = (
            f"[FALLBACK] The patient presents with a documented symptom cluster "
            "processed through the Healthcare MAS pipeline. "
            "All agent findings have been consolidated into this report."
        )
    except Exception as exc:
        raise RuntimeError(f"Ollama call failed: {exc}") from exc

    log_llm_response("ReportAgent", response_text, patient_id=patient_id)
    return response_text


def run_report_agent(state: PatientState) -> PatientState:
    """
    Execute the Report Generation Agent node.

    Reads the full PatientState, compiles a structured report using all
    prior agent outputs, generates an LLM executive summary, saves the
    report to disk, and updates state with final_report and report_filepath.

    Args:
        state: Current PatientState (must contain all prior agent outputs).

    Returns:
        Updated PatientState with final_report and report_filepath populated.
    """
    pid = state.get("patient_id", "UNKNOWN")
    trace: list[str] = list(state.get("agent_trace") or [])
    errors: list[str] = list(state.get("error_log") or [])

    symptoms = state.get("structured_symptoms") or []
    conditions = state.get("possible_conditions") or []
    red_flags = state.get("red_flags") or []

    trace.append(log_event("AGENT_START", "ReportAgent", {}, patient_id=pid))

    # ── LLM: Generate executive summary ───────────────────────────────────
    llm_prompt = (
        f"Patient ID: {pid}\n"
        f"Symptoms: {', '.join(symptoms)}\n"
        f"Severity: {state.get('severity_score', 'N/A')}/10\n"
        f"Top conditions: {', '.join(conditions[:3])}\n"
        f"Red flags: {len(red_flags)} detected\n"
        f"Triage level: {state.get('triage_level', 'N/A')}\n\n"
        "Write an executive summary paragraph for this patient's report."
    )

    try:
        executive_summary = _call_ollama(llm_prompt, pid)
    except RuntimeError as exc:
        err_msg = log_error("ReportAgent", exc, patient_id=pid)
        errors.append(err_msg)
        executive_summary = (
            f"Patient {pid} presents with {len(symptoms)} reported symptom(s). "
            f"Triage level: {state.get('triage_level', 'N/A')}."
        )

    # Inject executive summary into symptom_analysis for display
    enriched_analysis = (
        f"EXECUTIVE SUMMARY:\n{executive_summary}\n\n"
        f"SYMPTOM AGENT ANALYSIS:\n{state.get('symptom_analysis', '')}"
    )

    # ── Tool 1: Compile structured report ─────────────────────────────────
    try:
        log_tool_call("ReportAgent", "compile_report",
                      {"patient_id": pid}, patient_id=pid)
        report_text = compile_report(
            patient_id=pid,
            raw_symptoms=state.get("raw_symptoms", ""),
            structured_symptoms=symptoms,
            severity_score=state.get("severity_score", 0),
            symptom_analysis=enriched_analysis,
            possible_conditions=conditions,
            research_summary=state.get("research_summary", ""),
            red_flags=red_flags,
            triage_level=state.get("triage_level", "ROUTINE"),
            triage_reasoning=state.get("triage_reasoning", ""),
            triage_advice=state.get("triage_reasoning", "Please see a doctor."),
        )
        trace.append(
            log_tool_result("ReportAgent", "compile_report",
                            f"report length={len(report_text)} chars", patient_id=pid)
        )
    except Exception as exc:
        err_msg = log_error("ReportAgent", exc, patient_id=pid)
        errors.append(err_msg)
        report_text = f"Report compilation failed: {exc}"

    # ── Tool 2: Save report to file ────────────────────────────────────────
    try:
        log_tool_call("ReportAgent", "save_report",
                      {"patient_id": pid}, patient_id=pid)
        filepath = save_report(report_text, pid)
        trace.append(
            log_tool_result("ReportAgent", "save_report",
                            f"saved to {filepath}", patient_id=pid)
        )
    except Exception as exc:
        err_msg = log_error("ReportAgent", exc, patient_id=pid)
        errors.append(err_msg)
        filepath = "SAVE_FAILED"

    trace.append(log_event("AGENT_END", "ReportAgent",
                           {"report_saved": filepath != "SAVE_FAILED",
                            "filepath": filepath},
                           patient_id=pid))

    return {
        **state,
        "final_report": report_text,
        "report_filepath": filepath,
        "agent_trace": trace,
        "error_log": errors,
    }