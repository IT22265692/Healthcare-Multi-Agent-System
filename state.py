"""
state.py — Global state schema for the Healthcare MAS.

All agents read from and write to this TypedDict, which LangGraph
threads through the graph so no context is ever lost between handoffs.
"""

from __future__ import annotations
from typing import TypedDict, Optional, List


class PatientState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    patient_id: str
    raw_symptoms: str                    # Free-text from the user

    # ── Symptom Agent output ────────────────────────────────────────────────
    structured_symptoms: Optional[List[str]]     # Cleaned, deduplicated list
    severity_score: Optional[int]                # 1-10 scale
    symptom_analysis: Optional[str]              # LLM reasoning narrative

    # ── Research Agent output ───────────────────────────────────────────────
    possible_conditions: Optional[List[str]]     # Ranked differential list
    research_summary: Optional[str]              # Evidence-based reasoning
    red_flags: Optional[List[str]]               # Urgent warning signs found

    # ── Triage Agent output ─────────────────────────────────────────────────
    triage_level: Optional[str]                  # EMERGENCY / URGENT / ROUTINE
    triage_reasoning: Optional[str]              # Why this triage level

    # ── Report Agent output ─────────────────────────────────────────────────
    final_report: Optional[str]                  # Full patient report text
    report_filepath: Optional[str]               # Path to saved .txt report

    # ── Observability ───────────────────────────────────────────────────────
    agent_trace: Optional[List[str]]             # Ordered list of agent log entries
    error_log: Optional[List[str]]               # Any errors caught during execution