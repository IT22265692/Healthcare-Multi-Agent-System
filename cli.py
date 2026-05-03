"""
cli.py — Interactive CLI Chat Interface for the Healthcare MAS.

This is the main user-facing entry point. The user types their symptoms
in plain English, and the full 4-agent pipeline runs automatically,
printing each agent's progress and the final triage report to the terminal.

Usage:
    python cli.py

The interface supports:
    - Free-text symptom input
    - Automatic patient ID generation
    - Live agent progress display
    - Coloured triage level output
    - Option to run another case or exit
"""

from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime

# ── ANSI colour codes ──────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
BLUE   = "\033[94m"
GREY   = "\033[90m"
WHITE  = "\033[97m"

LEVEL_COLOURS = {
    "EMERGENCY":   RED,
    "URGENT":      YELLOW,
    "SEMI-URGENT": CYAN,
    "ROUTINE":     GREEN,
}

LEVEL_ICONS = {
    "EMERGENCY":   "🚨",
    "URGENT":      "⚠️ ",
    "SEMI-URGENT": "🔵",
    "ROUTINE":     "✅",
}


def _colour(text: str, code: str) -> str:
    return f"{code}{text}{RESET}"


def _banner() -> None:
    """Print the application banner."""
    print()
    print(_colour("=" * 62, BLUE))
    print(_colour("  🏥  Healthcare AI Triage System", BOLD + WHITE))
    print(_colour("  Powered by LangGraph + Ollama (llama3:8b)", GREY))
    print(_colour("=" * 62, BLUE))
    print(_colour("  ⚠️  FOR DEMONSTRATION PURPOSES ONLY", YELLOW))
    print(_colour("  This system does NOT provide medical advice.", YELLOW))
    print(_colour("  Always consult a qualified healthcare professional.", YELLOW))
    print(_colour("=" * 62, BLUE))
    print()


def _agent_step(step: int, name: str, icon: str) -> None:
    """Print a live agent progress indicator."""
    print(_colour(f"  [{step}/4] {icon}  {name}...", CYAN), end="", flush=True)


def _agent_done(result_preview: str = "") -> None:
    """Print completion tick for an agent step."""
    preview = f" → {result_preview}" if result_preview else ""
    print(_colour(f"  ✓{preview}", GREEN))


def _print_report(state: dict) -> None:
    """Pretty-print the triage result to the terminal."""
    level     = state.get("triage_level", "UNKNOWN")
    symptoms  = state.get("structured_symptoms") or []
    score     = state.get("severity_score", 0)
    conditions = state.get("possible_conditions") or []
    red_flags  = state.get("red_flags") or []
    reasoning  = state.get("triage_reasoning", "")
    filepath   = state.get("report_filepath", "")
    pid        = state.get("patient_id", "")

    colour = LEVEL_COLOURS.get(level, WHITE)
    icon   = LEVEL_ICONS.get(level, "")

    print()
    print(_colour("─" * 62, BLUE))
    print(_colour(f"  TRIAGE RESULT FOR PATIENT {pid}", BOLD + WHITE))
    print(_colour("─" * 62, BLUE))

    # Triage level — most important, shown prominently
    print(f"\n  {icon}  Triage Level: {_colour(BOLD + level, colour)}\n")

    # Symptoms
    print(_colour("  Identified Symptoms:", BOLD))
    for s in symptoms:
        print(f"    • {s}")

    print(f"\n  Severity Score: {_colour(str(score) + '/10', BOLD)}")

    # Conditions
    print(_colour("\n  Possible Conditions (ranked):", BOLD))
    for i, c in enumerate(conditions[:5], 1):
        print(f"    {i}. {c}")

    # Red flags
    if red_flags:
        print(_colour("\n  ⚠️  RED FLAGS DETECTED:", BOLD + RED))
        for f in red_flags:
            print(_colour(f"    {f}", RED))

    # Reasoning
    if reasoning:
        print(_colour("\n  Triage Reasoning:", BOLD))
        # Wrap long reasoning text
        words = reasoning.replace("\n", " ").split()
        line, lines = [], []
        for word in words:
            line.append(word)
            if len(" ".join(line)) > 58:
                lines.append("  " + " ".join(line[:-1]))
                line = [word]
        if line:
            lines.append("  " + " ".join(line))
        for l in lines[:6]:   # Show first 6 lines max
            print(_colour(l, GREY))

    # Report saved
    if filepath and filepath != "SAVE_FAILED":
        print(_colour(f"\n  📄 Full report saved to:", BOLD))
        print(_colour(f"     {filepath}", GREY))

    print(_colour("\n" + "─" * 62, BLUE))


def _get_symptoms() -> str:
    """Prompt the user to enter their symptoms."""
    print(_colour("  Describe your symptoms in plain English.", WHITE))
    print(_colour("  (e.g. 'I have a fever, cough and feel very tired')", GREY))
    print()
    while True:
        try:
            symptoms = input(_colour("  Your symptoms: ", BOLD + CYAN)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return ""
        if len(symptoms) >= 3:
            return symptoms
        print(_colour("  ⚠️  Please enter at least a few words.", YELLOW))


def run_interactive() -> None:
    """Main interactive loop."""
    _banner()

    while True:
        # ── Get patient input ──────────────────────────────────────────
        symptoms = _get_symptoms()
        if not symptoms:
            print(_colour("\n  Goodbye! Stay healthy. 👋\n", GREEN))
            break

        patient_id = f"P{datetime.now().strftime('%H%M%S')}"
        print()
        print(_colour(f"  Patient ID: {patient_id}", GREY))
        print(_colour("  Running 4-agent triage pipeline...", GREY))
        print()

        # ── Import and run pipeline ────────────────────────────────────
        # Imports are deferred so banner shows before any langgraph startup noise
        try:
            from agents.symptom_agent import run_symptom_agent
            from agents.research_agent import run_research_agent
            from agents.triage_agent import run_triage_agent
            from agents.report_agent import run_report_agent
            from state import PatientState
        except ImportError as e:
            print(_colour(f"  ❌ Import error: {e}", RED))
            print(_colour("  Make sure you are running from the project root directory.", YELLOW))
            break

        state: PatientState = {
            "patient_id": patient_id,
            "raw_symptoms": symptoms,
            "structured_symptoms": None,
            "severity_score": None,
            "symptom_analysis": None,
            "possible_conditions": None,
            "research_summary": None,
            "red_flags": None,
            "triage_level": None,
            "triage_reasoning": None,
            "final_report": None,
            "report_filepath": None,
            "agent_trace": [],
            "error_log": [],
        }

        # Run agents one by one so we can show live progress
        _agent_step(1, "Symptom Agent  (normalising & scoring)", "🔬")
        state = run_symptom_agent(state)
        syms = state.get("structured_symptoms") or []
        _agent_done(f"{len(syms)} symptom(s) found, severity {state.get('severity_score')}/10")

        _agent_step(2, "Research Agent (differential diagnosis)", "📚")
        state = run_research_agent(state)
        conds = state.get("possible_conditions") or []
        flags = state.get("red_flags") or []
        _agent_done(f"{len(conds)} condition(s), {len(flags)} red flag(s)")

        _agent_step(3, "Triage Agent   (urgency classification)", "🏥")
        state = run_triage_agent(state)
        level = state.get("triage_level", "?")
        _agent_done(f"Level = {level}")

        _agent_step(4, "Report Agent   (compiling & saving)   ", "📄")
        state = run_report_agent(state)
        _agent_done("report saved")

        # ── Display result ─────────────────────────────────────────────
        _print_report(state)

        # ── Ask to continue ────────────────────────────────────────────
        print()
        try:
            again = input(_colour("  Run another case? (y/n): ", BOLD + CYAN)).strip().lower()
        except (EOFError, KeyboardInterrupt):
            again = "n"

        if again != "y":
            print(_colour("\n  Goodbye! Stay healthy. 👋\n", GREEN))
            break
        print()


if __name__ == "__main__":
    # Change to the script's directory so relative imports work
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_interactive()