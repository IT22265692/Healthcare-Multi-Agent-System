
from __future__ import annotations

import sys
from typing import Any

from langgraph.graph import StateGraph, END

from state import PatientState
from agents.symptom_agent import run_symptom_agent
from agents.research_agent import run_research_agent
from agents.triage_agent import run_triage_agent
from agents.report_agent import run_report_agent
from observability import log_event, logger


#  Build the LangGraph 

def build_graph() -> Any:
    """
    Construct and compile the LangGraph state machine.

    Returns:
        A compiled LangGraph application ready for .invoke().
    """
    graph = StateGraph(PatientState)

    # Register nodes
    graph.add_node("symptom_agent", run_symptom_agent)
    graph.add_node("research_agent", run_research_agent)
    graph.add_node("triage_agent", run_triage_agent)
    graph.add_node("report_agent", run_report_agent)

    # Define sequential pipeline edges
    graph.set_entry_point("symptom_agent")
    graph.add_edge("symptom_agent", "research_agent")
    graph.add_edge("research_agent", "triage_agent")
    graph.add_edge("triage_agent", "report_agent")
    graph.add_edge("report_agent", END)

    return graph.compile()


def run_pipeline(patient_id: str, raw_symptoms: str) -> PatientState:
    """
    Run the complete Healthcare MAS pipeline for a single patient.

    Args:
        patient_id:   A unique string identifier for the patient.
        raw_symptoms: Free-text symptoms as entered by the patient.

    Returns:
        The final PatientState after all 4 agents have executed.
    """
    logger.info(f"{'='*60}")
    logger.info(f"Healthcare MAS — Starting pipeline for patient: {patient_id}")
    logger.info(f"{'='*60}")

    # Initialise state
    initial_state: PatientState = {
        "patient_id": patient_id,
        "raw_symptoms": raw_symptoms,
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

    app = build_graph()
    final_state: PatientState = app.invoke(initial_state)

    logger.info(f"{'='*60}")
    logger.info(f"Pipeline complete for {patient_id}")
    logger.info(f"Triage Level : {final_state.get('triage_level')}")
    logger.info(f"Report saved : {final_state.get('report_filepath')}")
    logger.info(f"Trace entries: {len(final_state.get('agent_trace', []))}")
    if final_state.get("error_log"):
        logger.warning(f"Errors       : {len(final_state['error_log'])}")
    logger.info(f"{'='*60}")

    return final_state


#  CLI entry point 

if __name__ == "__main__":
    # Demo cases
    demo_cases = [
        {
            "patient_id": "P001",
            "symptoms": "I have a fever, runny nose and I feel really tired for the past 3 days",
        },
        {
            "patient_id": "P002",
            "symptoms": "severe chest pain and difficulty breathing, started 30 minutes ago",
        },
        {
            "patient_id": "P003",
            "symptoms": "headache and sore throat with mild fever",
        },
    ]

    # Allow CLI override: python main.py "P999" "my symptoms here"
    if len(sys.argv) == 3:
        demo_cases = [{"patient_id": sys.argv[1], "symptoms": sys.argv[2]}]

    for case in demo_cases:
        print(f"\n{'#'*60}")
        print(f"Processing: {case['patient_id']}")
        print(f"{'#'*60}")
        result = run_pipeline(case["patient_id"], case["symptoms"])

        # Print report to console
        print("\n" + (result.get("final_report") or "No report generated."))
        print(f"\n Report saved to: {result.get('report_filepath')}")