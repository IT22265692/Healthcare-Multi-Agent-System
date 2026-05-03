"""
observability.py — LLMOps tracing and structured logging for the Healthcare MAS.

Every agent call, tool invocation, and LLM response is captured here
with timestamps and structured JSON entries so the full execution can
be replayed, audited, or fed into a monitoring dashboard.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


#  File + console handler 
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

_LOG_FILE = LOG_DIR / f"mas_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("healthcare_mas")

_file_handler = logging.FileHandler(_LOG_FILE)
_file_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_file_handler)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_event(
    event_type: str,
    agent: str,
    details: Dict[str, Any],
    patient_id: Optional[str] = None,
) -> str:
    """
    Record a structured log event and return the human-readable summary line.

    Args:
        event_type: One of 'AGENT_START', 'TOOL_CALL', 'TOOL_RESULT',
                    'LLM_CALL', 'LLM_RESPONSE', 'AGENT_END', 'ERROR'.
        agent:      Name of the agent emitting the event.
        details:    Arbitrary key-value payload.
        patient_id: Optional patient identifier for correlation.

    Returns:
        A concise human-readable string suitable for PatientState.agent_trace.
    """
    payload = {
        "timestamp": _now(),
        "event_type": event_type,
        "agent": agent,
        "patient_id": patient_id,
        **details,
    }
    # Write JSONL to file
    _file_handler.stream.write(json.dumps(payload) + "\n")
    _file_handler.stream.flush()

    # Human-readable console line
    summary = f"[{agent}] {event_type}"
    if "tool" in details:
        summary += f" -> tool={details['tool']}"
    if "input_preview" in details:
        preview = str(details["input_preview"])[:80]
        summary += f" | input='{preview}'"
    if "output_preview" in details:
        preview = str(details["output_preview"])[:80]
        summary += f" | output='{preview}'"
    if "error" in details:
        summary += f" | ERROR: {details['error']}"

    if event_type == "ERROR":
        logger.error(summary)
    else:
        logger.info(summary)

    return summary


def log_llm_call(agent: str, prompt_preview: str, patient_id: Optional[str] = None) -> str:
    """Convenience wrapper for LLM invocation events."""
    return log_event(
        "LLM_CALL", agent,
        {"input_preview": prompt_preview[:200]},
        patient_id=patient_id,
    )


def log_llm_response(agent: str, response_preview: str, patient_id: Optional[str] = None) -> str:
    """Convenience wrapper for LLM response events."""
    return log_event(
        "LLM_RESPONSE", agent,
        {"output_preview": response_preview[:200]},
        patient_id=patient_id,
    )


def log_tool_call(agent: str, tool_name: str, args: Dict[str, Any],
                  patient_id: Optional[str] = None) -> str:
    """Convenience wrapper for tool invocations."""
    return log_event(
        "TOOL_CALL", agent,
        {"tool": tool_name, "args": args},
        patient_id=patient_id,
    )


def log_tool_result(agent: str, tool_name: str, result_preview: str,
                    patient_id: Optional[str] = None) -> str:
    """Convenience wrapper for tool result events."""
    return log_event(
        "TOOL_RESULT", agent,
        {"tool": tool_name, "output_preview": result_preview[:200]},
        patient_id=patient_id,
    )


def log_error(agent: str, error: Exception, patient_id: Optional[str] = None) -> str:
    """Convenience wrapper for error events."""
    return log_event(
        "ERROR", agent,
        {"error": str(error), "error_type": type(error).__name__},
        patient_id=patient_id,
    )