"""
tools/triage_tools.py — Custom tools for the Triage Agent.

Tools:
    - classify_triage_level: Maps severity score + red flags to a triage tier.
    - generate_triage_advice: Produces actionable next-step recommendations.
"""

from __future__ import annotations

from typing import List, Tuple


# Manchester Triage System-inspired thresholds
_TRIAGE_THRESHOLDS: list[tuple[int, str]] = [
    (9, "EMERGENCY"),   # Immediate life threat — resuscitation room
    (7, "URGENT"),      # Serious — seen within 30 minutes
    (4, "SEMI-URGENT"), # Stable — seen within 2 hours
    (1, "ROUTINE"),     # Non-urgent — seen within 4 hours
]

_LEVEL_ADVICE: dict[str, str] = {
    "EMERGENCY": (
        "Call emergency services (119/911) immediately or proceed to the nearest "
        "Emergency Department without delay. Do not drive yourself."
    ),
    "URGENT": (
        "Seek assessment at an Urgent Care centre or Emergency Department within "
        "the next 1–2 hours. Have someone accompany you if possible."
    ),
    "SEMI-URGENT": (
        "Book an appointment with your General Practitioner today or attend a "
        "walk-in clinic within 4–6 hours."
    ),
    "ROUTINE": (
        "Schedule a routine GP appointment within the next 1–3 days. "
        "Monitor your symptoms and return if they worsen."
    ),
}


def classify_triage_level(
    severity_score: int,
    red_flags: List[str],
) -> Tuple[str, str]:
    """
    Classify triage urgency using the severity score and detected red flags.

    Any presence of red flags automatically elevates the triage level to
    at least URGENT, regardless of the numeric score.

    Args:
        severity_score: Integer 1–10 from score_severity().
        red_flags:      List of red-flag strings from extract_red_flags().

    Returns:
        Tuple of (level: str, reasoning: str).
        level is one of: 'EMERGENCY', 'URGENT', 'SEMI-URGENT', 'ROUTINE'.

    Raises:
        ValueError: If severity_score is outside the range 1–10.

    Example:
        >>> classify_triage_level(8, ["⚠️  CHEST PAIN detected"])
        ('EMERGENCY', 'Severity score 8 ≥ 7 and red flags present → EMERGENCY')
    """
    if not (1 <= severity_score <= 10):
        raise ValueError(f"severity_score must be 1–10, got {severity_score}.")

    # Determine level from score alone
    level = "ROUTINE"
    for threshold, tier in _TRIAGE_THRESHOLDS:
        if severity_score >= threshold:
            level = tier
            break

    # Red flags force elevation
    if red_flags:
        if level in ("ROUTINE", "SEMI-URGENT"):
            level = "URGENT"
        elif level == "URGENT":
            level = "EMERGENCY"
        reasoning = (
            f"Severity score {severity_score} combined with {len(red_flags)} red flag(s) "
            f"detected → triage elevated to {level}."
        )
    else:
        reasoning = (
            f"Severity score {severity_score} maps to {level} per Manchester Triage "
            "thresholds; no red flags detected."
        )

    return level, reasoning


def generate_triage_advice(level: str) -> str:
    """
    Generate patient-facing actionable advice for the given triage level.

    Args:
        level: Triage level string — one of 'EMERGENCY', 'URGENT',
               'SEMI-URGENT', or 'ROUTINE'.

    Returns:
        A plain-English advice string suitable for inclusion in the patient report.

    Raises:
        KeyError: If level is not a recognised triage tier.

    Example:
        >>> generate_triage_advice("ROUTINE")
        'Schedule a routine GP appointment within the next 1–3 days...'
    """
    if level not in _LEVEL_ADVICE:
        raise KeyError(f"Unknown triage level: '{level}'. "
                       f"Valid levels: {list(_LEVEL_ADVICE.keys())}")
    return _LEVEL_ADVICE[level]