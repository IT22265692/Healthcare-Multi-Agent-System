"""
tools/symptom_tools.py — Custom tools for the Symptom Analysis Agent.

Tools:
    - normalize_symptoms: Cleans and deduplicates raw symptom text.
    - score_severity:     Assigns an evidence-based 1-10 severity score.
"""

from __future__ import annotations

import re
from typing import List, Tuple


# Known symptom vocabulary for normalisation

_SYNONYM_MAP: dict[str, str] = {
    "temp": "fever",
    "temperature": "fever",
    "pyrexia": "fever",
    "running nose": "runny nose",
    "rhinorrhoea": "runny nose",
    "cough": "cough",
    "throwing up": "vomiting",
    "nausea and vomiting": "nausea",
    "vomit": "vomiting",
    "headache": "headache",
    "head pain": "headache",
    "migraine": "headache",
    "sore throat": "sore throat",
    "pharyngitis": "sore throat",
    "shortness of breath": "dyspnoea",
    "difficulty breathing": "dyspnoea",
    "sob": "dyspnoea",
    "chest pain": "chest pain",
    "chest tightness": "chest pain",
    "dizzy": "dizziness",
    "lightheaded": "dizziness",
    "rash": "skin rash",
    "hives": "skin rash",
    "tired": "fatigue",
    "exhausted": "fatigue",
    "weakness": "fatigue",
    "stomach ache": "abdominal pain",
    "belly pain": "abdominal pain",
    "stomach pain": "abdominal pain",
    "diarrhoea": "diarrhoea",
    "diarrhea": "diarrhoea",
    "loose stools": "diarrhoea",
    "joint pain": "arthralgia",
    "muscle pain": "myalgia",
    "body aches": "myalgia",
}

# High-severity keywords that bump the score
_HIGH_SEVERITY_TERMS: set[str] = {
    "chest pain", "dyspnoea", "loss of consciousness",
    "syncope", "seizure", "stroke", "paralysis",
    "severe headache", "haemorrhage", "blood",
}

# Moderate-severity keywords
_MODERATE_SEVERITY_TERMS: set[str] = {
    "fever", "vomiting", "diarrhoea", "dizziness",
    "abdominal pain", "skin rash", "arthralgia", "myalgia",
}


def normalize_symptoms(raw_text: str) -> List[str]:
    """
    Clean, tokenise, normalise, and deduplicate a free-text symptom string.

    The function:
    1. Lowercases and strips punctuation.
    2. Splits on common delimiters (comma, semicolon, 'and', 'with', newline).
    3. Maps synonyms to canonical medical terms.
    4. Removes duplicates while preserving the order of first occurrence.

    Args:
        raw_text: Free-text patient-reported symptoms, e.g.
                  "I have a fever, running nose and I feel really tired".

    Returns:
        Ordered list of canonical symptom strings, e.g.
        ["fever", "runny nose", "fatigue"].

    Raises:
        ValueError: If raw_text is empty or contains only whitespace.

    Example:
        >>> normalize_symptoms("fever, cough and SOB")
        ['fever', 'cough', 'dyspnoea']
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("raw_text must be a non-empty string.")

    text = raw_text.lower()

    # Remove common filler phrases
    fillers = [
        r"\bi have\b", r"\bi am\b", r"\bi feel\b", r"\bfeeling\b",
        r"\bsuffering from\b", r"\bexperiencing\b", r"\bsince\b",
        r"\bfor the past\b", r"\bdays?\b", r"\bweeks?\b",
        r"\breally\b", r"\bvery\b", r"\bquite\b",
    ]
    for filler in fillers:
        text = re.sub(filler, "", text)
    # Strip leading articles after filler removal (e.g. "a fever" → "fever")
    text = re.sub(r"\b(a|an|the)\s+", " ", text)

    # Split on delimiters
    tokens = re.split(r"[,;\n]|\band\b|\bwith\b|\balso\b|\bplus\b", text)

    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        token = token.strip().strip(".")
        if not token:
            continue
        # Apply synonym mapping (check full token first, then sub-phrases)
        canonical = _SYNONYM_MAP.get(token, token)
        # Try partial matches for multi-word synonyms
        for phrase, mapped in _SYNONYM_MAP.items():
            if phrase in token:
                canonical = mapped
                break
        canonical = canonical.strip()
        if canonical and canonical not in seen:
            seen.add(canonical)
            result.append(canonical)

    return result if result else [raw_text.strip().lower()]


def score_severity(symptoms: List[str]) -> Tuple[int, str]:
    """
    Assign an evidence-based severity score (1–10) to a symptom list.

    Scoring logic:
    - Base score = number of symptoms (capped at 4).
    - Each HIGH_SEVERITY symptom adds +3 (max total: 10).
    - Each MODERATE_SEVERITY symptom adds +1.
    - Score is clamped to [1, 10].

    Args:
        symptoms: List of canonical symptom strings produced by
                  normalize_symptoms().

    Returns:
        Tuple of (score: int, rationale: str) where score is 1–10
        and rationale explains the contributing factors.

    Raises:
        ValueError: If symptoms list is empty.

    Example:
        >>> score_severity(["chest pain", "dyspnoea"])
        (10, "High-severity symptoms detected: chest pain, dyspnoea. Score: 10/10")
    """
    if not symptoms:
        raise ValueError("symptoms list must not be empty.")

    base = min(len(symptoms), 4)
    high_hits = [s for s in symptoms if s in _HIGH_SEVERITY_TERMS]
    moderate_hits = [s for s in symptoms if s in _MODERATE_SEVERITY_TERMS]

    score = base + (len(high_hits) * 3) + len(moderate_hits)
    score = max(1, min(10, score))

    parts: list[str] = [f"Base score from {len(symptoms)} symptom(s): {base}."]
    if high_hits:
        parts.append(f"High-severity symptoms detected: {', '.join(high_hits)} (+{len(high_hits)*3}).")
    if moderate_hits:
        parts.append(f"Moderate-severity symptoms detected: {', '.join(moderate_hits)} (+{len(moderate_hits)}).")
    parts.append(f"Final clamped score: {score}/10.")

    return score, " ".join(parts)