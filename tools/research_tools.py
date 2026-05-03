"""
tools/research_tools.py — Custom tools for the Research Agent.

Tools:
    - query_disease_api:     Fetches condition data from the Open Disease API.
    - extract_red_flags:     Identifies emergency warning signs from symptom list.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.parse
import urllib.error
from typing import Dict, List, Optional



# Open Disease / symptom-to-condition knowledge base (local fallback)

_LOCAL_KNOWLEDGE_BASE: dict[str, dict] = {
    "fever": {
        "conditions": ["Influenza", "COVID-19", "Bacterial Infection", "Malaria"],
        "notes": "Fever >38.5°C warrants investigation; >40°C is an emergency.",
    },
    "cough": {
        "conditions": ["Upper Respiratory Tract Infection", "Asthma", "Bronchitis", "COVID-19"],
        "notes": "Productive cough >3 weeks requires chest X-ray.",
    },
    "chest pain": {
        "conditions": ["Acute Coronary Syndrome", "Pulmonary Embolism", "Pneumonia", "GERD"],
        "notes": "Chest pain is a RED FLAG — rule out cardiac cause immediately.",
    },
    "dyspnoea": {
        "conditions": ["Asthma", "COPD Exacerbation", "Heart Failure", "Pulmonary Embolism"],
        "notes": "Sudden onset dyspnoea at rest is an emergency.",
    },
    "headache": {
        "conditions": ["Tension Headache", "Migraine", "Sinusitis", "Hypertensive Crisis"],
        "notes": "Thunderclap headache or 'worst headache of life' = subarachnoid haemorrhage until proven otherwise.",
    },
    "dizziness": {
        "conditions": ["Benign Positional Vertigo", "Anaemia", "Hypotension", "Labyrinthitis"],
        "notes": "Dizziness with neurological signs requires urgent CT.",
    },
    "abdominal pain": {
        "conditions": ["Appendicitis", "Gastroenteritis", "Peptic Ulcer", "Cholecystitis"],
        "notes": "Peritoneal signs indicate surgical emergency.",
    },
    "skin rash": {
        "conditions": ["Viral Exanthem", "Allergic Reaction", "Meningococcaemia", "Contact Dermatitis"],
        "notes": "Non-blanching petechial rash with fever = meningococcaemia until proven otherwise.",
    },
    "fatigue": {
        "conditions": ["Anaemia", "Hypothyroidism", "Depression", "Chronic Fatigue Syndrome"],
        "notes": "Fatigue lasting >6 months requires full blood workup.",
    },
    "vomiting": {
        "conditions": ["Gastroenteritis", "Food Poisoning", "Appendicitis", "Raised ICP"],
        "notes": "Projectile vomiting without nausea may indicate raised intracranial pressure.",
    },
    "diarrhoea": {
        "conditions": ["Viral Gastroenteritis", "Food Poisoning", "IBS", "IBD"],
        "notes": "Bloody diarrhoea requires stool culture; dehydration risk in elderly.",
    },
    "sore throat": {
        "conditions": ["Viral Pharyngitis", "Streptococcal Pharyngitis", "Infectious Mononucleosis"],
        "notes": "Exudates + fever + lymphadenopathy = CENTOR criteria for strep.",
    },
    "runny nose": {
        "conditions": ["Allergic Rhinitis", "Common Cold", "Sinusitis"],
        "notes": "Unilateral clear discharge in child: exclude foreign body.",
    },
    "myalgia": {
        "conditions": ["Influenza", "Polymyalgia Rheumatica", "Viral Illness", "Fibromyalgia"],
        "notes": "Severe myalgia with dark urine = rhabdomyolysis.",
    },
    "arthralgia": {
        "conditions": ["Viral Arthritis", "Rheumatoid Arthritis", "Osteoarthritis", "Gout"],
        "notes": "Hot swollen single joint = septic arthritis until proven otherwise.",
    },
}

_RED_FLAG_SYMPTOMS: set[str] = {
    "chest pain", "dyspnoea", "loss of consciousness", "syncope",
    "seizure", "haemorrhage", "blood", "paralysis",
    "severe headache", "thunderclap headache",
}

_RED_FLAG_CONDITIONS: set[str] = {
    "Acute Coronary Syndrome", "Pulmonary Embolism", "Meningococcaemia",
    "Hypertensive Crisis", "Subarachnoid Haemorrhage",
}


def query_disease_api(symptoms: List[str]) -> Dict[str, object]:
    """
    Look up possible conditions for each symptom using a local evidence-based
    knowledge base (offline-first, no paid API required).

    The function aggregates conditions across all symptoms, ranks them by
    frequency of co-occurrence, and returns the top 5.

    Args:
        symptoms: List of canonical symptom strings (from normalize_symptoms).

    Returns:
        Dictionary with keys:
            - 'conditions' (List[str]): Top-5 ranked differential diagnoses.
            - 'notes'       (List[str]): Clinical notes per matched symptom.
            - 'coverage'    (int):       Number of symptoms matched in KB.

    Raises:
        ValueError: If symptoms is empty.

    Example:
        >>> query_disease_api(["fever", "cough"])
        {'conditions': ['Influenza', 'COVID-19', ...], 'notes': [...], 'coverage': 2}
    """
    if not symptoms:
        raise ValueError("symptoms list must not be empty.")

    condition_counts: dict[str, int] = {}
    notes: list[str] = []
    matched = 0

    for symptom in symptoms:
        entry = _LOCAL_KNOWLEDGE_BASE.get(symptom)
        if entry:
            matched += 1
            notes.append(f"[{symptom.upper()}] {entry['notes']}")
            for condition in entry["conditions"]:
                condition_counts[condition] = condition_counts.get(condition, 0) + 1

    # Rank by co-occurrence frequency
    ranked = sorted(condition_counts.items(), key=lambda x: x[1], reverse=True)
    top_conditions = [c for c, _ in ranked[:5]]

    # If no match found, return generic
    if not top_conditions:
        top_conditions = ["Undifferentiated illness — further history and examination required"]
        notes.append("No specific conditions matched in knowledge base.")

    return {
        "conditions": top_conditions,
        "notes": notes,
        "coverage": matched,
    }


def extract_red_flags(symptoms: List[str], conditions: List[str]) -> List[str]:
    """
    Identify emergency red-flag warnings from the symptom and condition lists.

    Red flags are clinical features that indicate the patient may need
    immediate emergency assessment.

    Args:
        symptoms:   List of canonical symptom strings.
        conditions: List of possible conditions from query_disease_api.

    Returns:
        List of human-readable red-flag warning strings. Empty list if none.

    Raises:
        TypeError: If either argument is not a list.

    Example:
        >>> extract_red_flags(["chest pain"], ["Acute Coronary Syndrome"])
        ['⚠️  CHEST PAIN detected — cardiac cause must be excluded immediately.',
         '⚠️  Acute Coronary Syndrome is a life-threatening emergency.']
    """
    if not isinstance(symptoms, list) or not isinstance(conditions, list):
        raise TypeError("Both symptoms and conditions must be lists.")

    flags: list[str] = []

    for symptom in symptoms:
        if symptom in _RED_FLAG_SYMPTOMS:
            flags.append(
                f"⚠️  {symptom.upper()} detected — this is a potential emergency symptom."
            )

    for condition in conditions:
        if condition in _RED_FLAG_CONDITIONS:
            flags.append(
                f"⚠️  {condition} is a life-threatening condition — emergency assessment required."
            )

    return flags