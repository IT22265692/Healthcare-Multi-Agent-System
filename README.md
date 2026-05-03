# Healthcare-Multi-Agent-System
**SE4010 CTSE — Assignment 2**

> A locally-hosted Multi-Agent System that automates clinical symptom triage using **LangGraph** + **Ollama (llama3:8b)** — 100% local, zero cloud cost.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.1+-green)
![Ollama](https://img.shields.io/badge/Ollama-llama3:8b-orange)
![Tests](https://img.shields.io/badge/Tests-196%20passed-brightgreen)

> ⚠️ **For educational/demonstration purposes only. Not medical advice.**

---

## Overview

A patient types their symptoms in plain English. Four specialised AI agents process the input through a LangGraph pipeline and produce a structured triage report.

| Agent | Persona | Role |
|-------|---------|------|
| Agent 1 — Symptom Agent | Dr. Intake | Parse + normalise symptoms, score severity 1–10 |
| Agent 2 — Research Agent | Dr. Evidence | Differential diagnosis + red flag detection |
| Agent 3 — Triage Agent | Nurse Triage | Manchester Triage System classification |
| Agent 4 — Report Agent | Dr. Scribe | Compile + save structured patient report |

---

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed
- llama3:8b model (~4.5 GB)

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-team/healthcare-mas.git
cd healthcare-mas

# 2. Install dependencies
pip install langgraph requests pytest

# 3. Pull the model
ollama pull llama3:8b

# 4. Start Ollama (keep running in a separate terminal)
ollama serve
```

---

## Running the System

```bash
# Interactive CLI — recommended
python cli.py

# Batch mode (3 demo cases)
python main.py

# Single custom case
python main.py "P001" "chest pain and shortness of breath"
```

**CLI demo:**
```
  Your symptoms: fever, cough and feel very tired

  [1/4] 🔬  Symptom Agent  ...  ✓ → 3 symptom(s), severity 4/10
  [2/4] 📚  Research Agent ...  ✓ → 5 condition(s), 0 red flag(s)
  [3/4] 🏥  Triage Agent   ...  ✓ → Level = SEMI-URGENT
  [4/4] 📄  Report Agent   ...  ✓ → report saved
```

---

## Running the Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run per student
python -m pytest tests/test_student1_symptom_agent.py -v
python -m pytest tests/test_student2_research_agent.py -v
python -m pytest tests/test_student3_triage_agent.py -v
python -m pytest tests/test_student4_report_agent.py -v
```

**Result:** `196 passed, 6 skipped` *(6 skipped = LLM-as-a-Judge, requires Ollama)*

---

## Individual Contributions

| | Agent | Tool File | Tests |
|--|-------|-----------|-------|
| **Student 1** | `symptom_agent.py` — Dr. Intake | `symptom_tools.py` | 43 |
| **Student 2** | `research_agent.py` — Dr. Evidence | `research_tools.py` | 43 |
| **Student 3** | `triage_agent.py` — Nurse Triage | `triage_tools.py` | 46 |
| **Student 4** | `report_agent.py` — Dr. Scribe | `report_tools.py` | 64 |

---

*SE4010 CTSE — Assignment 2 | Sri Lanka Institute of Information Technology*
