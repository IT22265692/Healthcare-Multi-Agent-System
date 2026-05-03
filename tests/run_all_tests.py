"""
tests/run_all_tests.py — Unified Test Runner
=============================================
Runs all 4 student test files and prints a grouped summary showing each
student's results separately, followed by the overall pass/fail count.

Usage:
    python tests/run_all_tests.py          # run all
    python tests/run_all_tests.py --fast   # skip LLM-as-a-Judge tests
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent

STUDENT_FILES = [
    ("Student 1 — Symptom Agent", "tests/test_student1_symptom_agent.py"),
    ("Student 2 — Research Agent", "tests/test_student2_research_agent.py"),
    ("Student 3 — Triage Agent",   "tests/test_student3_triage_agent.py"),
    ("Student 4 — Report Agent",   "tests/test_student4_report_agent.py"),
]

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BLUE   = "\033[94m"
GREY   = "\033[90m"


def _colour(text: str, code: str) -> str:
    return f"{code}{text}{RESET}"


def _run_file(label: str, filepath: str) -> dict:
    """Run a single pytest file and capture results."""
    print(_colour(f"\n{'─'*60}", BLUE))
    print(_colour(f"  {label}", BOLD))
    print(_colour(f"  File: {filepath}", GREY))
    print(_colour(f"{'─'*60}", BLUE))

    start = time.time()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", filepath, "-v", "--tb=short", "--no-header"],
        cwd=ROOT,
        capture_output=False,
    )
    elapsed = time.time() - start

    return {
        "label": label,
        "file": filepath,
        "returncode": result.returncode,
        "elapsed": elapsed,
    }


def main() -> None:
    fast = "--fast" in sys.argv
    print(_colour("=" * 60, BLUE))
    print(_colour("  Healthcare MAS — Unified Test Runner", BOLD))
    print(_colour("  Running all 4 student test files", GREY))
    if fast:
        print(_colour("  Mode: --fast (LLM-as-a-Judge tests skipped)", YELLOW))
    print(_colour("=" * 60, BLUE))

    results = []
    for label, filepath in STUDENT_FILES:
        r = _run_file(label, filepath)
        results.append(r)

    # Summary table
    print()
    print(_colour("=" * 60, BLUE))
    print(_colour("  SUMMARY", BOLD))
    print(_colour("=" * 60, BLUE))

    total_pass = 0
    total_fail = 0
    for r in results:
        icon = _colour("✅  PASSED", GREEN) if r["returncode"] == 0 else _colour("❌  FAILED", RED)
        print(f"  {icon}  {r['label']}  ({r['elapsed']:.1f}s)")
        if r["returncode"] == 0:
            total_pass += 1
        else:
            total_fail += 1

    print(_colour("─" * 60, BLUE))
    print(f"  Files passed: {_colour(str(total_pass), GREEN)} / {len(results)}")
    if total_fail:
        print(f"  Files failed: {_colour(str(total_fail), RED)}")

    print(_colour("=" * 60, BLUE))
    print()
    print("  To run a single student's tests:")
    for label, filepath in STUDENT_FILES:
        print(_colour(f"    python -m pytest {filepath} -v", GREY))
    print()

    sys.exit(0 if total_fail == 0 else 1)


if __name__ == "__main__":
    main()