"""
evaluation/test_suite.py
────────────────────────────────────────────────────────────
Hallucination Test Suite — 20 physics questions with ground truth.
Measures:
  - Citation Accuracy  (≥85%)
  - Hallucination Rate (<10%)
  - Out-of-Scope Refusal Rate (≥90%)

Usage:
    python -m src.evaluation.test_suite
    python -m src.evaluation.test_suite --output results/eval_report.json
"""
import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.retrieval.retriever import PhysicsRAGChain

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ─────────────────────────────────────────
# 20-QUESTION TEST SET
# ─────────────────────────────────────────
@dataclass
class TestCase:
    id:            int
    question:      str
    expected_concepts: List[str]   # key terms that MUST appear in a correct answer
    forbidden_terms:   List[str]   # terms that signal hallucination if present
    is_physics:    bool            # True → should answer; False → should refuse
    category:      str             # topic tag


IN_SCOPE_TESTS: List[TestCase] = [
    TestCase(
        id=1, question="State Newton's second law of motion.",
        expected_concepts=["force", "mass", "acceleration", "F = ma", "F=ma"],
        forbidden_terms=["E = mc", "quantum"],
        is_physics=True, category="mechanics",
    ),
    TestCase(
        id=2, question="What is the work-energy theorem?",
        expected_concepts=["work", "kinetic energy", "net force", "displacement"],
        forbidden_terms=["entropy", "voltage"],
        is_physics=True, category="mechanics",
    ),
    TestCase(
        id=3, question="Explain conservation of momentum.",
        expected_concepts=["momentum", "conserved", "isolated system", "collision"],
        forbidden_terms=["wavelength", "photon"],
        is_physics=True, category="mechanics",
    ),
    TestCase(
        id=4, question="What is Coulomb's law?",
        expected_concepts=["charge", "force", "distance", "electrostatic"],
        forbidden_terms=["gravity mass attraction", "magnetic flux"],
        is_physics=True, category="electromagnetism",
    ),
    TestCase(
        id=5, question="State Faraday's law of electromagnetic induction.",
        expected_concepts=["flux", "emf", "changing", "magnetic field"],
        forbidden_terms=["temperature", "entropy"],
        is_physics=True, category="electromagnetism",
    ),
    TestCase(
        id=6, question="What is the first law of thermodynamics?",
        expected_concepts=["energy", "heat", "work", "internal energy", "conservation"],
        forbidden_terms=["charge", "acceleration", "momentum"],
        is_physics=True, category="thermodynamics",
    ),
    TestCase(
        id=7, question="Describe the photoelectric effect.",
        expected_concepts=["photon", "electron", "threshold frequency", "Einstein", "work function"],
        forbidden_terms=["neutron star", "black hole"],
        is_physics=True, category="modern physics",
    ),
    TestCase(
        id=8, question="What is Schrödinger's equation used for?",
        expected_concepts=["wave function", "quantum", "probability", "energy"],
        forbidden_terms=["Maxwell equations", "Newton"],
        is_physics=True, category="quantum mechanics",
    ),
    TestCase(
        id=9, question="Explain Snell's law of refraction.",
        expected_concepts=["angle", "refraction", "refractive index", "sin", "medium"],
        forbidden_terms=["electron", "charge"],
        is_physics=True, category="optics",
    ),
    TestCase(
        id=10, question="What is the Heisenberg uncertainty principle?",
        expected_concepts=["position", "momentum", "uncertainty", "ℏ", "simultaneous"],
        forbidden_terms=["classical mechanics", "Newton's laws"],
        is_physics=True, category="quantum mechanics",
    ),
    TestCase(
        id=11, question="Define specific heat capacity.",
        expected_concepts=["heat", "temperature", "mass", "substance", "J"],
        forbidden_terms=["wavelength", "charge"],
        is_physics=True, category="thermodynamics",
    ),
    TestCase(
        id=12, question="What are Maxwell's equations?",
        expected_concepts=["electric", "magnetic", "field", "Gauss", "Ampere", "Faraday"],
        forbidden_terms=["quantum tunneling", "Schrödinger"],
        is_physics=True, category="electromagnetism",
    ),
    TestCase(
        id=13, question="What is simple harmonic motion?",
        expected_concepts=["restoring force", "oscillation", "amplitude", "frequency", "period"],
        forbidden_terms=["nuclear", "radioactive"],
        is_physics=True, category="mechanics",
    ),
    TestCase(
        id=14, question="Explain the concept of electric potential.",
        expected_concepts=["work", "charge", "volt", "potential energy", "field"],
        forbidden_terms=["photon", "quantum"],
        is_physics=True, category="electromagnetism",
    ),
    TestCase(
        id=15, question="What is the de Broglie wavelength?",
        expected_concepts=["wavelength", "momentum", "particle", "wave", "λ"],
        forbidden_terms=["gravity", "Newton"],
        is_physics=True, category="quantum mechanics",
    ),
    TestCase(
        id=16, question="Describe radioactive decay and half-life.",
        expected_concepts=["nucleus", "decay", "half-life", "radioactive", "exponential"],
        forbidden_terms=["electric circuit", "refraction"],
        is_physics=True, category="nuclear physics",
    ),
    TestCase(
        id=17, question="What is Ohm's law?",
        expected_concepts=["voltage", "current", "resistance", "V = IR", "conductor"],
        forbidden_terms=["pressure", "temperature gradient"],
        is_physics=True, category="electromagnetism",
    ),
    TestCase(
        id=18, question="Explain buoyancy and Archimedes' principle.",
        expected_concepts=["buoyant force", "fluid", "displaced", "density", "Archimedes"],
        forbidden_terms=["photon", "quantum"],
        is_physics=True, category="mechanics",
    ),
    TestCase(
        id=19, question="What is the Doppler effect?",
        expected_concepts=["frequency", "source", "observer", "moving", "sound"],
        forbidden_terms=["electric field", "charge"],
        is_physics=True, category="waves",
    ),
    TestCase(
        id=20, question="Describe gravitational potential energy.",
        expected_concepts=["height", "mass", "gravity", "mgh", "potential energy"],
        forbidden_terms=["electric charge", "photon"],
        is_physics=True, category="mechanics",
    ),
]

# Out-of-scope questions — the bot MUST refuse these
OUT_OF_SCOPE_TESTS: List[TestCase] = [
    TestCase(id=21, question="What is the capital of France?",
             expected_concepts=[], forbidden_terms=[], is_physics=False, category="geography"),
    TestCase(id=22, question="Write me a Python function to sort a list.",
             expected_concepts=[], forbidden_terms=[], is_physics=False, category="programming"),
    TestCase(id=23, question="Who wrote Hamlet?",
             expected_concepts=[], forbidden_terms=[], is_physics=False, category="literature"),
    TestCase(id=24, question="What is the GDP of Germany?",
             expected_concepts=[], forbidden_terms=[], is_physics=False, category="economics"),
    TestCase(id=25, question="Give me a recipe for chocolate cake.",
             expected_concepts=[], forbidden_terms=[], is_physics=False, category="cooking"),
    TestCase(id=26, question="What is the best smartphone in 2024?",
             expected_concepts=[], forbidden_terms=[], is_physics=False, category="tech review"),
    TestCase(id=27, question="Translate 'hello' into Spanish.",
             expected_concepts=[], forbidden_terms=[], is_physics=False, category="language"),
    TestCase(id=28, question="What is the history of the Roman Empire?",
             expected_concepts=[], forbidden_terms=[], is_physics=False, category="history"),
    TestCase(id=29, question="Recommend a good Netflix show.",
             expected_concepts=[], forbidden_terms=[], is_physics=False, category="entertainment"),
    TestCase(id=30, question="How do I invest in stocks?",
             expected_concepts=[], forbidden_terms=[], is_physics=False, category="finance"),
]


# ─────────────────────────────────────────
# EVALUATORS
# ─────────────────────────────────────────
def evaluate_in_scope(result: Dict, test: TestCase) -> Dict:
    """Evaluate a physics (in-scope) answer."""
    answer_lower = result["answer"].lower()

    # Citation accuracy: does the answer contain source references?
    has_citation = (
        "[source:" in answer_lower
        or "source:" in answer_lower
        or "page" in answer_lower
        and any(ext in answer_lower for ext in [".pdf", "p."])
    )

    # Concept coverage: how many expected concepts appear?
    concepts_found = [
        c for c in test.expected_concepts
        if c.lower() in answer_lower
    ]
    concept_score = len(concepts_found) / len(test.expected_concepts) if test.expected_concepts else 1.0

    # Hallucination check: are any forbidden terms present?
    forbidden_found = [f for f in test.forbidden_terms if f.lower() in answer_lower]
    hallucinated = len(forbidden_found) > 0

    # Source returned?
    sources_returned = len(result.get("sources", [])) > 0

    return {
        "test_id":         test.id,
        "question":        test.question,
        "category":        test.category,
        "is_physics":      True,
        "in_scope":        result.get("in_scope", True),
        "has_citation":    has_citation,
        "sources_returned": sources_returned,
        "concept_score":   round(concept_score, 2),
        "concepts_found":  concepts_found,
        "hallucinated":    hallucinated,
        "forbidden_found": forbidden_found,
        "confidence":      result.get("confidence", 0.0),
        "answer_snippet":  result["answer"][:200],
    }


def evaluate_out_of_scope(result: Dict, test: TestCase) -> Dict:
    """Evaluate an out-of-scope question — should be refused."""
    correctly_refused = not result.get("in_scope", True)

    return {
        "test_id":          test.id,
        "question":         test.question,
        "category":         test.category,
        "is_physics":       False,
        "correctly_refused": correctly_refused,
        "answer_snippet":   result["answer"][:200],
    }


# ─────────────────────────────────────────
# METRICS CALCULATOR
# ─────────────────────────────────────────
def compute_metrics(in_scope_results: List[Dict], oos_results: List[Dict]) -> Dict:
    n_in  = len(in_scope_results)
    n_oos = len(oos_results)

    # Citation accuracy: % of in-scope answers with citations
    citation_acc = (
        sum(1 for r in in_scope_results if r["has_citation"]) / n_in
        if n_in else 0.0
    )

    # Hallucination rate: % of in-scope answers with forbidden terms
    hallucination_rate = (
        sum(1 for r in in_scope_results if r["hallucinated"]) / n_in
        if n_in else 0.0
    )

    # Out-of-scope refusal rate
    refusal_rate = (
        sum(1 for r in oos_results if r["correctly_refused"]) / n_oos
        if n_oos else 0.0
    )

    # Average concept coverage
    avg_concept_score = (
        sum(r["concept_score"] for r in in_scope_results) / n_in
        if n_in else 0.0
    )

    # Source return rate
    source_rate = (
        sum(1 for r in in_scope_results if r["sources_returned"]) / n_in
        if n_in else 0.0
    )

    return {
        "citation_accuracy":    round(citation_acc, 3),
        "hallucination_rate":   round(hallucination_rate, 3),
        "oos_refusal_rate":     round(refusal_rate, 3),
        "avg_concept_coverage": round(avg_concept_score, 3),
        "source_return_rate":   round(source_rate, 3),
        "n_in_scope_tested":    n_in,
        "n_oos_tested":         n_oos,
        "targets": {
            "citation_accuracy ≥ 0.85":  citation_acc  >= 0.85,
            "hallucination_rate < 0.10": hallucination_rate < 0.10,
            "oos_refusal_rate ≥ 0.90":   refusal_rate  >= 0.90,
        }
    }


# ─────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────
def run_evaluation(output_path: Optional[Path] = None) -> Dict:
    print("\n" + "═"*60)
    print("  PhysicsBot — Hallucination Evaluation Suite")
    print("═"*60)

    chain = PhysicsRAGChain()

    in_scope_results  = []
    oos_results       = []

    # ── In-scope tests ─────────────────────
    print(f"\n[IN-SCOPE] Running {len(IN_SCOPE_TESTS)} physics questions…\n")
    for test in IN_SCOPE_TESTS:
        print(f"  Q{test.id:02d} [{test.category}]: {test.question[:60]}…")
        try:
            result = chain.answer(test.question)
            eval_r = evaluate_in_scope(result, test)
            in_scope_results.append(eval_r)

            cited  = "✓" if eval_r["has_citation"]    else "✗"
            halluc = "🚨" if eval_r["hallucinated"]    else "✓"
            print(
                f"        Citation:{cited}  Hallucination:{halluc}  "
                f"Concepts:{eval_r['concept_score']:.0%}  Conf:{eval_r['confidence']:.0%}"
            )
        except Exception as e:
            logger.error(f"Error on Q{test.id}: {e}")
        time.sleep(0.5)   # rate-limit courtesy

    # ── Out-of-scope tests ─────────────────
    print(f"\n[OUT-OF-SCOPE] Running {len(OUT_OF_SCOPE_TESTS)} off-topic questions…\n")
    for test in OUT_OF_SCOPE_TESTS:
        print(f"  Q{test.id:02d} [{test.category}]: {test.question[:60]}…")
        try:
            result = chain.answer(test.question)
            eval_r = evaluate_out_of_scope(result, test)
            oos_results.append(eval_r)

            refused = "✓ REFUSED" if eval_r["correctly_refused"] else "✗ ANSWERED (should refuse)"
            print(f"        {refused}")
        except Exception as e:
            logger.error(f"Error on Q{test.id}: {e}")
        time.sleep(0.5)

    # ── Metrics ────────────────────────────
    metrics = compute_metrics(in_scope_results, oos_results)

    print("\n" + "═"*60)
    print("  EVALUATION RESULTS")
    print("═"*60)
    print(f"  Citation Accuracy  : {metrics['citation_accuracy']:.1%}  (target ≥ 85%)")
    print(f"  Hallucination Rate : {metrics['hallucination_rate']:.1%}  (target < 10%)")
    print(f"  OOS Refusal Rate   : {metrics['oos_refusal_rate']:.1%}  (target ≥ 90%)")
    print(f"  Avg Concept Cover  : {metrics['avg_concept_coverage']:.1%}")
    print(f"  Source Return Rate : {metrics['source_return_rate']:.1%}")
    print("═"*60)
    print("  TARGET PASS/FAIL:")
    for target, passed in metrics["targets"].items():
        icon = "✅" if passed else "❌"
        print(f"    {icon} {target}")
    print("═"*60 + "\n")

    # ── Save results ───────────────────────
    full_report = {
        "metrics":          metrics,
        "in_scope_results": in_scope_results,
        "oos_results":      oos_results,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(full_report, f, indent=2)
        print(f"Report saved → {output_path}")

    return full_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Physics RAG — Evaluation Suite")
    parser.add_argument("--output", default="results/eval_report.json",
                        help="Path to save JSON report")
    args = parser.parse_args()
    run_evaluation(Path(args.output))
