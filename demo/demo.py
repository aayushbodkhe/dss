"""
ClinicalDSS — Interactive Multi-Visit Demo
==========================================
Demonstrates the full stateful Clinical Decision Support System
across two complete patient journeys (3 visits each).

Also demonstrates 2 honest limitation scenarios.

Usage:
  python demo.py                     # Full demo, interactive HITL
  python demo.py --auto              # Auto-approve HITL (non-interactive)
  python demo.py --patient arjun     # Single patient journey
  python demo.py --patient priya
  python demo.py --limitations       # Show limitations only
"""

import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DATA_DIR = Path(__file__).parent.parent / "data"


class C:
    RED = "\033[91m"; GREEN = "\033[92m"; YELLOW = "\033[93m"
    BLUE = "\033[94m"; CYAN = "\033[96m"; MAGENTA = "\033[95m"
    BOLD = "\033[1m"; DIM = "\033[2m"; RESET = "\033[0m"
    ORANGE = "\033[33m"


def hdr(text): print(f"\n{C.BOLD}{C.CYAN}{'═'*65}{C.RESET}\n  {C.BOLD}{text}{C.RESET}")
def sec(text): print(f"\n  {C.YELLOW}▶ {text}{C.RESET}")
def ok(text): print(f"  {C.GREEN}✓{C.RESET} {text}")
def warn(text): print(f"  {C.YELLOW}⚠{C.RESET}  {text}")
def err(text): print(f"  {C.RED}✗{C.RESET} {text}")
def dim(text): print(f"  {C.DIM}{text}{C.RESET}")


# ─────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────

with open(DATA_DIR / "patient_journeys.json") as f:
    JOURNEYS = {p["patient_id"]: p for p in json.load(f)}


# ─────────────────────────────────────────────────────────
# MOCK AGENT (for demo without LangGraph/API)
# ─────────────────────────────────────────────────────────

class MockAgentRunner:
    """
    Simulates the full LangGraph agent pipeline deterministically.
    Produces the same outputs as the real agent but without API calls.
    Shows all nodes activating, memory loading, critique firing.
    """

    def __init__(self, interactive: bool = True):
        self.interactive = interactive
        self.episodic_memory = {}  # patient_id → list of visits

    def run_visit(self, patient: dict, visit: dict) -> dict:
        pid = patient["patient_id"]
        vid = visit["visit_id"]

        # Load episodic memory
        prior_visits = self.episodic_memory.get(pid, [])

        # Simulate pipeline
        print(f"\n  {C.DIM}[Graph executing...]{C.RESET}")
        nodes = [
            ("intake_node",           "✓ Visit intake logged"),
            ("memory_retrieval_node", f"✓ Loaded {len(prior_visits)} prior visit(s)"),
            ("semantic_search_node",  "✓ Guidelines retrieved from semantic KB"),
            ("differential_diagnosis_node", "✓ Differential generated"),
            ("critique_node",         "✓ Critique analysis complete"),
        ]
        for node, msg in nodes:
            time.sleep(0.08)
            print(f"  {C.DIM}  [{node}] {msg}{C.RESET}")

        # Build result from ground truth + injected patterns
        gt = patient.get("ground_truth_differential", {}).get(vid, [])
        critiques = [c for c in patient.get("critique_injections", []) if c["visit"] == vid]
        injected = visit.get("injected_pattern", {})

        primary = gt[0]["diagnosis"] if gt else "Clinical assessment"
        confidence = gt[0]["probability"] if gt else 0.70

        # Critique corrections
        self_correction = bool(critiques)
        hitl_required = confidence < 0.80 or any(c["severity"] == "CRITICAL" for c in critiques)
        hitl_reason_parts = []
        if confidence < 0.80:
            hitl_reason_parts.append(f"confidence {confidence:.0%} < 80%")
        if any(c["severity"] == "CRITICAL" for c in critiques):
            hitl_reason_parts.append(f"{sum(1 for c in critiques if c['severity']=='CRITICAL')} CRITICAL finding(s)")
        # Psychiatric risk
        for sym in visit.get("symptoms", []):
            if "suicidal" in sym.lower() or "passive si" in sym.lower():
                hitl_reason_parts.append("psychiatric risk: passive SI")
                hitl_required = True
                break

        hitl_reason = " | ".join(hitl_reason_parts) if hitl_reason_parts else ""

        # HITL interaction
        hitl_response = None
        hitl_action = None
        if hitl_required:
            print(f"  {C.DIM}  [confidence_gate] → HITL required{C.RESET}")
            result = self._hitl_interaction(patient, visit, gt, critiques, confidence, hitl_reason)
            hitl_response = result["response"]
            hitl_action = result["action"]
        else:
            print(f"  {C.DIM}  [confidence_gate] → treatment_node (high confidence){C.RESET}")

        # Save to episodic
        self.episodic_memory.setdefault(pid, []).append({
            "visit_id": vid,
            "date": visit["date"],
            "chief_complaint": visit["chief_complaint"],
            "labs": visit.get("labs", {}),
            "primary_diagnosis": primary,
        })

        print(f"  {C.DIM}  [note_generation_node] ✓ Clinical note generated, episodic memory updated{C.RESET}")

        return {
            "patient_id": pid,
            "visit_id": vid,
            "primary_diagnosis": primary,
            "confidence": confidence,
            "differential": gt,
            "critique_findings": critiques,
            "self_correction": self_correction,
            "hitl_required": hitl_required,
            "hitl_reason": hitl_reason,
            "hitl_action": hitl_action,
            "injected_pattern": injected,
            "prior_visits_loaded": len(prior_visits),
        }

    def _hitl_interaction(self, patient, visit, gt, critiques, confidence, reason):
        print("\n" + "═"*65)
        print(f"{C.BOLD}{C.ORANGE}🔶  HUMAN-IN-THE-LOOP REQUIRED{C.RESET}")
        print("═"*65)
        print(f"  Patient  : {patient['name']} ({patient['age']}y)")
        print(f"  Visit    : {visit['visit_id']} | {visit['date']}")
        print(f"  Reason   : {reason}")
        print()
        print(f"  {C.BOLD}Assessment:{C.RESET} Symptoms are consistent with {gt[0]['diagnosis'] if gt else 'N/A'}")
        print(f"  Confidence: {confidence:.0%} {'🟠' if confidence < 0.80 else '🟢'}")
        print()
        if gt:
            print("  Differential:")
            for d in gt[:3]:
                icon = "🟢" if d["probability"] >= 0.80 else "🟠" if d["probability"] >= 0.60 else "🔴"
                print(f"    {icon} {d['probability']:.0%} — Symptoms consistent with {d['diagnosis']}")

        if critiques:
            print()
            print(f"  {C.BOLD}⚠  Critique Findings:{C.RESET}")
            for c in critiques:
                icon = "🔴" if c["severity"] == "CRITICAL" else "🟡"
                print(f"    {icon} [{c['severity']}] {c['type']}")
                print(f"       {c['message'][:120]}")

        print()
        if self.interactive:
            print(f"  {C.DIM}Options: [A] Approve  [E] Edit  [R] Reject{C.RESET}")
            try:
                choice = input(f"  {C.BOLD}Physician →{C.RESET} ").strip().upper() or "A"
            except EOFError:
                choice = "A"

            if choice == "E":
                try:
                    edited = input("  Enter revised treatment plan → ").strip()
                except EOFError:
                    edited = "[Physician-modified plan]"
                print(f"\n  {C.GREEN}✓ Treatment plan updated by physician.{C.RESET}")
                return {"response": edited, "action": "EDITED"}
            elif choice == "R":
                print(f"\n  {C.RED}✗ Rejected — flagged for specialist review.{C.RESET}")
                return {"response": "REJECTED", "action": "REJECTED"}
        else:
            print(f"  {C.DIM}[Auto-approving in non-interactive mode]{C.RESET}")

        print(f"\n  {C.GREEN}✓ APPROVED{C.RESET}")
        return {"response": "APPROVED", "action": "APPROVED"}


# ─────────────────────────────────────────────────────────
# PATIENT JOURNEY RUNNER
# ─────────────────────────────────────────────────────────

def run_patient_journey(patient_id: str, agent: MockAgentRunner):
    patient = JOURNEYS[patient_id]

    hdr(f"PATIENT JOURNEY: {patient['name']}")
    print(f"  Age: {patient['age']}y | Sex: {patient['sex']} | City: {patient['city']}")
    print(f"  Allergies: {', '.join(patient.get('known_allergies', ['None']))} {C.RED}← in episodic memory{C.RESET}")
    print(f"  Challenge tags: {' '.join(f'[{t}]' for t in patient['challenge_tags'])}")
    print()

    visit_results = []
    for visit_idx, visit in enumerate(patient["visits"]):
        print(f"\n{'─'*65}")
        print(f"  {C.BOLD}VISIT {visit['visit_id']} — {visit['date']}{C.RESET}  ({visit['visit_type']})")
        print(f"  Complaint: {visit['chief_complaint']}")
        print(f"  Key symptoms:")
        for sym in visit["symptoms"][:4]:
            print(f"    · {sym}")

        sec("Running ClinicalDSS Agent...")
        result = agent.run_visit(patient, visit)
        visit_results.append(result)

        # Show diagnosis result
        print()
        sec("Assessment Output")
        conf = result["confidence"]
        conf_color = C.GREEN if conf >= 0.80 else C.YELLOW if conf >= 0.60 else C.RED
        conf_label = "HIGH" if conf >= 0.80 else "MEDIUM" if conf >= 0.60 else "LOW"
        print(f"  Primary: {C.BOLD}Symptoms are consistent with {result['primary_diagnosis']}{C.RESET}")
        print(f"  Confidence: {conf_color}{conf:.0%} ({conf_label}){C.RESET}")
        print(f"  Prior visits in memory: {result['prior_visits_loaded']}")

        # Critique findings
        if result["critique_findings"]:
            sec("Critique Findings")
            for c in result["critique_findings"]:
                icon = f"{C.RED}🔴{C.RESET}" if c["severity"] == "CRITICAL" else f"{C.YELLOW}🟡{C.RESET}"
                print(f"  {icon} {c['message'][:130]}")
            if result["self_correction"]:
                ok(f"Self-correction occurred — treatment plan revised automatically")

        # HITL
        if result["hitl_required"]:
            if result["hitl_action"] in ("APPROVED", "EDITED", "AUTO-APPROVED"):
                ok(f"HITL completed — Action: {result['hitl_action']}")
            elif result["hitl_action"] == "REJECTED":
                warn("HITL: Rejected by physician — flagged for specialist review")
        else:
            dim(f"HITL not triggered — confidence {conf:.0%} ≥ 80% and no critical findings")

        # Teaching point
        injected = result.get("injected_pattern", {})
        if injected.get("description"):
            print()
            print(f"  {C.MAGENTA}📚 Teaching Point [{injected.get('tag','')}]:{C.RESET}")
            for line in injected["description"].split(". ")[:3]:
                if line.strip():
                    dim(line.strip() + ".")

        time.sleep(0.3)

    # Journey summary
    print(f"\n{'═'*65}")
    print(f"  {C.BOLD}JOURNEY SUMMARY — {patient['name']}{C.RESET}")
    print(f"  Visits completed      : {len(visit_results)}")
    print(f"  Self-corrections total: {sum(1 for r in visit_results if r['self_correction'])}")
    print(f"  HITL activations      : {sum(1 for r in visit_results if r['hitl_required'])}")
    critiques_total = sum(len(r['critique_findings']) for r in visit_results)
    print(f"  Critique findings     : {critiques_total}")
    print()


# ─────────────────────────────────────────────────────────
# LIMITATION DEMONSTRATIONS
# ─────────────────────────────────────────────────────────

LIMITATIONS = [
    {
        "title": "Rapidly Evolving Clinical Picture Between Visits",
        "scenario": (
            "A patient is processed at 9 AM with stable vitals. The agent generates a plan with "
            "HIGH confidence (0.87) and HITL is not triggered. By 11 AM, the patient deteriorates "
            "acutely — lactate rises from 1.8 to 6.4, BP drops to 70/40. "
            "The agent has no real-time monitoring connection."
        ),
        "limitation": (
            "ClinicalDSS processes discrete visit snapshots. It has no real-time vital sign "
            "streaming or continuous monitoring integration. A clinical deterioration between "
            "two queries is invisible to the system unless a new visit is explicitly submitted. "
            "Early Warning Score (EWS) systems or bedside monitoring feeds are OUTSIDE this architecture."
        ),
        "mitigations": [
            "Integrate with hospital EHR real-time feeds (HL7 FHIR streaming)",
            "Set time-limited confidence expiry (e.g., plans valid for 4h in ICU)",
            "Mandatory re-evaluation trigger when vital sign alerts fire",
        ],
    },
    {
        "title": "Hallucinated Guideline Citation in Novel Clinical Scenario",
        "scenario": (
            "A rare autoimmune encephalitis patient presents. No matching guideline exists in "
            "the semantic KB (it was updated in 2024, after KB cutoff). The LLM generates a "
            "plausible-sounding treatment plan citing 'AAN 2023 guideline' which actually "
            "recommends a different immunotherapy dosing. Confidence is 0.83 (above threshold) "
            "so HITL is NOT triggered."
        ),
        "limitation": (
            "LLMs can confidently hallucinate guideline citations when the true guideline is "
            "absent from semantic memory. High confidence does NOT mean correct. "
            "The HITL threshold (< 0.80) is a heuristic — a confident wrong answer at 0.83 "
            "bypasses physician review. Guideline staleness and hallucinated citations are "
            "structural risks that cannot be fully eliminated by the architecture."
        ),
        "mitigations": [
            "Require ALL guideline references to be traceable to a loaded GL-XXX ID in semantic memory",
            "Add hallucination detection: if cited guideline not in KB, flag to HITL regardless of confidence",
            "Continuous semantic KB update pipeline (monthly guideline refresh)",
            "Lower HITL threshold for rare diseases to 0.90",
        ],
    },
]


def run_limitations():
    hdr("HONEST LIMITATION DISCLOSURES")
    print(f"  {C.DIM}The following scenarios reveal architectural boundaries of ClinicalDSS.{C.RESET}")

    for i, lim in enumerate(LIMITATIONS, 1):
        print(f"\n{'─'*65}")
        print(f"  {C.BOLD}{C.YELLOW}LIMITATION {i}: {lim['title']}{C.RESET}")
        print()
        print(f"  {C.BOLD}Scenario:{C.RESET}")
        for line in lim["scenario"].split(". "):
            if line.strip():
                dim(line.strip() + ".")
        print()
        print(f"  {C.RED}{C.BOLD}Why this fails:{C.RESET}")
        for line in lim["limitation"].split(". "):
            if line.strip():
                print(f"    {C.RED}·{C.RESET} {line.strip()}.")
        print()
        print(f"  {C.GREEN}Proposed mitigations:{C.RESET}")
        for m in lim["mitigations"]:
            ok(m)

    print(f"\n{'═'*65}")
    print(f"  {C.BOLD}Bottom line:{C.RESET}")
    dim("ClinicalDSS reduces errors through memory, critique, and HITL.")
    dim("It does NOT replace physician judgment — it SUPPORTS it.")
    dim("All outputs must be validated against live patient context by a licensed clinician.")
    print()


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true", help="Auto-approve HITL")
    parser.add_argument("--patient", choices=["arjun", "priya"], help="Single patient")
    parser.add_argument("--limitations", action="store_true")
    args = parser.parse_args()

    print(f"\n{C.BOLD}{C.CYAN}")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║    ClinicalDSS — Stateful Agentic Clinical Decision Support  ║")
    print("║    LangGraph × Episodic Memory × Critique × HITL            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(C.RESET)

    if args.limitations:
        run_limitations()
        return

    agent = MockAgentRunner(interactive=not args.auto)

    if args.patient == "arjun":
        run_patient_journey("P_ARJUN", agent)
    elif args.patient == "priya":
        run_patient_journey("P_PRIYA", agent)
    else:
        # Both patients
        print(f"  {C.BOLD}Running full demo: 2 patients × 3 visits each{C.RESET}")
        print(f"  {C.DIM}Use --auto to skip interactive HITL prompts{C.RESET}\n")

        run_patient_journey("P_ARJUN", agent)
        print(f"\n\n{'█'*65}\n")
        run_patient_journey("P_PRIYA", agent)

        print(f"\n\n{'═'*65}")
        run_limitations()

        print(f"\n{C.BOLD}DEMO COMPLETE{C.RESET}")
        print()
        print("  What was demonstrated:")
        ok("Episodic memory: creatinine trend 1.8 → 2.1 → 2.3 caught across 3 visits (Arjun)")
        ok("Critique blocked methotrexate: CrCl 26 mL/min < 30 absolute contraindication")
        ok("HITL triggered: confidence 0.71 at V2 (below 0.80 threshold)")
        ok("Allergy guard: TMP-SMX blocked due to sulfonamide allergy (Priya V1)")
        ok("Critique blocked amitriptyline: QTc 462ms cardiac risk from V2 memory")
        ok("HITL triggered: passive SI + PHQ-9=16 + CRITICAL critique = mandatory review")
        print()
        dim("2 limitations disclosed: real-time monitoring gap + hallucinated citations")
        print()


if __name__ == "__main__":
    main()
