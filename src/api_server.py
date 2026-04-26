"""
ClinicalDSS API server for UI integration.

Run:
  uvicorn src.api_server:app --reload --port 8000
"""

from __future__ import annotations

import json
import os
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from src.agent import (
        ClinicalState,
        confidence_gate_node,
        critique_node,
        differential_diagnosis_node,
        episodic_store,
        intake_node,
        memory_retrieval_node,
        note_generation_node,
        semantic_search_node,
        treatment_node,
        trend_detection_node,
    )
except ModuleNotFoundError:
    # Support direct execution: `python src/api_server.py` or from `src/` directory.
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from agent import (  # type: ignore[no-redef]
        ClinicalState,
        confidence_gate_node,
        critique_node,
        differential_diagnosis_node,
        episodic_store,
        intake_node,
        memory_retrieval_node,
        note_generation_node,
        semantic_search_node,
        treatment_node,
        trend_detection_node,
    )


app = FastAPI(title="ClinicalDSS API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS: dict[str, dict[str, Any]] = {}
PENDING_REVIEWS: dict[str, ClinicalState] = {}


class SignInRequest(BaseModel):
    username: str
    password: str


class SignInResponse(BaseModel):
    token: str
    username: str


class VisitInput(BaseModel):
    patient_id: str
    patient_name: str
    patient_age: int
    patient_allergies: list[str] = Field(default_factory=list)

    visit_id: str
    visit_date: str
    chief_complaint: str
    symptoms: list[str] = Field(default_factory=list)
    vitals: dict[str, str] = Field(default_factory=dict)
    labs: dict[str, str] = Field(default_factory=dict)
    doctor_note: str = ""


class ReviewDecisionRequest(BaseModel):
    review_id: str
    action: Literal["APPROVE", "EDIT", "REJECT"]
    edited_treatment: str | None = None
    clinician_note: str | None = None


class NodeProgress(BaseModel):
    node: str
    status: str
    timestamp: str


class VisitRunResponse(BaseModel):
    status: Literal["completed", "hitl_required"]
    progress: list[NodeProgress]
    trend_summary: str
    trend_alerts: list[dict]
    critique_findings: list[dict]
    differential: list[dict]
    primary_diagnosis: str
    confidence_calibrated: float
    hitl_reason: str | None = None
    review_id: str | None = None
    proposed_treatment: str | None = None
    final_treatment: str | None = None
    care_plan: str | None = None
    explainability: dict[str, Any] = Field(default_factory=dict)


class ReviewDecisionResponse(BaseModel):
    status: Literal["completed", "flagged_for_specialist_review"]
    message: str
    final_treatment: str | None = None
    care_plan: str | None = None


class PastVisitsResponse(BaseModel):
    patient_id: str
    visits: list[dict]


class DifferentialCandidate(BaseModel):
    diagnosis: str
    probability: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


class DifferentialWhatIfRequest(BaseModel):
    patient_id: str = ""
    chief_complaint: str = ""
    symptoms: list[str] = Field(default_factory=list)
    current_primary: str = ""
    current_treatment: str = ""
    current_differential: list[DifferentialCandidate] = Field(default_factory=list)
    add_candidates: list[DifferentialCandidate] = Field(default_factory=list)
    remove_diagnoses: list[str] = Field(default_factory=list)


class DifferentialWhatIfResponse(BaseModel):
    updated_differential: list[dict]
    new_primary: str
    suggested_treatment: str
    confidence_estimate: float
    explainability_summary: str


class PatientTimelineResponse(BaseModel):
    patient_id: str
    events: list[dict]


class SimilarCasesResponse(BaseModel):
    anchor: dict[str, Any]
    similar_cases: list[dict]


def now_iso() -> str:
    return datetime.now().isoformat()


def progress_step(node: str, status: str = "completed") -> NodeProgress:
    return NodeProgress(node=node, status=status, timestamp=now_iso())


def _safe_float(raw_value: Any) -> float | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, (int, float)):
        return float(raw_value)
    match = re.search(r"[-+]?\d*\.?\d+", str(raw_value))
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def _normalize_text_tokens(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(tok) >= 3}


def _extract_medications(text: str) -> list[str]:
    aliases = {
        "furosemide": ["furosemide"],
        "bisoprolol": ["bisoprolol"],
        "ramipril": ["ramipril"],
        "dapagliflozin": ["dapagliflozin", "sglt2"],
        "methotrexate": ["methotrexate", "mtx"],
        "meropenem": ["meropenem"],
        "atenolol": ["atenolol"],
        "amitriptyline": ["amitriptyline"],
        "trimethoprim-sulfamethoxazole": ["trimethoprim-sulfamethoxazole", "bactrim", "co-trimoxazole"],
    }
    hay = (text or "").lower()
    found = []
    for canonical, keys in aliases.items():
        if any(k in hay for k in keys):
            found.append(canonical)
    return sorted(set(found))


def _build_explainability(state: ClinicalState) -> dict[str, Any]:
    confidence = float(state.get("confidence_calibrated", state.get("highest_confidence", 0.0)) or 0.0)
    trend_alerts = state.get("trend_alerts", [])
    critique = state.get("critique_findings", [])
    guideline_evidence = [
        {
            "id": g.get("id", ""),
            "title": g.get("title", ""),
            "category": g.get("category", ""),
            "excerpt": (g.get("body", "") or "")[:180],
        }
        for g in state.get("semantic_context", [])
    ]
    penalties = {
        "critical_trend_alerts": sum(1 for a in trend_alerts if a.get("severity") == "CRITICAL"),
        "high_trend_alerts": sum(1 for a in trend_alerts if a.get("severity") == "HIGH"),
        "critical_critique_findings": sum(1 for f in critique if f.get("severity") == "CRITICAL"),
    }

    return {
        "summary": state.get("diagnosis_reasoning", "") or "No reasoning generated.",
        "confidence": {
            "calibrated": round(confidence, 4),
            "label": "HIGH" if confidence >= 0.8 else "MEDIUM" if confidence >= 0.6 else "LOW",
            "penalties": penalties,
        },
        "guidelines_used": guideline_evidence,
        "trend_alerts": trend_alerts,
        "critique_findings": critique,
        "decision_path": [
            "episodic_memory",
            "trend_detection",
            "semantic_guidelines",
            "differential_generation",
            "critique_and_guardrails",
            "confidence_gate",
            "treatment_or_hitl",
        ],
    }


def _build_timeline_events(visits: list[dict]) -> list[dict]:
    ordered = sorted(visits, key=lambda v: (v.get("date", ""), v.get("visit_id", "")))
    events: list[dict] = []
    for visit in ordered:
        date = visit.get("date", "")
        visit_id = visit.get("visit_id", "")
        events.append(
            {
                "date": date,
                "visit_id": visit_id,
                "type": "visit",
                "title": f"Visit {visit_id}",
                "detail": visit.get("chief_complaint", ""),
            }
        )
        if visit.get("vitals"):
            events.append(
                {
                    "date": date,
                    "visit_id": visit_id,
                    "type": "vitals",
                    "title": "Vitals",
                    "detail": ", ".join(f"{k}: {v}" for k, v in visit.get("vitals", {}).items()),
                }
            )
        if visit.get("labs"):
            events.append(
                {
                    "date": date,
                    "visit_id": visit_id,
                    "type": "labs",
                    "title": "Labs",
                    "detail": ", ".join(f"{k}: {v}" for k, v in visit.get("labs", {}).items()),
                }
            )
        meds = _extract_medications(visit.get("final_treatment", ""))
        if meds:
            events.append(
                {
                    "date": date,
                    "visit_id": visit_id,
                    "type": "medications",
                    "title": "Treatment / Medications",
                    "detail": ", ".join(meds),
                }
            )
        if visit.get("trend_alerts"):
            events.append(
                {
                    "date": date,
                    "visit_id": visit_id,
                    "type": "alerts",
                    "title": "Safety Alerts",
                    "detail": " | ".join(a.get("message", "") for a in visit.get("trend_alerts", [])[:3]),
                }
            )
    return events


def _load_all_episodic_visits() -> list[dict]:
    store_dir = Path(episodic_store.store_path)
    cases: list[dict] = []
    if not store_dir.exists():
        return cases
    for file_path in store_dir.glob("*.json"):
        patient_id = file_path.stem
        try:
            with open(file_path, encoding="utf-8") as f:
                visits = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        for visit in visits:
            cases.append({"patient_id": patient_id, **visit})
    return cases


def _visit_similarity(anchor: dict[str, Any], candidate: dict[str, Any]) -> tuple[float, list[str]]:
    anchor_text = " ".join(
        [
            anchor.get("chief_complaint", ""),
            " ".join(anchor.get("symptoms", [])),
            anchor.get("primary_diagnosis", ""),
        ]
    )
    candidate_text = " ".join(
        [
            candidate.get("chief_complaint", ""),
            " ".join(candidate.get("symptoms", [])),
            candidate.get("primary_diagnosis", ""),
        ]
    )
    a_tokens = _normalize_text_tokens(anchor_text)
    c_tokens = _normalize_text_tokens(candidate_text)

    text_overlap = len(a_tokens & c_tokens) / max(1, len(a_tokens | c_tokens))

    a_labs = {k.lower() for k in (anchor.get("labs", {}) or {}).keys()}
    c_labs = {k.lower() for k in (candidate.get("labs", {}) or {}).keys()}
    lab_overlap = len(a_labs & c_labs) / max(1, len(a_labs | c_labs))

    a_diag = (anchor.get("primary_diagnosis", "") or "").strip().lower()
    c_diag = (candidate.get("primary_diagnosis", "") or "").strip().lower()
    diagnosis_match = 1.0 if a_diag and c_diag and a_diag == c_diag else 0.0

    score = (0.6 * text_overlap) + (0.25 * lab_overlap) + (0.15 * diagnosis_match)
    signals = sorted((a_tokens & c_tokens))[:6]
    return round(score, 4), signals


def _suggest_treatment_for_primary(primary: str, chief_complaint: str = "") -> str:
    text = f"{primary} {chief_complaint}".lower()
    if "heart failure" in text or "hfref" in text:
        return "Consider guideline-directed heart failure therapy with diuretic optimization, ACE/ARB/ARNI review, beta-blocker titration, and SGLT2 inhibitor if renal profile permits."
    if "urosepsis" in text or "sepsis" in text:
        return "Consider sepsis bundle: cultures, empiric antibiotics tailored to renal function, lactate-guided fluid strategy, and hemodynamic monitoring."
    if "rheumatoid" in text or "arthritis" in text:
        return "Consider DMARD-centered RA plan with renal-safe dosing, baseline CBC/LFT/creatinine, and rheumatology follow-up."
    if "depress" in text or "mood" in text:
        return "Consider depression management with suicide risk screening, ECG/QTc-aware medication choice, psychotherapy referral, and early follow-up."
    return "Consider symptom-guided treatment with guideline cross-check, contraindication review, and clinician confirmation."


def build_initial_state(visit: VisitInput) -> ClinicalState:
    return ClinicalState(
        patient_id=visit.patient_id,
        patient_name=visit.patient_name,
        patient_age=visit.patient_age,
        patient_allergies=visit.patient_allergies,
        visit_id=visit.visit_id,
        visit_date=visit.visit_date,
        chief_complaint=visit.chief_complaint,
        symptoms=visit.symptoms,
        vitals=visit.vitals,
        labs=visit.labs,
        doctor_note=visit.doctor_note,
        episodic_memory=[],
        semantic_context=[],
        semantic_citations=[],
        working_notes=[],
        trend_alerts=[],
        trend_summary="",
        safety_flags=[],
        differential_diagnosis=[],
        primary_diagnosis="",
        highest_confidence=0.0,
        confidence_calibrated=0.0,
        diagnosis_reasoning="",
        critique_findings=[],
        critique_passed=True,
        critique_corrected_diagnosis=None,
        self_correction_occurred=False,
        hitl_required=False,
        hitl_reason="",
        hitl_physician_response=None,
        hitl_feedback=None,
        hitl_approved=False,
        hitl_log=[],
        proposed_treatment="",
        final_treatment="",
        treatment_guardrail_applied=False,
        clinical_note="",
        outcome_followup=None,
        audit_log=[],
        error=None,
    )


def clinician_from_auth(authorization: str | None = Header(default=None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.replace("Bearer ", "", 1).strip()
    session = SESSIONS.get(token)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session token")
    return session["username"]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/auth/signin", response_model=SignInResponse)
def signin(payload: SignInRequest) -> SignInResponse:
    if not payload.username.strip() or not payload.password.strip():
        raise HTTPException(status_code=400, detail="Username and password are required")

    token = str(uuid.uuid4())
    SESSIONS[token] = {"username": payload.username.strip(), "created_at": now_iso()}
    return SignInResponse(token=token, username=payload.username.strip())


@app.post("/auth/signout")
def signout(authorization: str | None = Header(default=None)) -> dict[str, str]:
    if authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "", 1).strip()
        SESSIONS.pop(token, None)
    return {"status": "signed_out"}


@app.post("/visits/run", response_model=VisitRunResponse)
def run_visit(payload: VisitInput, clinician: str = Depends(clinician_from_auth)) -> VisitRunResponse:
    state = build_initial_state(payload)
    progress: list[NodeProgress] = []

    state = intake_node(state)
    progress.append(progress_step("intake_node"))

    state = memory_retrieval_node(state)
    progress.append(progress_step("memory_retrieval_node"))

    state = trend_detection_node(state)
    progress.append(progress_step("trend_detection_node"))

    state = semantic_search_node(state)
    progress.append(progress_step("semantic_search_node"))

    state = differential_diagnosis_node(state)
    progress.append(progress_step("differential_diagnosis_node"))

    state = critique_node(state)
    progress.append(progress_step("critique_node"))

    next_node = confidence_gate_node(state)
    progress.append(progress_step("confidence_gate_node", status=f"routed_to_{next_node}"))

    if next_node == "hitl_node":
        review_id = str(uuid.uuid4())
        PENDING_REVIEWS[review_id] = state
        explainability = _build_explainability(state)
        return VisitRunResponse(
            status="hitl_required",
            progress=progress,
            trend_summary=state.get("trend_summary", ""),
            trend_alerts=state.get("trend_alerts", []),
            critique_findings=state.get("critique_findings", []),
            differential=state.get("differential_diagnosis", []),
            primary_diagnosis=state.get("primary_diagnosis", ""),
            confidence_calibrated=state.get("confidence_calibrated", 0.0),
            hitl_reason=state.get("hitl_reason", ""),
            review_id=review_id,
            proposed_treatment=state.get("proposed_treatment", ""),
            explainability=explainability,
        )

    state = treatment_node(state)
    progress.append(progress_step("treatment_node"))

    state = note_generation_node(state)
    progress.append(progress_step("note_generation_node"))

    explainability = _build_explainability(state)
    return VisitRunResponse(
        status="completed",
        progress=progress,
        trend_summary=state.get("trend_summary", ""),
        trend_alerts=state.get("trend_alerts", []),
        critique_findings=state.get("critique_findings", []),
        differential=state.get("differential_diagnosis", []),
        primary_diagnosis=state.get("primary_diagnosis", ""),
        confidence_calibrated=state.get("confidence_calibrated", 0.0),
        final_treatment=state.get("final_treatment", ""),
        care_plan=state.get("clinical_note", ""),
        explainability=explainability,
    )


@app.post("/visits/review", response_model=ReviewDecisionResponse)
def review_decision(payload: ReviewDecisionRequest, clinician: str = Depends(clinician_from_auth)) -> ReviewDecisionResponse:
    state = PENDING_REVIEWS.get(payload.review_id)
    if not state:
        raise HTTPException(status_code=404, detail="Review ID not found or expired")

    action = payload.action.upper()
    if action == "REJECT":
        PENDING_REVIEWS.pop(payload.review_id, None)
        return ReviewDecisionResponse(
            status="flagged_for_specialist_review",
            message="Flagged for specialist review.",
        )

    if action == "EDIT":
        if not payload.edited_treatment:
            raise HTTPException(status_code=400, detail="edited_treatment is required for EDIT action")
        final_tx = payload.edited_treatment
        state["hitl_physician_response"] = final_tx
        state["hitl_feedback"] = payload.clinician_note or f"Edited by {clinician}"
        state["final_treatment"] = final_tx
        hitl_action = "EDITED"
    else:
        final_tx = state.get("proposed_treatment", "")
        state["hitl_physician_response"] = "APPROVED as proposed"
        state["hitl_feedback"] = payload.clinician_note or f"Approved by {clinician}"
        state["final_treatment"] = final_tx
        hitl_action = "APPROVED"

    state["hitl_approved"] = True
    state["hitl_log"].append(
        {
            "timestamp": now_iso(),
            "visit_id": state.get("visit_id"),
            "reason": state.get("hitl_reason"),
            "action": hitl_action,
            "original_treatment": state.get("proposed_treatment"),
            "final_treatment": final_tx,
            "confidence": state.get("confidence_calibrated", state.get("highest_confidence", 0.0)),
            "critique_findings": state.get("critique_findings", []),
            "clinician": clinician,
            "clinician_note": payload.clinician_note,
        }
    )

    state = treatment_node(state)
    state = note_generation_node(state)

    PENDING_REVIEWS.pop(payload.review_id, None)

    return ReviewDecisionResponse(
        status="completed",
        message="Treatment plan finalized after HITL approval.",
        final_treatment=state.get("final_treatment", ""),
        care_plan=state.get("clinical_note", ""),
    )


@app.get("/patients/{patient_id}/visits", response_model=PastVisitsResponse)
def past_visits(patient_id: str, clinician: str = Depends(clinician_from_auth)) -> PastVisitsResponse:
    visits = episodic_store.load_history(patient_id)
    return PastVisitsResponse(patient_id=patient_id, visits=visits)


@app.get("/patients/{patient_id}/timeline", response_model=PatientTimelineResponse)
def patient_timeline(patient_id: str, clinician: str = Depends(clinician_from_auth)) -> PatientTimelineResponse:
    visits = episodic_store.load_history(patient_id)
    return PatientTimelineResponse(patient_id=patient_id, events=_build_timeline_events(visits))


@app.get("/patients/{patient_id}/similar", response_model=SimilarCasesResponse)
def patient_similar_cases(
    patient_id: str,
    visit_id: str | None = None,
    top_k: int = 5,
    clinician: str = Depends(clinician_from_auth),
) -> SimilarCasesResponse:
    visits = episodic_store.load_history(patient_id)
    if not visits:
        raise HTTPException(status_code=404, detail="No visits found for patient")

    anchor = None
    if visit_id:
        for v in visits:
            if str(v.get("visit_id")) == str(visit_id):
                anchor = {"patient_id": patient_id, **v}
                break
        if anchor is None:
            raise HTTPException(status_code=404, detail="visit_id not found for patient")
    else:
        latest = sorted(visits, key=lambda v: (v.get("date", ""), v.get("visit_id", "")))[-1]
        anchor = {"patient_id": patient_id, **latest}

    all_cases = _load_all_episodic_visits()
    scored = []
    for case in all_cases:
        if case.get("patient_id") == anchor.get("patient_id") and case.get("visit_id") == anchor.get("visit_id"):
            continue
        score, signals = _visit_similarity(anchor, case)
        if score <= 0:
            continue
        scored.append(
            {
                "patient_id": case.get("patient_id", ""),
                "visit_id": case.get("visit_id", ""),
                "date": case.get("date", ""),
                "similarity_score": score,
                "chief_complaint": case.get("chief_complaint", ""),
                "primary_diagnosis": case.get("primary_diagnosis", ""),
                "outcome_summary": case.get("final_treatment", "")[:220],
                "matched_signals": signals,
            }
        )

    scored.sort(key=lambda c: c.get("similarity_score", 0), reverse=True)
    return SimilarCasesResponse(
        anchor={
            "patient_id": anchor.get("patient_id", ""),
            "visit_id": anchor.get("visit_id", ""),
            "date": anchor.get("date", ""),
            "chief_complaint": anchor.get("chief_complaint", ""),
            "primary_diagnosis": anchor.get("primary_diagnosis", ""),
        },
        similar_cases=scored[: max(1, min(top_k, 20))],
    )


@app.post("/visits/differential/what-if", response_model=DifferentialWhatIfResponse)
def differential_what_if(payload: DifferentialWhatIfRequest, clinician: str = Depends(clinician_from_auth)) -> DifferentialWhatIfResponse:
    remove_set = {d.strip().lower() for d in payload.remove_diagnoses if d.strip()}
    merged: dict[str, dict[str, Any]] = {}

    for cand in payload.current_differential:
        key = cand.diagnosis.strip()
        if not key or key.lower() in remove_set:
            continue
        merged[key.lower()] = {
            "diagnosis": key,
            "probability": float(cand.probability),
            "reasoning": cand.reasoning or "",
        }

    for cand in payload.add_candidates:
        key = cand.diagnosis.strip()
        if not key or key.lower() in remove_set:
            continue
        existing = merged.get(key.lower())
        if existing is None or cand.probability >= existing.get("probability", 0.0):
            merged[key.lower()] = {
                "diagnosis": key,
                "probability": float(cand.probability),
                "reasoning": cand.reasoning or "Added by clinician scenario builder.",
            }

    updated = sorted(merged.values(), key=lambda d: d.get("probability", 0.0), reverse=True)
    if not updated:
        raise HTTPException(status_code=400, detail="Differential cannot be empty after edits")

    primary = updated[0].get("diagnosis", "")
    confidence = float(updated[0].get("probability", 0.0))
    suggested = _suggest_treatment_for_primary(primary=primary, chief_complaint=payload.chief_complaint)

    summary = (
        f"Primary shifted to '{primary}' at {confidence:.0%} probability. "
        f"Treatment suggestion regenerated using complaint and differential context."
    )

    return DifferentialWhatIfResponse(
        updated_differential=updated,
        new_primary=primary,
        suggested_treatment=suggested,
        confidence_estimate=round(confidence, 4),
        explainability_summary=summary,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=False)
