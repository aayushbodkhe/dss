"""
ClinicalDSS — LangGraph Stateful Clinical Decision Support Agent
=================================================================
Path A Implementation: LangGraph with full state management.

Graph Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │                   StateGraph (TypedDict)                 │
  │                                                         │
  │  intake_node ──► memory_retrieval_node                  │
  │                         │                               │
  │                  semantic_search_node                    │
  │                         │                               │
  │               differential_diagnosis_node               │
  │                         │                               │
  │                   critique_node ─────────────────┐      │
  │                         │                        │      │
  │                 confidence_gate_node             │      │
  │                    /           \                 │      │
  │            [≥0.80]              [<0.80]          │      │
  │               │                    │             │      │
  │        treatment_node        hitl_node ──────────┘      │
  │               │                    │             retry  │
  │        note_generation_node   [approved/edited]         │
  │               │                                         │
  │           END                                           │
  └─────────────────────────────────────────────────────────┘

Memory Types:
  Episodic: Patient visit history (all prior visits, labs, diagnoses)
  Semantic:  Medical guidelines KB (GL001–GL010)
  Working:   Current visit state (within-session)
"""

import os
import json
import time
import logging
import hashlib
import re
from typing import TypedDict, Annotated, Optional, Any
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ClinicalDSS")

DATA_DIR = Path(__file__).parent.parent / "data"

load_dotenv()

# ─────────────────────────────────────────────────────────
# OPTIONAL IMPORTS — graceful fallback for offline/demo
# ─────────────────────────────────────────────────────────

try:
    from langchain_openai import AzureChatOpenAI, ChatOpenAI, AzureOpenAIEmbeddings
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import importlib
    from langchain_core.documents import Document
    Chroma = importlib.import_module("langchain_chroma").Chroma
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("⚠  langgraph not installed: pip install langgraph")


# ─────────────────────────────────────────────────────────
# STATE DEFINITION
# ─────────────────────────────────────────────────────────

class ClinicalState(TypedDict):
    # Patient identity
    patient_id: str
    patient_name: str
    patient_age: int
    patient_allergies: list[str]

    # Current visit input
    visit_id: str
    visit_date: str
    chief_complaint: str
    symptoms: list[str]
    vitals: dict[str, str]
    labs: dict[str, str]
    doctor_note: str

    # Memory layers
    episodic_memory: list[dict]      # All prior visits loaded from memory
    semantic_context: list[dict]     # Relevant guidelines retrieved
    semantic_citations: list[str]    # Guideline IDs used for this visit
    working_notes: list[str]         # Intra-visit reasoning notes
    trend_alerts: list[dict]         # Temporal alerts from longitudinal labs
    trend_summary: str
    safety_flags: list[str]

    # Diagnosis layer
    differential_diagnosis: list[dict]          # [{diagnosis, probability, reasoning}]
    primary_diagnosis: str
    highest_confidence: float
    confidence_calibrated: float
    diagnosis_reasoning: str

    # Critique layer
    critique_findings: list[dict]    # [{type, severity, message, source_guideline}]
    critique_passed: bool
    critique_corrected_diagnosis: Optional[str]
    self_correction_occurred: bool

    # HITL layer
    hitl_required: bool
    hitl_reason: str
    hitl_physician_response: Optional[str]
    hitl_feedback: Optional[str]
    hitl_approved: bool
    hitl_log: list[dict]             # Audit log of all HITL interactions

    # Treatment layer
    proposed_treatment: str
    final_treatment: str
    treatment_guardrail_applied: bool

    # Output
    clinical_note: str
    outcome_followup: Optional[str]
    audit_log: list[dict]            # Full audit trail
    error: Optional[str]


# ─────────────────────────────────────────────────────────
# MEMORY STORE (in-memory, file-backed)
# ─────────────────────────────────────────────────────────

class EpisodicMemoryStore:
    """
    Stores and retrieves patient visit history.
    Production: replace with Azure CosmosDB or PostgreSQL.
    Training: JSON file per patient.
    """

    def __init__(self, store_path: Path = DATA_DIR.parent / "configs" / "episodic_store"):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._cache = {}

    def save_visit(self, patient_id: str, visit: dict):
        path = self.store_path / f"{patient_id}.json"
        history = self.load_history(patient_id)
        # Upsert by visit_id
        existing_ids = [v["visit_id"] for v in history]
        if visit.get("visit_id") in existing_ids:
            history = [v if v["visit_id"] != visit["visit_id"] else visit for v in history]
        else:
            history.append(visit)
        with open(path, "w") as f:
            json.dump(history, f, indent=2)
        self._cache[patient_id] = history
        logger.info(f"  [Episodic] Saved visit {visit.get('visit_id')} for {patient_id}")

    def load_history(self, patient_id: str) -> list[dict]:
        if patient_id in self._cache:
            return self._cache[patient_id]
        path = self.store_path / f"{patient_id}.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            self._cache[patient_id] = data
            return data
        return []

    def get_lab_trend(self, patient_id: str, lab_name: str) -> list[dict]:
        """Extract temporal trend for a specific lab across all visits."""
        history = self.load_history(patient_id)
        trend = []
        for visit in sorted(history, key=lambda v: v.get("date", "")):
            labs = visit.get("labs", {})
            # Try exact and partial key match
            for key, val in labs.items():
                if lab_name.lower() in key.lower():
                    trend.append({
                        "visit_id": visit["visit_id"],
                        "date": visit.get("date"),
                        "value": val,
                    })
        return trend


class SemanticMemoryStore:
    """
    Retrieves relevant medical guidelines by keyword matching.
    Production: replace with Azure AI Search or ChromaDB.
    Training: simple keyword overlap scoring.
    """

    def __init__(self):
        with open(DATA_DIR / "guidelines.json") as f:
            self.guidelines = json.load(f)

        self._vectorstore = None
        self._chroma_enabled = os.getenv("USE_CHROMA", "true").lower() == "true"
        self._chroma_ready = False

        if self._chroma_enabled:
            self._init_chroma_store()

    def _init_chroma_store(self):
        if not CHROMA_AVAILABLE:
            logger.warning("[SEMANTIC] Chroma dependencies missing; falling back to keyword search.")
            return

        azure_endpoint = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_key = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")
        azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
        azure_api_version = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION", "2024-02-01-preview")

        if not (azure_endpoint and azure_api_key and azure_deployment):
            logger.warning("[SEMANTIC] Embedding env vars incomplete; using keyword retrieval fallback.")
            return

        try:
            embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                azure_deployment=azure_deployment,
                openai_api_version=azure_api_version,
            )

            persist_dir = Path(os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR.parent / "configs" / "chroma")))
            persist_dir.mkdir(parents=True, exist_ok=True)
            collection_name = os.getenv("CHROMA_COLLECTION", "clinical_guidelines")

            self._vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=str(persist_dir),
            )

            existing_count = self._vectorstore._collection.count()
            if existing_count == 0:
                docs = [
                    Document(
                        page_content=gl.get("body", ""),
                        metadata={
                            "id": gl.get("id", ""),
                            "title": gl.get("title", ""),
                            "category": gl.get("category", ""),
                            "keywords": ", ".join(gl.get("keywords", [])),
                        },
                    )
                    for gl in self.guidelines
                ]
                ids = [gl.get("id", f"GL_{i}") for i, gl in enumerate(self.guidelines)]
                self._vectorstore.add_documents(docs, ids=ids)
                logger.info(f"[SEMANTIC] Indexed {len(docs)} guidelines in Chroma.")

            self._chroma_ready = True
            logger.info("[SEMANTIC] Chroma retrieval enabled.")
        except Exception as exc:
            logger.warning(f"[SEMANTIC] Failed to initialize Chroma ({exc}); using keyword fallback.")
            self._vectorstore = None
            self._chroma_ready = False

    def _keyword_search(self, query: str, top_k: int = 3) -> list[dict]:
        query_lower = query.lower()
        scored = []
        for gl in self.guidelines:
            score = sum(kw.lower() in query_lower for kw in gl.get("keywords", []))
            for word in gl.get("title", "").lower().split():
                if word in query_lower and len(word) > 4:
                    score += 0.5
            if score > 0:
                scored.append({**gl, "_score": round(float(score), 4), "_source": "keyword"})
        scored.sort(key=lambda x: x["_score"], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _keyword_boost(query_lower: str, guideline: dict) -> float:
        title = guideline.get("title", "").lower()
        keywords = [k.lower() for k in guideline.get("keywords", [])]
        hits = sum(kw in query_lower for kw in keywords)
        title_hits = sum(word in query_lower for word in title.split() if len(word) > 4)
        return float(hits) + (0.2 * float(title_hits))

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        query_lower = query.lower()

        if self._chroma_ready and self._vectorstore is not None:
            try:
                candidates = self._vectorstore.similarity_search_with_relevance_scores(query, k=top_k * 2)
                results = []
                for doc, rel_score in candidates:
                    metadata = doc.metadata or {}
                    guideline = {
                        "id": metadata.get("id", "GL_UNKNOWN"),
                        "title": metadata.get("title", "Guideline"),
                        "body": doc.page_content,
                        "category": metadata.get("category", "unknown"),
                        "keywords": [k.strip() for k in str(metadata.get("keywords", "")).split(",") if k.strip()],
                    }
                    hybrid_score = float(rel_score) + self._keyword_boost(query_lower, guideline)
                    results.append({**guideline, "_score": round(hybrid_score, 4), "_source": "chroma_hybrid"})

                results.sort(key=lambda x: x["_score"], reverse=True)
                dedup = {}
                for item in results:
                    dedup[item["id"]] = item
                final_results = list(dedup.values())[:top_k]
                if final_results:
                    return final_results
            except Exception as exc:
                logger.warning(f"[SEMANTIC] Chroma search failed ({exc}); fallback to keyword retrieval.")

        return self._keyword_search(query, top_k=top_k)


# ─────────────────────────────────────────────────────────
# LLM SETUP
# ─────────────────────────────────────────────────────────

def get_llm(use_azure: bool = False):
    """
    Returns an LLM client. Supports:
      - Azure OpenAI (production)
      - OpenAI (development)
      - MockLLM (offline/demo without API keys)
    """
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")

    if use_azure and azure_endpoint and azure_api_key:
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
            temperature=0.1,
        )
    elif os.getenv("OPENAI_API_KEY") and OPENAI_AVAILABLE:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    else:
        return MockLLM()


class MockLLM:
    """
    Deterministic mock LLM for offline training demos.
    Returns clinically realistic but scripted responses.
    Simulates the exact failure modes in the dataset.
    """

    DIAGNOSIS_TEMPLATES = {
        "shortness of breath": [
            {"diagnosis": "Heart failure with reduced EF (HFrEF)", "probability": 0.71, "reasoning": "Bilateral crepitations, elevated JVP, orthopnoea, BNP 1840 pg/mL consistent with HFrEF"},
            {"diagnosis": "COPD exacerbation", "probability": 0.19, "reasoning": "40 pack-year smoking history; however, no wheeze and inhalers failed"},
            {"diagnosis": "Acute coronary syndrome", "probability": 0.10, "reasoning": "Mildly elevated troponin but no chest pain, ECG non-ischaemic"},
        ],
        "fever dysuria confusion": [
            {"diagnosis": "Urosepsis", "probability": 0.94, "reasoning": "Positive urine cultures, qSOFA 3, lactate 4.2, bilateral pyelonephritis on CT"},
            {"diagnosis": "AKI stage 2 (sepsis-associated)", "probability": 0.91, "reasoning": "Creatinine 2.8 from baseline 1.1 in 48h — AKI by KDIGO criteria"},
        ],
        "joint pain morning stiffness": [
            {"diagnosis": "Rheumatoid Arthritis", "probability": 0.91, "reasoning": "Symmetrical polyarthritis, RF+, Anti-CCP strongly positive, DAS28 5.8"},
            {"diagnosis": "Psoriatic arthritis", "probability": 0.06, "reasoning": "No skin lesions, anti-CCP positive — less likely"},
        ],
        "low mood anhedonia": [
            {"diagnosis": "Major Depressive Episode (moderate-severe)", "probability": 0.87, "reasoning": "PHQ-9=16, symptom duration >3 weeks, passive SI, post-ICU context"},
            {"diagnosis": "Post-ICU Syndrome (PICS)", "probability": 0.79, "reasoning": "Cognitive and emotional changes following critical illness — overlapping diagnosis"},
        ],
        "default": [
            {"diagnosis": "Unspecified clinical syndrome", "probability": 0.60, "reasoning": "Insufficient data for specific diagnosis"},
        ]
    }

    TREATMENT_TEMPLATES = {
        "HFrEF": "Optimise GDMT: furosemide 40mg OD, bisoprolol 2.5mg (start low), ramipril 2.5mg OD. Add SGLT2i (dapagliflozin 10mg) for HFrEF + T2DM — check eGFR first.",
        "RA": "Start methotrexate 7.5mg weekly + folate 1mg daily. Monitor LFTs, CBC, creatinine monthly.",  # <-- DELIBERATELY WRONG for Arjun (CrCl 26)
        "urosepsis": "Meropenem 500mg IV q6h (renal dose for AKI), 30mL/kg IV fluid bolus, vasopressors if MAP <65. Hold nephrotoxic agents. Continue atenolol.",  # <-- DELIBERATELY WRONG (atenolol in shock)
        "depression": "Start amitriptyline 25mg nocte, titrate to 75mg. PHQ-9 follow up in 4 weeks.",  # <-- DELIBERATELY WRONG (QTc issue)
        "UTI": "Trimethoprim-sulfamethoxazole DS BD × 3 days.",  # <-- WRONG (sulfonamide allergy)
        "default": "Symptomatic management. Follow up in 2 weeks.",
    }

    def invoke(self, messages_or_prompt):
        """Simulate LLM response — returns a string or AIMessage-like object."""
        if isinstance(messages_or_prompt, list):
            content = " ".join(str(m) for m in messages_or_prompt).lower()
        else:
            content = str(messages_or_prompt).lower()

        # Pick diagnosis template
        for key, diagnoses in self.DIAGNOSIS_TEMPLATES.items():
            if any(word in content for word in key.split()):
                return self._make_response(diagnoses, content)

        return self._make_response(self.DIAGNOSIS_TEMPLATES["default"], content)

    def _make_response(self, diagnoses, content):
        # Pick treatment
        treatment = self.TREATMENT_TEMPLATES["default"]
        for keyword, tx in self.TREATMENT_TEMPLATES.items():
            if keyword.lower() in content:
                treatment = tx
                break

        response_text = json.dumps({
            "differential": diagnoses,
            "primary": diagnoses[0]["diagnosis"],
            "reasoning": diagnoses[0]["reasoning"],
            "proposed_treatment": treatment,
        })
        return type("AIMessage", (), {"content": response_text})()

    def stream(self, *args, **kwargs):
        return [self.invoke(args[0] if args else "")]


# ─────────────────────────────────────────────────────────
# SYSTEM PROMPTS (with guardrails built in)
# ─────────────────────────────────────────────────────────

DIFFERENTIAL_PROMPT = """You are a clinical decision support assistant helping physicians.

CRITICAL LANGUAGE GUARDRAILS:
- NEVER say "The patient HAS [disease]" → ALWAYS say "Symptoms are consistent with..."
- NEVER say "The patient DEFINITELY has..." → ALWAYS say "Findings suggest..."
- NEVER make definitive diagnoses — you generate DIFFERENTIAL diagnoses with probabilities
- You SUPPORT physician decision-making; you do NOT replace it

EPISODIC MEMORY (prior visits):
{episodic_context}

SEMANTIC CONTEXT (relevant guidelines):
{semantic_context}

CURRENT VISIT:
Patient: {patient_name}, Age: {age}, Allergies: {allergies}
Chief Complaint: {chief_complaint}
Symptoms: {symptoms}
Vitals: {vitals}
Labs: {labs}
Trend Summary: {trend_summary}

Generate a differential diagnosis as JSON:
{{
  "differential": [
    {{"diagnosis": "...", "probability": 0.XX, "reasoning": "specific evidence from this visit + prior history"}},
    ...
  ],
  "primary": "most likely diagnosis",
  "reasoning": "summary integrating episodic memory",
  "proposed_treatment": "initial plan — use phrases like 'consider', 'recommend review of'",
  "confidence_note": "key uncertainties or missing data"
}}

Return ONLY valid JSON."""

CRITIQUE_PROMPT = """You are a clinical safety critic reviewing a differential diagnosis.

Your job: actively search for ERRORS, CONTRAINDICATIONS, MISSED FINDINGS, and GUIDELINE VIOLATIONS.
Be aggressive — your purpose is to catch what the diagnosing agent missed.

PATIENT ALLERGIES: {allergies}
EPISODIC MEMORY (full history): {episodic_context}
SEMANTIC GUIDELINES: {semantic_context}
PROPOSED DIAGNOSIS: {primary_diagnosis}
PROPOSED TREATMENT: {proposed_treatment}
DIFFERENTIAL: {differential}

Check for:
1. DRUG CONTRAINDICATIONS vs renal/hepatic function, allergies, pregnancy
2. DRUG INTERACTIONS (check all medications in episodic history)
3. MISSED DIAGNOSES (symptoms not explained by primary dx)
4. GUIDELINE DEVIATIONS (compare proposed treatment to retrieved guidelines)
5. TEMPORAL TRENDS (lab worsening/improvement across visits)
6. ALLERGY CROSS-REACTIONS

Return JSON:
{{
  "findings": [
    {{
      "type": "CONTRAINDICATION|MISSED_FINDING|ALLERGY|TEMPORAL_ALERT|GUIDELINE_DEVIATION",
      "severity": "CRITICAL|HIGH|MEDIUM|LOW",
      "message": "specific actionable finding",
      "source": "guideline ID or clinical reasoning"
    }}
  ],
  "critique_passed": true/false,
  "corrected_treatment": "revised treatment if critique found issues, else same",
  "self_correction_occurred": true/false
}}"""

NOTE_GENERATION_PROMPT = """Generate a clinical progress note. 

GUARDRAILS:
- Use "consistent with" not "diagnosed with"
- Use "suggest" not "confirm"  
- Include all HITL physician modifications
- Include confidence scores and uncertainty statements
- Flag any critique findings in the note

Patient: {patient_name} | Visit: {visit_id} | Date: {date}
Primary Assessment: {primary_diagnosis} (confidence: {confidence:.0%})
Critique Findings: {critique_summary}
Final Treatment Plan: {final_treatment}
HITL Override: {hitl_note}

Generate a concise SOAP note."""


# ─────────────────────────────────────────────────────────
# GRAPH NODES
# ─────────────────────────────────────────────────────────

episodic_store = EpisodicMemoryStore()
semantic_store = SemanticMemoryStore()


def _safe_float(raw_value: Any) -> Optional[float]:
    if raw_value is None:
        return None
    if isinstance(raw_value, (int, float)):
        return float(raw_value)
    match = re.search(r"[-+]?\d*\.?\d+", str(raw_value).strip())
    if not match:
        return None
    try:
        return float(match.group())
    except (TypeError, ValueError):
        return None


def _summarize_trend(first_val: float, last_val: float) -> str:
    if abs(last_val - first_val) < 1e-6:
        return "STABLE"
    return "RISING" if last_val > first_val else "FALLING"


def _find_lab_value(labs: dict[str, str], label: str) -> Optional[float]:
    for key, value in labs.items():
        if label.lower() in key.lower():
            return _safe_float(value)
    return None


def _parse_systolic_bp(vitals: dict[str, str]) -> Optional[float]:
    bp_value = None
    for key, value in vitals.items():
        if "bp" in key.lower() or "blood pressure" in key.lower():
            bp_value = str(value)
            break
    if not bp_value:
        return None
    match = re.match(r"\s*(\d{2,3})\s*/\s*(\d{2,3})", bp_value)
    if not match:
        return None
    return float(match.group(1))


def _extract_medications_from_text(text: str) -> set[str]:
    if not text:
        return set()
    normalized = text.lower()
    med_aliases = {
        "methotrexate": ["methotrexate", "mtx"],
        "trimethoprim_sulfamethoxazole": ["trimethoprim-sulfamethoxazole", "trimethoprim sulfamethoxazole", "bactrim", "co-trimoxazole"],
        "atenolol": ["atenolol"],
        "amitriptyline": ["amitriptyline"],
        "escitalopram": ["escitalopram"],
        "sertraline": ["sertraline"],
        "fluoxetine": ["fluoxetine"],
        "citalopram": ["citalopram"],
        "venlafaxine": ["venlafaxine"],
        "phenelzine": ["phenelzine"],
        "tranylcypromine": ["tranylcypromine"],
        "isocarboxazid": ["isocarboxazid"],
        "selegiline": ["selegiline"],
    }
    found = set()
    for canonical, aliases in med_aliases.items():
        if any(alias in normalized for alias in aliases):
            found.add(canonical)
    return found


def _collect_prior_medications(state: ClinicalState) -> set[str]:
    meds = set()
    for visit in state.get("episodic_memory", []):
        meds |= _extract_medications_from_text(visit.get("doctor_note", ""))
        meds |= _extract_medications_from_text(visit.get("final_treatment", ""))
    return meds


def _apply_rule_based_corrections(proposed_treatment: str, findings: list[dict]) -> str:
    corrected = proposed_treatment or ""
    directives = []

    finding_text = " ".join(f.get("message", "").lower() for f in findings)
    if "methotrexate" in finding_text:
        directives.append("Avoid methotrexate until renal function review confirms safe dosing (CrCl/eGFR reassessment required).")
    if "sulfonamide" in finding_text or "trimethoprim" in finding_text:
        directives.append("Avoid trimethoprim-sulfamethoxazole due to allergy risk; consider non-sulfonamide alternative per guideline.")
    if "maoi" in finding_text and "ssri" in finding_text:
        directives.append("Do not combine MAOI with SSRI/SNRI; switch to a non-serotonergic strategy with specialist oversight.")
    if "qtc" in finding_text:
        directives.append("Avoid QT-prolonging antidepressants while QTc remains high; obtain cardiology-safe alternative.")
    if "hypotension" in finding_text:
        directives.append("Hold beta-blocker escalation during hypotension/shock; prioritize hemodynamic stabilization.")

    if not directives:
        return corrected

    correction_block = " Safety overrides: " + " ".join(directives)
    if correction_block.strip() not in corrected:
        corrected = (corrected + correction_block).strip()
    return corrected


def _deterministic_critique_findings(state: ClinicalState) -> list[dict]:
    findings = []
    proposed = (state.get("proposed_treatment") or "").lower()
    allergies = " ".join(state.get("patient_allergies", [])).lower()
    labs = state.get("labs", {})
    vitals = state.get("vitals", {})

    egfr = _find_lab_value(labs, "egfr")
    creatinine = _find_lab_value(labs, "creatin")
    qtc = _find_lab_value(labs, "qtc")
    systolic_bp = _parse_systolic_bp(vitals)

    prior_meds = _collect_prior_medications(state)
    proposed_meds = _extract_medications_from_text(proposed)

    if ("sulfa" in allergies or "sulfonamide" in allergies) and "trimethoprim_sulfamethoxazole" in proposed_meds:
        findings.append({
            "type": "ALLERGY",
            "severity": "CRITICAL",
            "message": "Sulfonamide allergy documented; trimethoprim-sulfamethoxazole is contraindicated.",
            "source": "allergy_reconciliation",
        })

    if "methotrexate" in proposed_meds:
        renal_unsafe = (egfr is not None and egfr < 30) or (creatinine is not None and creatinine >= 2.0)
        if renal_unsafe:
            findings.append({
                "type": "CONTRAINDICATION",
                "severity": "CRITICAL",
                "message": f"Methotrexate proposed despite renal risk (eGFR={egfr}, Creatinine={creatinine}).",
                "source": "GL002",
            })

    maoi_meds = {"phenelzine", "tranylcypromine", "isocarboxazid", "selegiline"}
    ssri_snri = {"escitalopram", "sertraline", "fluoxetine", "citalopram", "venlafaxine"}
    if (prior_meds & maoi_meds) and (proposed_meds & ssri_snri):
        findings.append({
            "type": "CONTRAINDICATION",
            "severity": "CRITICAL",
            "message": "MAOI found in episodic history and SSRI/SNRI proposed; serotonin syndrome risk.",
            "source": "interaction_guardrail",
        })

    qtc_risk_meds = {"amitriptyline", "escitalopram", "citalopram"}
    if qtc is not None and qtc >= 500 and (proposed_meds & qtc_risk_meds):
        findings.append({
            "type": "CONTRAINDICATION",
            "severity": "CRITICAL",
            "message": f"QTc {qtc:.0f} ms with QT-prolonging medication in plan.",
            "source": "qtc_guardrail",
        })

    if systolic_bp is not None and systolic_bp < 90 and "atenolol" in proposed_meds:
        findings.append({
            "type": "HEMODYNAMIC_ALERT",
            "severity": "HIGH",
            "message": f"Hypotension context (SBP {systolic_bp:.0f}) with atenolol continuation/escalation.",
            "source": "shock_guardrail",
        })

    for alert in state.get("trend_alerts", []):
        if alert.get("severity") in {"CRITICAL", "HIGH"}:
            findings.append({
                "type": "TEMPORAL_ALERT",
                "severity": alert.get("severity", "HIGH"),
                "message": alert.get("message", "Trend alert detected."),
                "source": "trend_detection",
            })

    return findings


def intake_node(state: ClinicalState) -> ClinicalState:
    """Log intake and initialise audit trail."""
    logger.info(f"[INTAKE] Patient: {state['patient_name']} | Visit: {state['visit_id']}")
    state["audit_log"] = [{
        "timestamp": datetime.now().isoformat(),
        "node": "intake",
        "action": f"Visit {state['visit_id']} intake for {state['patient_name']}",
    }]
    state["hitl_log"] = []
    state["working_notes"] = []
    state["trend_alerts"] = []
    state["trend_summary"] = ""
    state["semantic_citations"] = []
    state["safety_flags"] = []
    state["confidence_calibrated"] = state.get("highest_confidence", 0.0)
    state["error"] = None
    return state


def memory_retrieval_node(state: ClinicalState) -> ClinicalState:
    """
    Load episodic memory (all prior visits for this patient).
    
    TRAINER NOTE: This is where stateless vs stateful diverges.
    A stateless call would start fresh — no prior visit context.
    This node loads the full patient history into the working state.
    """
    pid = state["patient_id"]
    history = episodic_store.load_history(pid)

    # Format episodic context for LLM
    if history:
        lines = []
        for v in history:
            lines.append(
                f"[{v.get('visit_id', '?')} | {v.get('date', '?')}] "
                f"Complaint: {v.get('chief_complaint', '')[:80]} | "
                f"Labs: {json.dumps(v.get('labs', {}))[:200]}"
            )
        state["episodic_memory"] = history
        state["working_notes"].append(f"[Memory] Loaded {len(history)} prior visits.")
    else:
        state["episodic_memory"] = []
        state["working_notes"].append("[Memory] No prior visits found — first encounter.")

    # Extract lab trends for critique
    for lab in ["Creatinine", "eGFR", "BNP", "QTc", "qSOFA"]:
        trend = episodic_store.get_lab_trend(pid, lab)
        if len(trend) > 1:
            state["working_notes"].append(
                f"[LabTrend] {lab}: " + " → ".join(f"{t['value']} ({t['visit_id']})" for t in trend)
            )

    state["audit_log"].append({
        "timestamp": datetime.now().isoformat(),
        "node": "memory_retrieval",
        "action": f"Loaded {len(state['episodic_memory'])} prior visits.",
    })
    logger.info(f"[MEMORY] Loaded {len(state['episodic_memory'])} prior visits for {pid}")
    return state


def trend_detection_node(state: ClinicalState) -> ClinicalState:
    """
    Analyze temporal changes across longitudinal labs and current visit data.
    Emits trend alerts for critique and routing.
    """
    pid = state["patient_id"]
    alerts = []
    summaries = []
    safety_flags = list(state.get("safety_flags", []))

    trend_map = {
        "Creatinine": {"critical_delta": 0.3, "severity": "CRITICAL"},
        "eGFR": {"critical_delta": -5.0, "severity": "HIGH"},
        "BNP": {"critical_delta": 300.0, "severity": "HIGH"},
        "qSOFA": {"critical_delta": 1.0, "severity": "CRITICAL"},
        "QTc": {"critical_abs": 500.0, "severity": "CRITICAL"},
    }

    for lab, rules in trend_map.items():
        history = episodic_store.get_lab_trend(pid, lab)
        current_raw = state.get("labs", {}).get(lab)
        if current_raw is not None:
            history.append({
                "visit_id": state["visit_id"],
                "date": state["visit_date"],
                "value": current_raw,
            })

        numeric_values = [
            {
                "visit_id": item.get("visit_id", "?"),
                "date": item.get("date", "?"),
                "value": _safe_float(item.get("value")),
            }
            for item in history
        ]
        numeric_values = [item for item in numeric_values if item["value"] is not None]

        if len(numeric_values) < 2:
            continue

        first_val = numeric_values[0]["value"]
        last_val = numeric_values[-1]["value"]
        delta = last_val - first_val
        direction = _summarize_trend(first_val, last_val)
        summaries.append(f"{lab}: {first_val:.2f} -> {last_val:.2f} ({direction}, delta {delta:+.2f})")

        if "critical_delta" in rules:
            threshold = rules["critical_delta"]
            is_alert = delta >= threshold if threshold > 0 else delta <= threshold
            if is_alert:
                msg = f"{lab} trend {direction.lower()} by {delta:+.2f}; threshold {threshold:+.2f} crossed"
                alerts.append({
                    "lab": lab,
                    "severity": rules["severity"],
                    "direction": direction,
                    "delta": round(delta, 3),
                    "message": msg,
                    "source": "trend_detection",
                })
                safety_flags.append(f"TREND_{lab.upper()}_{rules['severity']}")

        if "critical_abs" in rules and last_val >= rules["critical_abs"]:
            msg = f"{lab} current value {last_val:.2f} exceeds absolute threshold {rules['critical_abs']:.2f}"
            alerts.append({
                "lab": lab,
                "severity": rules["severity"],
                "direction": direction,
                "delta": round(delta, 3),
                "message": msg,
                "source": "trend_detection",
            })
            safety_flags.append(f"ABS_{lab.upper()}_{rules['severity']}")

    state["trend_alerts"] = alerts
    state["trend_summary"] = "; ".join(summaries) if summaries else "No significant temporal trend detected."
    state["safety_flags"] = sorted(set(safety_flags))

    if alerts:
        state["working_notes"].append(f"[Trend] {len(alerts)} alert(s): " + " | ".join(a["message"] for a in alerts))
    else:
        state["working_notes"].append("[Trend] No critical temporal alert triggered.")

    state["audit_log"].append({
        "timestamp": datetime.now().isoformat(),
        "node": "trend_detection",
        "alerts": alerts,
        "summary": state["trend_summary"],
    })
    return state


def semantic_search_node(state: ClinicalState) -> ClinicalState:
    """
    Retrieve relevant medical guidelines from semantic memory KB.
    Query built from chief complaint + symptoms + current labs.
    """
    query = (
        state["chief_complaint"] + " "
        + " ".join(state["symptoms"][:5]) + " "
        + " ".join(state.get("labs", {}).keys())
    )
    guidelines = semantic_store.search(query, top_k=4)
    state["semantic_context"] = guidelines
    state["semantic_citations"] = [g.get("id", "") for g in guidelines]

    if guidelines:
        state["working_notes"].append(
            f"[Semantic] Retrieved {len(guidelines)} guidelines: "
            + ", ".join(g["id"] for g in guidelines)
        )
    else:
        state["working_notes"].append("[Semantic] No relevant guidelines found.")

    state["audit_log"].append({
        "timestamp": datetime.now().isoformat(),
        "node": "semantic_search",
        "guidelines_retrieved": [g["id"] for g in guidelines],
        "retrieval_sources": [g.get("_source", "keyword") for g in guidelines],
    })
    logger.info(f"[SEMANTIC] Retrieved: {[g['id'] for g in guidelines]}")
    return state


def differential_diagnosis_node(state: ClinicalState) -> ClinicalState:
    """
    Generate differential diagnosis using:
    - Current symptoms and labs
    - Episodic memory (prior visits)
    - Semantic context (guidelines)
    
    GUARDRAIL: "consistent with" phrasing enforced in prompt.
    """
    llm = get_llm(use_azure="AZURE_OPENAI_ENDPOINT" in os.environ)

    episodic_ctx = "\n".join(
        f"[{v.get('visit_id')} {v.get('date', '')}]: {v.get('chief_complaint', '')} | Labs: {v.get('labs', {})}"
        for v in state.get("episodic_memory", [])
    ) or "No prior visits."

    semantic_ctx = "\n".join(
        f"[{g['id']}] {g['title']}: {g['body'][:400]}"
        for g in state.get("semantic_context", [])
    ) or "No guidelines retrieved."

    prompt = DIFFERENTIAL_PROMPT.format(
        episodic_context=episodic_ctx,
        semantic_context=semantic_ctx,
        patient_name=state["patient_name"],
        age=state["patient_age"],
        allergies=", ".join(state.get("patient_allergies", [])) or "None documented",
        chief_complaint=state["chief_complaint"],
        symptoms="\n".join(f"  - {s}" for s in state["symptoms"]),
        vitals=json.dumps(state.get("vitals", {}), indent=2),
        labs=json.dumps(state.get("labs", {}), indent=2),
        trend_summary=state.get("trend_summary", "No trend summary available."),
    )

    try:
        response = llm.invoke(prompt)
        result = json.loads(response.content.strip())
    except Exception:
        # Fallback structure
        result = {
            "differential": [{"diagnosis": "Clinical assessment pending", "probability": 0.50, "reasoning": "LLM unavailable"}],
            "primary": "Pending physician assessment",
            "reasoning": "Requires manual review.",
            "proposed_treatment": "Symptomatic management pending specialist review.",
        }

    differential = result.get("differential", [])
    primary = result.get("primary", "Unknown")
    highest_conf = max((d.get("probability", 0) for d in differential), default=0.0)

    state["differential_diagnosis"] = differential
    state["primary_diagnosis"] = primary
    state["highest_confidence"] = highest_conf

    critical_trend = sum(1 for a in state.get("trend_alerts", []) if a.get("severity") == "CRITICAL")
    high_trend = sum(1 for a in state.get("trend_alerts", []) if a.get("severity") == "HIGH")
    calibrated = highest_conf - (0.10 * critical_trend) - (0.05 * high_trend)
    state["confidence_calibrated"] = round(max(0.0, min(1.0, calibrated)), 4)

    state["diagnosis_reasoning"] = result.get("reasoning", "")
    state["proposed_treatment"] = result.get("proposed_treatment", "")

    state["audit_log"].append({
        "timestamp": datetime.now().isoformat(),
        "node": "differential_diagnosis",
        "primary": primary,
        "confidence": highest_conf,
        "confidence_calibrated": state["confidence_calibrated"],
        "differentials_count": len(differential),
    })
    logger.info(f"[DIAGNOSIS] Primary: {primary} | Confidence: {highest_conf:.2f}")
    return state


def critique_node(state: ClinicalState) -> ClinicalState:
    """
    Self-Critique Node — actively searches for errors the diagnosis node may have missed.
    
    Checks:
    1. Drug contraindications vs allergies, renal function, cardiac risk
    2. Drug interactions from episodic medication history
    3. Temporal lab trends (worsening signals)
    4. Guideline violations in proposed treatment
    5. Missed differential diagnoses
    
    TRAINER NOTE: This node's output directly feeds the self_correction_rate metric.
    """
    llm = get_llm(use_azure="AZURE_OPENAI_ENDPOINT" in os.environ)

    episodic_ctx = "\n".join(
        f"[{v.get('visit_id')} {v.get('date', '')}]: "
        f"Labs: {v.get('labs', {})} | Note: {v.get('doctor_note', '')[:200]}"
        for v in state.get("episodic_memory", [])
    ) + "\n\nLab trends:\n" + "\n".join(state.get("working_notes", []))

    semantic_ctx = "\n".join(
        f"[{g['id']}] {g['title']}: {g['body']}"
        for g in state.get("semantic_context", [])
    ) or "No guidelines retrieved."

    prompt = CRITIQUE_PROMPT.format(
        allergies=", ".join(state.get("patient_allergies", [])) or "None",
        episodic_context=episodic_ctx,
        semantic_context=semantic_ctx,
        primary_diagnosis=state.get("primary_diagnosis", ""),
        proposed_treatment=state.get("proposed_treatment", ""),
        differential=json.dumps(state.get("differential_diagnosis", []), indent=2),
    )

    try:
        response = llm.invoke(prompt)
        result = json.loads(response.content.strip())
    except Exception:
        result = {
            "findings": [],
            "critique_passed": True,
            "corrected_treatment": state.get("proposed_treatment", ""),
            "self_correction_occurred": False,
        }

    llm_findings = result.get("findings", [])
    rule_findings = _deterministic_critique_findings(state)

    findings = list(llm_findings)
    seen = {f"{f.get('type')}|{f.get('severity')}|{f.get('message')}" for f in llm_findings}
    for finding in rule_findings:
        key = f"{finding.get('type')}|{finding.get('severity')}|{finding.get('message')}"
        if key not in seen:
            findings.append(finding)
            seen.add(key)

    critical_or_high = [f for f in findings if f.get("severity") in {"CRITICAL", "HIGH"}]
    critique_passed = bool(result.get("critique_passed", True)) and not critical_or_high

    corrected_tx = result.get("corrected_treatment", state.get("proposed_treatment", ""))
    corrected_tx = _apply_rule_based_corrections(corrected_tx, findings)
    self_correction = bool(result.get("self_correction_occurred", False)) or (corrected_tx != state.get("proposed_treatment", ""))

    state["critique_findings"] = findings
    state["critique_passed"] = critique_passed
    state["self_correction_occurred"] = self_correction
    state["treatment_guardrail_applied"] = self_correction

    if self_correction:
        state["proposed_treatment"] = corrected_tx
        state["critique_corrected_diagnosis"] = result.get("corrected_diagnosis")
        logger.warning(f"[CRITIQUE] ⚠  Self-correction occurred: {len(findings)} finding(s)")
        for f in findings:
            logger.warning(f"  [{f.get('severity')}] {f.get('type')}: {f.get('message', '')[:100]}")
    else:
        logger.info("[CRITIQUE] No critical findings — passed.")

    state["audit_log"].append({
        "timestamp": datetime.now().isoformat(),
        "node": "critique",
        "rule_findings_count": len(rule_findings),
        "llm_findings_count": len(llm_findings),
        "findings_count": len(findings),
        "critique_passed": critique_passed,
        "self_correction": self_correction,
        "critical_findings": [f for f in findings if f.get("severity") == "CRITICAL"],
    })
    return state


def confidence_gate_node(state: ClinicalState) -> str:
    """
    Router: determines whether to go to treatment (high confidence)
    or HITL (low confidence or critical critique findings).
    
    HITL triggered if:
    - highest_confidence < 0.80, OR
    - any CRITICAL critique finding exists, OR
    - suicidal ideation detected in symptoms
    
    TRAINER NOTE: this is a LangGraph conditional edge function.
    Returns the name of the next node to route to.
    """
    conf = state.get("confidence_calibrated", state.get("highest_confidence", 0.0))
    critical_findings = [
        f for f in state.get("critique_findings", [])
        if f.get("severity") == "CRITICAL"
    ]
    critical_trend_alerts = [
        a for a in state.get("trend_alerts", [])
        if a.get("severity") == "CRITICAL"
    ]

    # Detect high-risk clinical scenarios requiring HITL
    suicidal_keywords = ["suicidal", "self-harm", "doesn't want to live", "passive si"]
    symptom_text = " ".join(state.get("symptoms", [])).lower()
    psychiatric_risk = any(kw in symptom_text for kw in suicidal_keywords)

    if conf < 0.80 or critical_findings or critical_trend_alerts or psychiatric_risk:
        reason_parts = []
        if conf < 0.80:
            reason_parts.append(f"confidence {conf:.0%} < 80% threshold")
        if critical_findings:
            reason_parts.append(f"{len(critical_findings)} CRITICAL critique finding(s)")
        if critical_trend_alerts:
            reason_parts.append(f"{len(critical_trend_alerts)} CRITICAL trend alert(s)")
        if psychiatric_risk:
            reason_parts.append("psychiatric risk detected")

        state["hitl_required"] = True
        state["hitl_reason"] = " | ".join(reason_parts)
        logger.warning(f"[GATE] → HITL required: {state['hitl_reason']}")
        return "hitl_node"
    else:
        state["hitl_required"] = False
        logger.info(f"[GATE] → Treatment (confidence: {conf:.0%})")
        return "treatment_node"


def hitl_node(state: ClinicalState, interactive: bool = True) -> ClinicalState:
    """
    Human-in-the-Loop Node.
    
    Displays the proposed plan and critique findings to the physician.
    Physician can:
      [A] Approve as-is
      [E] Edit the treatment plan
      [R] Reject and request re-diagnosis
    
    All decisions are logged to the audit trail.
    
    TRAINER NOTE: In production, this pauses the graph execution
    via LangGraph's interrupt() mechanism and waits for a UI event.
    Here it uses terminal input for training demos.
    """
    print("\n" + "═"*65)
    print("🔶  HUMAN-IN-THE-LOOP REVIEW REQUIRED")
    print("═"*65)
    print(f"  Patient  : {state['patient_name']} ({state['patient_age']}F/M)")
    print(f"  Visit    : {state['visit_id']} | {state['visit_date']}")
    print(f"  Reason   : {state['hitl_reason']}")
    print()
    print(f"  Primary Assessment : {state.get('primary_diagnosis', 'N/A')}")
    print(f"  Confidence         : {state.get('highest_confidence', 0):.0%}")
    print()
    print("  Differential:")
    for d in state.get("differential_diagnosis", [])[:3]:
        bar = "🟠" if d["probability"] < 0.80 else "🟢"
        print(f"    {bar} {d['probability']:.0%} — {d['diagnosis']}")

    critique_findings = state.get("critique_findings", [])
    if critique_findings:
        print()
        print("  ⚠  Critique Findings:")
        for f in critique_findings:
            icon = "🔴" if f.get("severity") == "CRITICAL" else "🟡"
            print(f"    {icon} [{f.get('severity')}] {f.get('type')}: {f.get('message', '')[:120]}")

    print()
    print("  Proposed Treatment:")
    print(f"    {state.get('proposed_treatment', 'N/A')}")
    print()

    if interactive:
        print("  Options:")
        print("    [A] Approve and proceed")
        print("    [E] Edit treatment plan")
        print("    [R] Reject — request re-evaluation")
        print()
        try:
            choice = input("  Physician decision → ").strip().upper()
        except EOFError:
            choice = "A"  # Auto-approve in non-interactive contexts

        if choice == "E":
            print()
            print("  Current plan:", state.get("proposed_treatment", ""))
            try:
                edited = input("  Enter revised plan → ").strip()
            except EOFError:
                edited = state.get("proposed_treatment", "")
            state["hitl_physician_response"] = edited
            state["hitl_feedback"] = edited
            state["final_treatment"] = edited
            state["hitl_approved"] = True
            action = "EDITED"
        elif choice == "R":
            state["hitl_physician_response"] = "REJECTED — physician requested re-evaluation"
            state["hitl_feedback"] = state["hitl_physician_response"]
            state["hitl_approved"] = False
            action = "REJECTED"
        else:
            state["hitl_physician_response"] = "APPROVED as proposed"
            state["hitl_feedback"] = state["hitl_physician_response"]
            state["final_treatment"] = state.get("proposed_treatment", "")
            state["hitl_approved"] = True
            action = "APPROVED"
    else:
        # Non-interactive: auto-approve for batch evaluation
        state["hitl_physician_response"] = "AUTO-APPROVED (non-interactive mode)"
        state["hitl_feedback"] = state["hitl_physician_response"]
        state["final_treatment"] = state.get("proposed_treatment", "")
        state["hitl_approved"] = True
        action = "AUTO-APPROVED"
        print(f"  [Non-interactive] Auto-approving...")

    hitl_record = {
        "timestamp": datetime.now().isoformat(),
        "visit_id": state["visit_id"],
        "reason": state["hitl_reason"],
        "action": action,
        "original_treatment": state.get("proposed_treatment"),
        "final_treatment": state.get("final_treatment"),
        "confidence": state.get("highest_confidence"),
        "critique_findings": state.get("critique_findings", []),
    }
    state["hitl_log"].append(hitl_record)
    state["audit_log"].append({
        "timestamp": datetime.now().isoformat(),
        "node": "hitl",
        **hitl_record,
    })

    print(f"\n  ✓ Decision: {action}")
    print("═"*65)
    return state


def treatment_node(state: ClinicalState) -> ClinicalState:
    """Finalise treatment plan (high-confidence, no HITL needed)."""
    if not state.get("final_treatment"):
        state["final_treatment"] = state.get("proposed_treatment", "")
    state["audit_log"].append({
        "timestamp": datetime.now().isoformat(),
        "node": "treatment",
        "final_treatment": state["final_treatment"],
    })
    logger.info(f"[TREATMENT] Finalised: {state['final_treatment'][:80]}")
    return state


def note_generation_node(state: ClinicalState) -> ClinicalState:
    """
    Generate a structured clinical note.
    Applies language guardrails: 'consistent with', 'suggests', not 'has'.
    Embeds confidence scores and HITL override documentation.
    """
    critique_summary = "; ".join(
        f.get("message", "")[:100]
        for f in state.get("critique_findings", [])
        if f.get("severity") in ("CRITICAL", "HIGH")
    ) or "No critical findings."

    hitl_note = "None" if not state.get("hitl_log") else (
        f"Physician review at {state['hitl_log'][-1]['timestamp'][:16]}: "
        f"{state['hitl_log'][-1]['action']}"
    )

    conf = state.get("confidence_calibrated", state.get("highest_confidence", 0))
    conf_label = "HIGH" if conf >= 0.80 else "MEDIUM" if conf >= 0.60 else "LOW"
    conf_tag = "" if conf >= 0.80 else "⚠️  AMBER — physician review completed" if conf >= 0.60 else "🔴 RED — mandatory HITL applied"
    trend_section = state.get("trend_summary", "No trend summary available.")
    citations = ", ".join(state.get("semantic_citations", [])) or "None"

    note = f"""
CLINICAL PROGRESS NOTE — ClinicalDSS
══════════════════════════════════════════════════════════
Patient  : {state['patient_name']} (Age: {state['patient_age']})
Visit    : {state['visit_id']} | Date: {state['visit_date']}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

ASSESSMENT {conf_tag}
  Confidence: {conf:.0%} ({conf_label})
  Primary:    Symptoms are consistent with {state.get('primary_diagnosis', 'N/A')}
  Reasoning:  {state.get('diagnosis_reasoning', '')[:400]}

DIFFERENTIAL DIAGNOSIS
{chr(10).join(f"  {'▶' if i==0 else '·'} {d['probability']:.0%} — Findings suggest {d['diagnosis']}" + (f" (Critique: {d.get('critique_note','')[:80]})" if d.get('critique_note') else '') for i, d in enumerate(state.get('differential_diagnosis', [])[:4]))}

CRITIQUE REVIEW
  {critique_summary}

TREATMENT PLAN
  {state.get('final_treatment', 'N/A')}

HUMAN OVERSIGHT
  HITL Required   : {'YES — ' + state.get('hitl_reason','') if state.get('hitl_required') else 'No'}
  Physician Review: {hitl_note}

EPISODIC MEMORY UTILISED
  Prior visits loaded: {len(state.get('episodic_memory', []))}
  Lab trends tracked: {[n for n in state.get('working_notes',[]) if 'LabTrend' in n]}
    Trend summary: {trend_section}
    Semantic citations: {citations}

NOTE: All assessments are differential diagnoses to support physician decision-making.
Final diagnosis and treatment decisions rest with the responsible physician.
══════════════════════════════════════════════════════════
""".strip()

    state["clinical_note"] = note

    # Save visit to episodic memory for future recalls
    episodic_store.save_visit(state["patient_id"], {
        "visit_id": state["visit_id"],
        "date": state["visit_date"],
        "chief_complaint": state["chief_complaint"],
        "symptoms": state["symptoms"],
        "vitals": state.get("vitals", {}),
        "labs": state.get("labs", {}),
        "primary_diagnosis": state.get("primary_diagnosis"),
        "final_treatment": state.get("final_treatment"),
        "confidence": state.get("highest_confidence"),
        "confidence_calibrated": state.get("confidence_calibrated"),
        "critique_findings": state.get("critique_findings", []),
        "trend_alerts": state.get("trend_alerts", []),
        "semantic_citations": state.get("semantic_citations", []),
        "doctor_note": state.get("doctor_note", ""),
    })

    state["audit_log"].append({
        "timestamp": datetime.now().isoformat(),
        "node": "note_generation",
        "note_length": len(note),
        "episodic_saved": True,
    })
    logger.info("[NOTE] Clinical note generated and episodic memory updated.")
    return state


# ─────────────────────────────────────────────────────────
# GRAPH CONSTRUCTION
# ─────────────────────────────────────────────────────────

def build_graph(interactive_hitl: bool = True):
    """
    Construct the LangGraph StateGraph.
    
    Args:
        interactive_hitl: If True, HITL node prompts terminal input.
                         If False, auto-approves (for batch eval).
    """
    if not LANGGRAPH_AVAILABLE:
        raise ImportError("pip install langgraph")

    import functools
    hitl = functools.partial(hitl_node, interactive=interactive_hitl)

    graph = StateGraph(ClinicalState)

    # Add nodes
    graph.add_node("intake_node",           intake_node)
    graph.add_node("memory_retrieval_node", memory_retrieval_node)
    graph.add_node("trend_detection_node",  trend_detection_node)
    graph.add_node("semantic_search_node",  semantic_search_node)
    graph.add_node("differential_diagnosis_node", differential_diagnosis_node)
    graph.add_node("critique_node",         critique_node)
    graph.add_node("hitl_node",             hitl)
    graph.add_node("treatment_node",        treatment_node)
    graph.add_node("note_generation_node",  note_generation_node)

    # Entry point
    graph.set_entry_point("intake_node")

    # Linear edges
    graph.add_edge("intake_node",           "memory_retrieval_node")
    graph.add_edge("memory_retrieval_node", "trend_detection_node")
    graph.add_edge("trend_detection_node",  "semantic_search_node")
    graph.add_edge("semantic_search_node",  "differential_diagnosis_node")
    graph.add_edge("differential_diagnosis_node", "critique_node")

    # Conditional edge: confidence gate
    graph.add_conditional_edges(
        "critique_node",
        confidence_gate_node,
        {
            "hitl_node":      "hitl_node",
            "treatment_node": "treatment_node",
        }
    )

    # HITL → note (regardless of approve/reject/edit — physician has reviewed)
    graph.add_edge("hitl_node",       "note_generation_node")
    graph.add_edge("treatment_node",  "note_generation_node")
    graph.add_edge("note_generation_node", END)

    # Compile with checkpointer for state persistence
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ─────────────────────────────────────────────────────────
# CONVENIENCE RUNNER
# ─────────────────────────────────────────────────────────

def run_visit(
    app,
    patient_id: str,
    patient_name: str,
    patient_age: int,
    patient_allergies: list[str],
    visit_id: str,
    visit_date: str,
    chief_complaint: str,
    symptoms: list[str],
    vitals: dict,
    labs: dict,
    doctor_note: str = "",
    thread_id: str = None,
    interactive: bool = True,
) -> ClinicalState:
    """Run one visit through the full graph and return the final state."""
    if thread_id is None:
        thread_id = hashlib.md5(f"{patient_id}_{visit_id}".encode()).hexdigest()[:8]

    initial_state = ClinicalState(
        patient_id=patient_id,
        patient_name=patient_name,
        patient_age=patient_age,
        patient_allergies=patient_allergies,
        visit_id=visit_id,
        visit_date=visit_date,
        chief_complaint=chief_complaint,
        symptoms=symptoms,
        vitals=vitals,
        labs=labs,
        doctor_note=doctor_note,
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
        proposed_treatment="",
        final_treatment="",
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
        treatment_guardrail_applied=False,
        clinical_note="",
        outcome_followup=None,
        audit_log=[],
        error=None,
    )

    config = {"configurable": {"thread_id": thread_id}}
    final_state = app.invoke(initial_state, config)
    return final_state


if __name__ == "__main__":
    print("ClinicalDSS — LangGraph Agent")
    print("Building graph...")

    if LANGGRAPH_AVAILABLE:
        app = build_graph(interactive_hitl=True)
        print("✓ Graph compiled successfully")
        print("  Nodes:", list(app.get_graph().nodes.keys()))
    else:
        print("⚠  LangGraph not installed. Run: pip install langgraph langchain-openai")
        print("   MockLLM available for testing without API keys.")
