# ClinicalDSS — Stateful Agentic Clinical Decision Support
### Episodic Memory × Semantic Memory × Trend Detection × Self-Critique × Human-in-the-Loop
**CitiusTech Gen AI & Agentic AI Training —  Project 3**

---

## The Problem a Stateless LLM Cannot Solve

```
Doctor: "The patient came in 2 weeks ago with creatinine 1.8. Today it's 2.3.
         I'm thinking of starting methotrexate for her new RA diagnosis. Thoughts?"
```

A stateless LLM has no idea the creatinine was 1.8 two weeks ago. It has no memory of the prior visit. It may confidently suggest methotrexate. But:

- Creatinine 2.3, age 67 → CrCl ~26 mL/min
- ACR 2021: Methotrexate **ABSOLUTELY CONTRAINDICATED** if CrCl < 30 mL/min

This is a potentially fatal error from a **stateless** system that looks at one snapshot.

ClinicalDSS builds the architecture that catches this.

---

## Functional Agent Diagram

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                        ClinicalDSS — Agent Interaction Map                      ║
╚══════════════════════════════════════════════════════════════════════════════════╝

 ┌─────────────────────────────────────────────────────────────────────────────┐
 │  EXTERNAL WORLD                                                              │
 │  Patient Vitals · Lab Results · Chief Complaint · Medication List            │
 └─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │  [1] INTAKE AGENT                                                            │
 │  • Parses raw clinical input into structured state                           │
 │  • Validates required fields (patient_id, chief_complaint, vitals)           │
 │  • Initialises audit trail with timestamp                                    │
 │  • Detects missing critical data (no creatinine? → flag for HITL)            │
 │  OUTPUT → structured ClinicalState TypedDict                                 │
 └─────────────────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                       ▼
 ┌─────────────────────┐              ┌──────────────────────────┐
 │  [2] EPISODIC MEMORY │              │  [3] SEMANTIC MEMORY     │
 │  RETRIEVAL AGENT     │              │  RETRIEVAL AGENT         │
 │                      │              │                          │
 │  Reads per-patient   │              │  Searches guideline KB   │
 │  visit history from  │              │  (GL001-010) by keyword  │
 │  episodic_store/     │              │  overlap + embedding     │
 │                      │              │  similarity              │
 │  Computes lab trends │              │                          │
 │  (see Trend          │              │  Returns: AHA, KDIGO,    │
 │  Detection section)  │              │  ACR, NICE, Sepsis-3     │
 │                      │              │  guideline snippets      │
 │  OUTPUT →            │              │  OUTPUT →                │
 │  prior_visits[]      │              │  semantic_context[]      │
 │  lab_trends{}        │              │  guideline_ids[]         │
 └──────────┬───────────┘              └───────────┬──────────────┘
            └───────────────────┬──────────────────┘
                                ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │  [4] DIFFERENTIAL DIAGNOSIS AGENT                                            │
 │                                                                              │
 │  INPUTS: current vitals + labs + chief_complaint + prior_visits +            │
 │          lab_trends + semantic_context                                       │
 │                                                                              │
 │  TASKS:                                                                      │
 │  • Generates ranked differential (3-5 diagnoses with probabilities)          │
 │  • Enforces language guardrails (no definitive language)                     │
 │  • Incorporates temporal context from episodic memory                        │
 │  • Cites retrieved guidelines where applicable                               │
 │                                                                              │
 │  OUTPUT → { differential[], primary_dx, confidence, reasoning,              │
 │             proposed_treatment, proposed_drugs[] }                           │
 └─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │  [5] CRITIQUE AGENT  ◄── THE SAFETY NET                                     │
 │                                                                              │
 │  INPUTS: proposed_drugs[] + allergy_list + episodic_meds + lab_trends +     │
 │          semantic_context + [OPTIONAL: external API results]                 │
 │                                                                              │
 │  SAFETY CHECKS (see full detail below):                                      │
 │  ├── Allergy cross-reactivity check                                          │
 │  ├── Renal function threshold check (Cockcroft-Gault CrCl per drug)         │
 │  ├── QTc prolongation risk (CredibleMeds categories)                         │
 │  ├── Drug-drug interaction check (episodic med list × proposed drugs)        │
 │  ├── Hepatic safety check (Child-Pugh × drug metabolism pathway)             │
 │  ├── Beers Criteria check (age ≥ 65)                                         │
 │  ├── Prescribing cascade detection                                           │
 │  ├── AKI temporal flag (KDIGO: Cr rise ≥ 0.3 in 48h)                        │
 │  └── [STRETCH] Real-time FDA / PubMed / Serper web search                   │
 │                                                                              │
 │  OUTPUT → { findings[], critique_passed, corrected_treatment,                │
 │             self_correction_occurred, confidence_delta }                     │
 └─────────────────────────────────────────────────────────────────────────────┘
                                │
              ┌─────────────────┴─────────────────┐
              │  Confidence Gate Decision          │
              │  conf ≥ 0.80 AND no CRITICAL       │
              │  AND no psychiatric_risk?          │
              └──────────┬────────────────────────┘
               YES ▼                        NO ▼
 ┌──────────────────────────┐     ┌──────────────────────────────┐
 │  [6] TREATMENT AGENT     │     │  [7] HUMAN-IN-THE-LOOP AGENT │
 │                          │     │                              │
 │  Finalises treatment     │     │  Presents critique + dx      │
 │  plan from corrected     │     │  to physician in structured  │
 │  output                  │     │  format                      │
 │                          │     │                              │
 │  Applies dose adjustments│     │  Captures: APPROVE / EDIT /  │
 │  for renal/hepatic       │     │  REJECT with free-text reason│
 │  impairment              │     │                              │
 │                          │     │  Logs everything to          │
 │  OUTPUT → treatment plan │     │  hitl_log[] for audit        │
 └──────────┬───────────────┘     └──────────────┬───────────────┘
            └─────────────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │  [8] NOTE GENERATION + EPISODIC MEMORY WRITE AGENT                          │
 │                                                                              │
 │  • Generates structured clinical note from finalised state                  │
 │  • Enforces language guardrails (validates no "patient HAS" phrasing)       │
 │  • Appends mandatory disclaimer                                              │
 │  • Writes this visit to episodic_store/ (becomes prior visit for next run)  │
 │  • Closes audit_log with node completion timestamps                          │
 │                                                                              │
 │  OUTPUT → clinical_note.txt + updated episodic store                        │
 └─────────────────────────────────────────────────────────────────────────────┘

Legend:
  [N] = LangGraph node (one per agent)
  ──► = data flow
  ◄── = reads from
  Critique findings that are CRITICAL automatically trigger HITL regardless of confidence
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 LangGraph StateGraph                        │
│                                                             │
│  intake_node ──► memory_retrieval_node                     │
│                         │                                   │
│                  semantic_search_node                       │
│                         │                                   │
│               differential_diagnosis_node                   │
│                         │                                   │
│                   critique_node ─── catches contraindications│
│                     /        \                              │
│          [confidence ≥ 0.80]  [confidence < 0.80 OR        │
│                 │              CRITICAL finding OR SI]      │
│         treatment_node              hitl_node               │
│                 │                       │                   │
│          note_generation_node  ─────────┘                   │
│                 │                                           │
│               [END]                                         │
│         (episodic memory updated)                           │
└─────────────────────────────────────────────────────────────┘
```

### Memory Architecture

```
┌──────────────────────────────────────────────────────┐
│                  Memory Layers                        │
├──────────────────────────────────────────────────────┤
│  Episodic Memory  │ All prior visits for this patient │
│  (Long-term)      │ Labs, vitals, diagnoses, drugs    │
│                   │ Stored per patient ID             │
├──────────────────────────────────────────────────────┤
│  Semantic Memory  │ Medical guidelines KB (GL001-010) │
│  (Declarative)    │ Retrieved by keyword/embedding    │
│                   │ AHA, ADA, KDIGO, NICE, Sepsis-3   │
├──────────────────────────────────────────────────────┤
│  Working Memory   │ Current visit state               │
│  (Session)        │ Lab trends computed from episodic │
│                   │ Critique findings accumulate here │
└──────────────────────────────────────────────────────┘
```

---

## Trend Detection — Techniques, Tradeoffs, and Implementation

The `memory_retrieval_node` must compute **lab trends** from episodic history. This is more nuanced than it looks — the right algorithm depends on the lab, the number of data points, and the clinical question. Participants must understand the tradeoffs and choose or combine approaches.

### Why trend detection matters clinically

| Lab | Static snapshot | With trend |
|-----|----------------|------------|
| Creatinine 2.3 | Borderline — maybe ok for methotrexate | Creatinine 1.8→2.1→2.3 over 4 weeks → AKI trajectory, ABSOLUTE contraindication |
| BNP 450 | Elevated, concerning | BNP 200→320→450 → decompensating heart failure, escalate NOW |
| qSOFA 1 | Low sepsis risk | qSOFA 0→1→3 in 48h → rapid deterioration, meets Sepsis-3 criteria |
| eGFR 38 | Stage 3b CKD — stable | eGFR 55→45→38 over 6 months → progressive CKD, adjust drug selection |

### Technique 1 — Simple Delta (Baseline)

```python
def simple_delta(values: list[float], timestamps: list[str]) -> dict:
    if len(values) < 2:
        return {"trend": "insufficient_data"}
    delta = values[-1] - values[-2]
    pct_change = (delta / values[-2]) * 100
    return {"absolute_change": delta, "pct_change": pct_change,
            "direction": "rising" if delta > 0 else "falling"}
```

| | |
|--|--|
| **Pros** | Trivially simple, zero dependencies, clinically intuitive, KDIGO-compliant (uses pair-wise delta) |
| **Cons** | Single-point noise distorts signal (outlier lab → false alarm), no smoothing, cannot distinguish sustained rise from a one-off spike |
| **Use when** | Exactly 2 data points, or applying domain rule-based thresholds (KDIGO: Cr ≥ 0.3 in 48h) |

### Technique 2 — Domain Rule-Based Threshold Flags

Hard-code clinically validated alarm rules directly from guidelines:

```python
TREND_RULES = {
    "Creatinine": [
        # KDIGO AKI criterion 1
        {"rule": "AKI_48h", "fn": lambda vals, times: (
            vals[-1] - vals[-2] >= 0.3 and hours_between(times[-2], times[-1]) <= 48
        ), "severity": "CRITICAL", "message": "KDIGO AKI Stage 1: Cr rise ≥0.3 in 48h"},
        # KDIGO AKI criterion 2
        {"rule": "AKI_7d", "fn": lambda vals, times: (
            vals[-1] >= 1.5 * min(vals)
        ), "severity": "CRITICAL", "message": "KDIGO AKI Stage 1: Cr ≥1.5× baseline in 7d"},
    ],
    "qSOFA": [
        {"rule": "Sepsis3_rapid", "fn": lambda vals, times: (
            vals[-1] >= 2 and vals[-1] > vals[0]
        ), "severity": "CRITICAL", "message": "qSOFA ≥2 and rising — Sepsis-3 screen positive"},
    ],
    "QTc": [
        {"rule": "QTc_danger", "fn": lambda vals, _: vals[-1] >= 500,
         "severity": "CRITICAL", "message": "QTc ≥500ms — Torsades risk, halt QT-prolonging drugs"},
    ],
}
```

| | |
|--|--|
| **Pros** | Directly guideline-backed (KDIGO, Sepsis-3, AHA), zero false negative risk for known patterns, fast and explainable |
| **Cons** | Does not generalise — each rule must be hand-coded, misses novel deterioration patterns, brittle to edge cases |
| **Use when** | Implementing the minimum viable critique for patient safety |

### Technique 3 — Linear Regression Slope

Fit a line to `(timestamp, lab_value)` pairs and use the slope as the rate of change:

```python
import numpy as np

def linear_regression_trend(values: list[float], timestamps: list[str]) -> dict:
    if len(values) < 3:
        return simple_delta(values, timestamps)
    # Convert timestamps to hours since first measurement
    t = np.array([(parse_ts(ts) - parse_ts(timestamps[0])).total_seconds() / 3600
                  for ts in timestamps])
    v = np.array(values)
    slope, intercept = np.polyfit(t, v, 1)  # units: lab_unit/hour
    r_squared = np.corrcoef(t, v)[0,1] ** 2
    return {
        "slope_per_hour": slope,
        "slope_per_week": slope * 168,
        "r_squared": r_squared,
        "extrapolated_7d": intercept + slope * (t[-1] + 168),
        "trend": "rising" if slope > 0.001 else "falling" if slope < -0.001 else "stable"
    }
```

| | |
|--|--|
| **Pros** | Robust to single-point noise, gives a *rate* (e.g., "creatinine rising 0.05 mg/dL/day"), can project future values, works with 3+ data points |
| **Cons** | Assumes monotonic linear trajectory — misses U-shapes or plateau-then-spike patterns, sensitive to irregular sampling intervals |
| **Use when** | ≥3 data points over weeks; chronic disease monitoring (CKD progression, HbA1c trajectory) |

### Technique 4 — CUSUM (Cumulative Sum Control Chart)

CUSUM detects a sustained *shift* in a process, even when each individual change is small:

```python
def cusum_detect(values: list[float], k: float = 0.5, h: float = 4.0) -> dict:
    """
    k = allowable slack (in std devs) before CUSUM counts a deviation
    h = decision threshold — alarm fires when cumulative sum exceeds h
    """
    mean = np.mean(values[:-1])  # baseline from prior visits
    std  = np.std(values[:-1]) or 1.0
    s_hi, s_lo = 0.0, 0.0
    alarms = []
    for i, v in enumerate(values):
        z = (v - mean) / std
        s_hi = max(0, s_hi + z - k)
        s_lo = max(0, s_lo - z - k)
        if s_hi > h:
            alarms.append({"index": i, "direction": "upward_drift", "cusum": s_hi})
        if s_lo > h:
            alarms.append({"index": i, "direction": "downward_drift", "cusum": s_lo})
    return {"alarms": alarms, "drift_detected": len(alarms) > 0}
```

| | |
|--|--|
| **Pros** | Highly sensitive to sustained shifts even when each increment is small (catches slow AKI), used in clinical surveillance systems, low false-positive rate for gradual trends |
| **Cons** | Requires a stable baseline distribution to calibrate `k` and `h`, more complex to tune, harder to explain to clinicians than "rose by 0.3" |
| **Use when** | Patient has many prior visits (≥5); monitoring labs that drift slowly (eGFR, HbA1c, BNP) |

### Technique 5 — LLM-Mediated Trend Narrative

Pass the raw time series to the LLM and ask it to produce a clinical narrative:

```python
TREND_PROMPT = """
You are a clinical pharmacist reviewing lab trends for patient safety.

Lab: {lab_name}
Time series (oldest to newest):
{formatted_series}

Clinical context: {chief_complaint}

Provide:
1. A one-sentence clinical interpretation of this trend
2. Whether the trend is clinically significant (YES/NO) and why
3. Any urgent action warranted based on this trend alone

Do NOT suggest diagnosis. Focus only on the trend direction and rate.
"""
```

| | |
|--|--|
| **Pros** | Handles complex patterns (plateau → spike → recovery), can incorporate clinical context, generates human-readable explanations, no algorithmic tuning required |
| **Cons** | Hallucination risk — LLM may fabricate trend significance, expensive (extra LLM call per lab), non-deterministic (same series may get different interpretation on re-runs), not guideline-anchored |
| **Use when** | Generating the narrative for the clinical note; NOT for triggering safety alarms (use rule-based for safety-critical decisions) |

### Recommended Hybrid Strategy

```
Lab values available:
  2 points  → Simple Delta + Domain Rule-Based Flags
  3-5 points → Linear Regression + Domain Rules
  6+ points  → CUSUM + Linear Regression + Domain Rules
  Any count  → LLM Narrative (for note generation only, not for safety gates)
```

---

## The Critique Agent — Deep Dive

The critique node is the most clinically significant component. It must act like a hostile expert reviewer trying to find every way the proposed treatment could harm the patient. Below are the full battery of checks participants must implement, with innovative extensions.

### Core Safety Checks

#### Check 1 — Allergy Cross-Reactivity Engine

Simple string matching is not enough. A patient allergic to penicillin is not just at risk from penicillin — they have a 1-2% cross-reactivity risk with cephalosporins.

```python
ALLERGY_CROSS_REACTIVITY = {
    "penicillin": {
        "direct":     ["amoxicillin", "ampicillin", "piperacillin", "nafcillin"],
        "cross_react":["cephalexin", "ceftriaxone", "cefazolin"],  # 1-2% risk
        "note": "Carbapenems: ~1% risk. Aztreonam: safe."
    },
    "sulfonamide": {
        "direct":     ["trimethoprim-sulfamethoxazole", "sulfadiazine"],
        "cross_react":["furosemide", "hydrochlorothiazide", "celecoxib"],  # structural similarity
        "note": "Non-antimicrobial sulfonamides: risk is disputed but flag for review"
    },
    "nsaid": {
        "direct":     ["ibuprofen", "naproxen", "diclofenac"],
        "cross_react":["aspirin", "ketorolac", "indomethacin"],
        "note": "COX-2 selective (celecoxib) may be tolerated — requires physician decision"
    },
    "contrast_dye": {
        "direct":     ["iodinated contrast"],
        "cross_react":[],
        "note": "Pre-medicate if contrast required. Flag for radiology."
    },
}

def check_allergy_crossreact(allergy_list: list[str],
                              proposed_drugs: list[str]) -> list[dict]:
    findings = []
    for allergy in allergy_list:
        rule = ALLERGY_CROSS_REACTIVITY.get(allergy.lower(), {})
        for drug in proposed_drugs:
            if drug.lower() in rule.get("direct", []):
                findings.append({
                    "type": "ALLERGY_DIRECT",
                    "severity": "CRITICAL",
                    "message": f"{drug} is a direct allergen (patient: {allergy} allergy)",
                })
            elif drug.lower() in rule.get("cross_react", []):
                findings.append({
                    "type": "ALLERGY_CROSS_REACT",
                    "severity": "HIGH",
                    "message": f"{drug} cross-reacts with {allergy}. {rule['note']}",
                })
    return findings
```

**What participants implement:**
- The full cross-reactivity map (penicillin, sulfonamide, NSAID, quinolone, cephalosporin families)
- Severity stratification (anaphylaxis history → CRITICAL, rash history → HIGH)
- Allergy phenotype encoding in episodic memory

---

#### Check 2 — Renal Function Threshold per Drug

Every drug that is renally cleared must be checked against the patient's **computed CrCl**, not just a generic flag:

```python
# Cockcroft-Gault formula
def compute_crcl(creatinine: float, age: int, weight_kg: float, sex: str) -> float:
    crcl = ((140 - age) * weight_kg) / (72 * creatinine)
    if sex.lower() == "female":
        crcl *= 0.85
    return round(crcl, 1)

# Drug-specific renal thresholds — participants must expand this table
RENAL_THRESHOLDS = {
    "methotrexate": {
        "contraindicated_below": 30,
        "caution_below": 60,
        "mechanism": "Renally cleared — accumulates causing mucositis, bone marrow suppression",
        "guideline": "ACR 2021 GL002",
        "alternative": "Hydroxychloroquine 200mg BD (renally safe DMARD)"
    },
    "metformin": {
        "contraindicated_below": 30,
        "caution_below": 45,
        "mechanism": "Lactic acidosis risk due to accumulation",
        "guideline": "ADA 2023",
        "alternative": "SGLT2i (if eGFR ≥ 20) or DPP-4 inhibitor (dose-adjust for CrCl)"
    },
    "apixaban": {
        "contraindicated_below": 15,
        "dose_reduce_below": 30,
        "mechanism": "27% renal excretion — higher bleeding risk at low CrCl",
        "guideline": "ESC 2020 AF Guidelines",
        "dose_reduction_rule": "Apixaban 2.5mg BD if ≥2 of: age≥80, weight≤60kg, Cr≥1.5"
    },
    "rivaroxaban": {
        "contraindicated_below": 15,
        "caution_below": 50,
        "mechanism": "33% renal excretion",
        "guideline": "ESC 2020"
    },
    "dabigatran": {
        "contraindicated_below": 30,
        "mechanism": "80% renal excretion — highest renal dependence of DOACs",
        "guideline": "ESC 2020"
    },
    "enoxaparin": {
        "dose_reduce_below": 30,
        "mechanism": "Accumulates — anti-Xa monitoring required or switch to UFH",
        "guideline": "CHEST 2021",
        "alternative": "Unfractionated heparin (not renally cleared)"
    },
    "gabapentin": {
        "dose_reduce_below": 60,
        "contraindicated_below": None,
        "mechanism": "100% renal excretion — sedation and fall risk at normal doses",
        "guideline": "Prescribers' Digital Reference",
    },
    "trimethoprim-sulfamethoxazole": {
        "contraindicated_below": 15,
        "caution_below": 30,
        "mechanism": "Raises creatinine by blocking tubular secretion (spurious AKI)",
        "guideline": "KDIGO CKD 2012"
    },
    "nsaids": {
        "contraindicated_below": 30,
        "caution_below": 60,
        "mechanism": "Inhibit prostaglandin-mediated afferent arteriolar dilation → AKI risk",
        "guideline": "KDIGO AKI 2012",
        "alternative": "Acetaminophen for analgesia"
    },
    "allopurinol": {
        "dose_reduce_below": 60,
        "mechanism": "Oxypurinol accumulates → allopurinol hypersensitivity syndrome risk",
        "guideline": "ACR Gout Guidelines 2020"
    },
}

def check_renal_thresholds(proposed_drugs: list[str],
                            crcl: float,
                            creatinine_trend: dict) -> list[dict]:
    findings = []
    # Extra conservatism if CrCl is trending downward
    safety_margin = 1.2 if creatinine_trend.get("direction") == "rising" else 1.0

    for drug in proposed_drugs:
        spec = RENAL_THRESHOLDS.get(drug.lower())
        if not spec:
            continue
        effective_threshold = (spec.get("contraindicated_below") or 0) * safety_margin
        if crcl < effective_threshold:
            findings.append({
                "type": "RENAL_CONTRAINDICATION",
                "severity": "CRITICAL",
                "drug": drug,
                "crcl": crcl,
                "threshold": spec["contraindicated_below"],
                "message": f"{drug} CONTRAINDICATED: CrCl {crcl} < {spec['contraindicated_below']} mL/min. {spec['mechanism']}",
                "guideline": spec.get("guideline"),
                "alternative": spec.get("alternative"),
            })
        elif crcl < (spec.get("caution_below") or 0):
            findings.append({
                "type": "RENAL_CAUTION",
                "severity": "HIGH",
                "drug": drug,
                "message": f"{drug}: CrCl {crcl} — use with caution, consider dose reduction",
            })
    return findings
```

**What participants implement:**
- Expand `RENAL_THRESHOLDS` to cover at minimum 15 drugs
- Implement CKD-EPI as alternative to Cockcroft-Gault (more accurate for eGFR estimation)
- Apply the "trending downward" safety margin (if creatinine rising, use a 20% stricter threshold)
- Suggest dose adjustments, not just flags (e.g., "reduce gabapentin from 300mg TID to 100mg TID")

---

#### Check 3 — QTc Prolongation Risk (CredibleMeds Integration)

```python
# CredibleMeds risk categories
QTCMED_RISK = {
    # Known Risk — sufficient evidence of QTc prolongation AND TdP
    "amiodarone":     "KNOWN",
    "sotalol":        "KNOWN",
    "haloperidol":    "KNOWN",
    "methadone":      "KNOWN",
    "azithromycin":   "KNOWN",
    "ciprofloxacin":  "KNOWN",
    "ondansetron":    "KNOWN",
    # Conditional Risk — TdP only under certain conditions
    "amitriptyline":  "CONDITIONAL",
    "quetiapine":     "CONDITIONAL",
    "venlafaxine":    "CONDITIONAL",
    # Possible Risk — some evidence but not established
    "sertraline":     "POSSIBLE",
    "escitalopram":   "CONDITIONAL",  # dose-dependent — >20mg is KNOWN
}

QTC_THRESHOLDS = {
    "alarm":    500,   # Stop all QT-prolonging drugs immediately
    "caution":  470,   # Male baseline; avoid KNOWN risk drugs
    "female":   480,   # Female baseline (longer normal QTc)
}

def check_qtc_risk(proposed_drugs: list[str],
                   current_qtc: float,
                   sex: str,
                   current_meds: list[str]) -> list[dict]:
    findings = []
    threshold = QTC_THRESHOLDS["female"] if sex == "F" else QTC_THRESHOLDS["caution"]
    qtc_drugs_on_board = [m for m in current_meds if QTCMED_RISK.get(m.lower())]

    for drug in proposed_drugs:
        risk = QTCMED_RISK.get(drug.lower())
        if not risk:
            continue
        # Adding a QT drug to a patient already on another QT drug = additive risk
        if current_qtc >= QTC_THRESHOLDS["alarm"]:
            findings.append({"type": "QTC_ABSOLUTE", "severity": "CRITICAL",
                              "message": f"QTc {current_qtc}ms (≥500). Adding {drug} ({risk} risk) is contraindicated."})
        elif current_qtc >= threshold or qtc_drugs_on_board:
            existing = ", ".join(qtc_drugs_on_board) if qtc_drugs_on_board else "elevated baseline QTc"
            findings.append({"type": "QTC_ADDITIVE", "severity": "HIGH",
                              "message": f"{drug} ({risk} risk) + {existing} → additive QTc prolongation risk. ECG monitoring required."})
    return findings
```

**What participants implement:**
- Full CredibleMeds risk table (at minimum 20 drugs)
- Additive risk logic (patient already on QT drug + new QT drug = higher severity)
- Sex-specific thresholds
- ECG monitoring recommendation as part of corrected treatment

---

#### Check 4 — Drug-Drug Interaction (DDI) Graph

```python
# Build DDI graph from episodic medication list
DDI_DATABASE = {
    ("warfarin", "aspirin"):       {"severity": "MAJOR",    "effect": "Additive bleeding risk"},
    ("warfarin", "nsaids"):        {"severity": "MAJOR",    "effect": "GI bleed risk +3x"},
    ("warfarin", "metronidazole"): {"severity": "MAJOR",    "effect": "CYP2C9 inhibition → INR spike"},
    ("ssri",     "tramadol"):      {"severity": "MAJOR",    "effect": "Serotonin syndrome risk"},
    ("ace_inhibitor", "potassium"):{"severity": "MAJOR",    "effect": "Hyperkalemia — ACE + K supplementation"},
    ("metformin", "contrast_dye"): {"severity": "MAJOR",    "effect": "Lactic acidosis risk — hold 48h post-contrast"},
    ("lithium",  "nsaids"):        {"severity": "MAJOR",    "effect": "NSAIDs reduce renal Li clearance → toxicity"},
    ("ssri",     "maoi"):          {"severity": "CONTRAIND","effect": "Serotonin syndrome — 14-day washout required"},
    ("statins",  "amiodarone"):    {"severity": "MAJOR",    "effect": "CYP3A4 inhibition → statin myopathy risk"},
    ("digoxin",  "amiodarone"):    {"severity": "MAJOR",    "effect": "Amiodarone increases digoxin levels 2x"},
}

def check_drug_interactions(proposed_drugs: list[str],
                             current_meds: list[str]) -> list[dict]:
    all_drugs = set(proposed_drugs + current_meds)
    findings = []
    for (d1, d2), info in DDI_DATABASE.items():
        if d1 in all_drugs and d2 in all_drugs:
            findings.append({
                "type": "DDI",
                "severity": "CRITICAL" if info["severity"] == "CONTRAIND" else info["severity"],
                "pair": f"{d1} + {d2}",
                "message": info["effect"],
            })
    return findings
```

**What participants implement:**
- DDI database with at minimum 25 drug pairs
- Drug class normalisation (any SSRI should match the "ssri" key, not just sertraline)
- CYP450 pathway-based interaction logic (which drugs inhibit CYP3A4, CYP2C9, CYP2D6)

---

#### Check 5 — Beers Criteria (Elderly Safety)

```python
BEERS_CRITERIA = [
    {"drug_class": "benzodiazepines",
     "risk": "Fall risk, cognitive impairment, paradoxical agitation",
     "alternative": "CBT-I for insomnia; non-pharmacologic anxiety management"},
    {"drug_class": "anticholinergics",  # diphenhydramine, oxybutynin, tricyclics
     "risk": "Confusion, urinary retention, constipation, fall risk",
     "alternative": "Mirabegron for OAB; SSRIs for depression"},
    {"drug_class": "nsaids",
     "risk": "GI bleed, fluid retention, AKI — risk increases with age",
     "alternative": "Topical NSAIDs, acetaminophen"},
    {"drug_class": "first_gen_antihistamines",  # diphenhydramine, hydroxyzine
     "risk": "Sedation, cognitive impairment — crosses BBB more in elderly",
     "alternative": "Loratadine, cetirizine (low CNS penetration)"},
    {"drug_class": "muscle_relaxants",   # cyclobenzaprine, methocarbamol
     "risk": "Sedation, anticholinergic effects, fall risk",
     "alternative": "Physical therapy; low-dose NSAIDs short-term"},
]

def check_beers_criteria(proposed_drugs: list[str], age: int) -> list[dict]:
    if age < 65:
        return []
    findings = []
    for drug in proposed_drugs:
        for criteria in BEERS_CRITERIA:
            if drug_matches_class(drug, criteria["drug_class"]):
                findings.append({
                    "type": "BEERS_CRITERIA",
                    "severity": "HIGH",
                    "message": f"{drug} — AGS Beers Criteria: {criteria['risk']}",
                    "alternative": criteria["alternative"],
                })
    return findings
```

---

#### Check 6 — Prescribing Cascade Detection

A prescribing cascade occurs when a drug causes a side effect that is misidentified as a new disease, leading to a second drug:

```python
PRESCRIBING_CASCADES = [
    {
        "culprit_drug":    "nsaids",
        "likely_sideeffect": "hypertension",
        "potential_cascade": "antihypertensive",
        "message": "New antihypertensive in a patient on NSAIDs — consider NSAID-induced HTN first"
    },
    {
        "culprit_drug":    "ace_inhibitor",
        "likely_sideeffect": "cough",
        "potential_cascade": "antitussive",
        "message": "Antitussive added — rule out ACE inhibitor cough (switch to ARB?)"
    },
    {
        "culprit_drug":    "calcium_channel_blocker",
        "likely_sideeffect": "peripheral_edema",
        "potential_cascade": "diuretic",
        "message": "Diuretic added to CCB patient — may be CCB-induced pedal oedema"
    },
    {
        "culprit_drug":    "antipsychotic",
        "likely_sideeffect": "extrapyramidal_symptoms",
        "potential_cascade": "anticholinergic",
        "message": "Anticholinergic added — may be treating antipsychotic-induced EPS"
    },
]

def detect_prescribing_cascade(proposed_drugs: list[str],
                                current_meds: list[str],
                                chief_complaint: str) -> list[dict]:
    findings = []
    for cascade in PRESCRIBING_CASCADES:
        culprit_on_board = any(drug_matches_class(m, cascade["culprit_drug"])
                               for m in current_meds)
        cascade_being_added = any(drug_matches_class(d, cascade["potential_cascade"])
                                  for d in proposed_drugs)
        if culprit_on_board and cascade_being_added:
            findings.append({
                "type": "PRESCRIBING_CASCADE",
                "severity": "MODERATE",
                "message": cascade["message"],
            })
    return findings
```

---

### Innovative Critique Extensions — Web Search & External API Integration

The following extensions push the critique agent from a rule-based safety checker into a **live intelligence agent** that knows what happened last week in pharmacovigilance.

#### Extension A — OpenFDA Real-Time Drug Safety Alerts

```python
import httpx

async def query_openfda_adverse_events(drug_name: str,
                                        top_n: int = 3) -> list[dict]:
    """
    Queries FDA FAERS (Adverse Event Reporting System) for the most common
    serious adverse events reported for this drug in the last 12 months.
    """
    url = "https://api.fda.gov/drug/event.json"
    params = {
        "search": f'patient.drug.medicinalproduct:"{drug_name}" AND serious:1',
        "count":  "patient.reaction.reactionmeddrapt.exact",
        "limit":  top_n,
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params, timeout=5.0)
    if resp.status_code != 200:
        return []
    data = resp.json()
    return [{"reaction": r["term"], "count": r["count"]}
            for r in data.get("results", [])]

async def query_openfda_recalls(drug_name: str) -> list[dict]:
    """Check for active FDA recalls on the proposed drug."""
    url = "https://api.fda.gov/drug/enforcement.json"
    params = {
        "search": f'product_description:"{drug_name}" AND status:"Ongoing"',
        "limit": 3,
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params, timeout=5.0)
    if resp.status_code != 200:
        return []
    return resp.json().get("results", [])
```

> **Participants implement:** Wrap both functions, parse and rank findings by severity, inject into critique findings if top adverse event is serious (anaphylaxis, agranulocytosis, hepatotoxicity). Add graceful fallback when API is unavailable.

---

#### Extension B — Serper / Web Search for Guideline Currency

Medical guidelines update. The critique agent's embedded guideline KB is frozen at training time. Use Serper (or any search API) to retrieve the *most recent* version of a relevant guideline:

```python
import httpx, os

async def search_latest_guideline(condition: str,
                                   drug: str,
                                   serper_api_key: str = None) -> dict:
    """
    Search for the most recent clinical guideline regarding
    the proposed drug for this condition.
    """
    key = serper_api_key or os.environ.get("SERPER_API_KEY")
    if not key:
        return {"status": "unavailable", "results": []}

    query = f"clinical guideline {condition} {drug} contraindication site:nih.gov OR site:nice.org.uk OR site:acc.org"
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": key, "Content-Type": "application/json"},
            json={"q": query, "num": 3},
            timeout=8.0,
        )
    results = resp.json().get("organic", [])
    return {
        "status": "ok",
        "query": query,
        "results": [{"title": r["title"], "link": r["link"], "snippet": r["snippet"]}
                    for r in results],
    }
```

> **Use case:** Critique node proposes hydroxychloroquine as alternative. Before accepting, search: *"hydroxychloroquine renal dosing guideline 2024"* to verify the alternative is also safe. Flag if web results suggest recent safety updates.

---

#### Extension C — PubMed Evidence Quality Check

```python
async def pubmed_evidence_lookup(drug: str,
                                  condition: str,
                                  years: int = 3) -> dict:
    """
    Query PubMed E-utilities for recent RCTs on this drug + condition.
    Returns evidence grade estimate from publication count and study types.
    """
    from datetime import date
    min_date = (date.today().year - years)
    query = f"{drug}[tiab] AND {condition}[tiab] AND (randomized controlled trial[pt])"
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed", "term": query,
        "retmode": "json", "retmax": 5,
        "mindate": min_date, "maxdate": date.today().year,
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params, timeout=8.0)
    data = resp.json()
    count = int(data.get("esearchresult", {}).get("count", 0))
    return {
        "rct_count_last_3y": count,
        "evidence_strength": "HIGH" if count >= 3 else "MODERATE" if count >= 1 else "LOW",
        "note": f"{count} RCT(s) in last {years} years for {drug} in {condition}",
    }
```

> **Use case:** Critique agent checks: if evidence is LOW for the proposed treatment, downgrade confidence score and flag for HITL — even if no explicit contraindication exists.

---

#### Extension D — Pharmacogenomics Check (Stretch Goal)

```python
PHARMACOGENOMICS_FLAGS = {
    "clopidogrel": {
        "gene":       "CYP2C19",
        "poor_metabolizer_risk": "Reduced antiplatelet effect — 30% higher MACE risk",
        "testing_note": "If CYP2C19 *2/*2 or *2/*3: switch to prasugrel or ticagrelor",
    },
    "codeine": {
        "gene":       "CYP2D6",
        "ultra_metabolizer_risk": "Rapid morphine conversion → respiratory depression risk",
        "poor_metabolizer_risk": "No analgesia — prodrug not converted",
    },
    "warfarin": {
        "genes":      ["CYP2C9", "VKORC1"],
        "note": "VKORC1 A/A → 30-40% lower warfarin dose required; CYP2C9 *2/*3 → slow metaboliser",
    },
    "simvastatin": {
        "gene":       "SLCO1B1",
        "poor_metabolizer_risk": "SLCO1B1 *5 allele → 17x higher myopathy risk at 80mg",
        "testing_note": "If SLCO1B1 *5: cap at 20mg or switch to rosuvastatin/pravastatin",
    },
}

def check_pharmacogenomics(proposed_drugs: list[str],
                            genomic_data: dict | None) -> list[dict]:
    findings = []
    for drug in proposed_drugs:
        flag = PHARMACOGENOMICS_FLAGS.get(drug.lower())
        if not flag:
            continue
        if genomic_data:
            gene = flag.get("gene") or flag.get("genes", [None])[0]
            genotype = genomic_data.get(gene)
            if genotype and is_poor_metabolizer(gene, genotype):
                findings.append({"type": "PHARMACOGENOMICS", "severity": "HIGH",
                                  "message": flag["poor_metabolizer_risk"]})
        else:
            findings.append({"type": "PHARMACOGENOMICS_UNTESTED", "severity": "INFO",
                              "message": f"Genomic data unavailable for {drug}. "
                                         f"Consider {flag.get('gene')} testing if treatment fails."})
    return findings
```

---

## What Participants Must Implement

This is the exhaustive list of implementation tasks. Items marked **[CORE]** are required for baseline credit. Items marked **[ADVANCED]** are stretch goals that demonstrate deeper understanding.

### Phase 1 — Memory Infrastructure

- **[CORE]** Implement `EpisodicStore` class with `save_visit()` and `get_prior_visits(patient_id)` backed by JSON files
- **[CORE]** Implement `get_lab_trend(patient_id, lab_name)` returning values + timestamps sorted by date
- **[CORE]** Implement `memory_retrieval_node` that loads prior visits and writes `lab_trends` to working state
- **[ADVANCED]** Implement vector embedding store for episodic memory (use FAISS or Chroma) to allow semantic retrieval of *similar* prior cases, not just the same patient

### Phase 2 — Trend Detection

- **[CORE]** Implement `simple_delta()` and `domain_rule_flags()` using the KDIGO, Sepsis-3, and AHA rules
- **[CORE]** Implement `linear_regression_trend()` for labs with ≥3 data points
- **[ADVANCED]** Implement `cusum_detect()` with configurable `k` and `h` parameters
- **[ADVANCED]** Integrate LLM-mediated trend narrative into the clinical note (not for safety gates)
- **[ADVANCED]** Implement the hybrid strategy router that selects the appropriate algorithm based on data point count

### Phase 3 — Semantic Memory

- **[CORE]** Implement `semantic_search_node` with keyword overlap scoring against `guidelines.json`
- **[ADVANCED]** Upgrade to embedding-based retrieval (use `text-embedding-ada-002` or `sentence-transformers`) so that semantically related but keyword-mismatched guidelines are retrieved
- **[ADVANCED]** Implement a `guideline_freshness_check()` that flags if the retrieved guideline is more than 2 years old and triggers a Serper web search for the latest version

### Phase 4 — Critique Agent

- **[CORE]** Implement `check_allergy_crossreact()` with direct + cross-reactive allergen mapping for penicillin, sulfonamide, and NSAID families
- **[CORE]** Implement `compute_crcl()` using Cockcroft-Gault and `check_renal_thresholds()` for at least 10 drugs from the table above
- **[CORE]** Implement `check_qtc_risk()` with CredibleMeds Known/Conditional/Possible risk categories
- **[CORE]** Implement `check_drug_interactions()` with at least 20 drug-drug pairs
- **[CORE]** Implement `check_beers_criteria()` for patients aged ≥65
- **[ADVANCED]** Implement `detect_prescribing_cascade()` for at least 4 cascade patterns
- **[ADVANCED]** Implement `check_pharmacogenomics()` and handle the case where genomic data is unavailable (flag for testing)
- **[ADVANCED]** Implement `query_openfda_adverse_events()` and surface findings in critique output
- **[ADVANCED]** Implement `search_latest_guideline()` using Serper and inject results as additional context for the critique LLM prompt
- **[ADVANCED]** Implement CKD-EPI as an alternative CrCl estimator and compare both; use the more conservative value

### Phase 5 — HITL Gate

- **[CORE]** Implement the confidence gate with the three triggers: `conf < 0.80`, `CRITICAL critique finding`, `psychiatric_risk`
- **[CORE]** Implement `hitl_node` that presents the structured summary (diagnosis + critique findings + proposed treatment) and captures physician action (APPROVE / EDIT / REJECT) with a free-text reason
- **[CORE]** Write HITL action to `state["hitl_log"]` with timestamp, action, reason, and physician ID
- **[ADVANCED]** Implement a "second opinion" trigger: if the physician EDITS a CRITICAL finding override, automatically elevate to require a second physician sign-off

### Phase 6 — Language Guardrails

- **[CORE]** Implement the prohibited language validator in `note_generation_node` — scan the generated note for any of the prohibited phrases and raise a `GuardrailViolation` exception
- **[CORE]** Enforce non-definitive language in `differential_diagnosis_node` system prompt
- **[ADVANCED]** Integrate NeMo Guardrails as an outer wrapper on the diagnosis LLM call

### Phase 7 — Evaluation

- **[CORE]** Implement the 6-stage progressive evaluation: baseline → +episodic → +semantic → +critique → +HITL → full system
- **[CORE]** Compute: accuracy, self-correction rate, contraindication miss rate, HITL recall, and latency per stage
- **[CORE]** Generate the three evaluation charts (progressive improvement, HITL confidence analysis, capability radar)
- **[ADVANCED]** Add a **false confidence analysis**: for cases where the agent was ≥80% confident, what fraction were incorrect? Show how the critique node eliminates false confidence

---

## Quick Start

### Step 1: Install Dependencies

```bash
pip install langgraph langchain-openai langchain-core
pip install langchain-chroma chromadb
pip install numpy pandas matplotlib  # for evaluation charts
pip install httpx                    # for OpenFDA / Serper / PubMed integrations
pip install faiss-cpu sentence-transformers  # for ADVANCED embedding retrieval
```

### Step 2: Set up Environment

**Option A: Azure OpenAI (production path)**
```bash
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o"
export AZURE_OPENAI_API_VERSION="2024-08-01-preview"
```

**Option B: OpenAI (development)**
```bash
export OPENAI_API_KEY="sk-..."
```

**Option C: MockLLM (no API key — for training)**
```
# No config needed. MockLLM activates automatically when no API key is set.
# Returns deterministic, clinically realistic responses that match the dataset.
```

**Option D: External API keys (for ADVANCED critique extensions)**
```bash
export SERPER_API_KEY="..."    # from serper.dev — web search for guideline currency
# OpenFDA and PubMed APIs are public and do not require a key
```

### Step 3: Generate Data

```bash
cd data/
python generate_dataset.py
```

### Step 3.5: Build Chroma Semantic Index

```bash
cd ..
python -m src.build_chroma_index         # idempotent refresh
python -m src.build_chroma_index --rebuild  # full rebuild
```

### Step 4: Run the Demo

```bash
cd demo/
python demo.py --auto          # Non-interactive (auto-approve HITL)
python demo.py                 # Interactive (terminal HITL prompts)
python demo.py --patient arjun # Single patient journey
python demo.py --limitations   # Limitation disclosures only
```

### Step 5: Run Evaluation

```bash
cd src/
python evaluation_runner.py    # Prints metrics table + saves CSV

cd evaluation/
python eval_dashboard.py       # Generate all 3 charts
```

### Step 6: Build and Test the LangGraph Agent

```bash
cd src/
python agent.py                # Builds and verifies graph structure
```

### Step 7: Run API + React UI

Terminal A (backend):
```bash
uvicorn src.api_server:app --reload --port 8000
```

Terminal B (frontend):
```bash
cd ui/
npm install
npm run dev
```

Open the UI at `http://localhost:5173`.

---

## Project Structure

```
clinicaldss/
├── data/
│   ├── generate_dataset.py       ← Run this first
│   ├── guidelines.json           ← 30 medical guidelines (Semantic Memory KB)
│   ├── patient_journeys.json     ← 2 demo patients × 3 visits each
│   └── eval_patients.json        ← 8 evaluation scenarios
│
├── src/
│   ├── agent.py                  ← LangGraph graph + all nodes + memory stores
│   ├── api_server.py             ← FastAPI endpoints for UI (signin, run visit, HITL, past visits)
│   ├── api_test.py               ← Azure Chat + Embedding connectivity smoke tests
│   ├── build_chroma_index.py     ← Chroma index builder for guideline embeddings
│   ├── trend_detection.py        ← [IMPLEMENT] All 5 trend algorithms + hybrid router
│   ├── critique_engine.py        ← [IMPLEMENT] Full battery of safety checks
│   ├── external_apis.py          ← [IMPLEMENT] OpenFDA / Serper / PubMed integrations
│   └── evaluation_runner.py      ← Progressive metrics simulation
│
├── evaluation/
│   ├── eval_dashboard.py         ← 3 evaluation charts
│   ├── 01_progressive_improvement.png
│   ├── 02_hitl_confidence_analysis.png
│   └── 03_capability_radar.png
│
├── demo/
│   └── demo.py                   ← Full 2-patient demo + limitations

├── ui/
│   ├── package.json              ← React + Vite app
│   ├── index.html
│   └── src/
│       ├── App.jsx               ← Home, sign in/out, add visit flow, HITL modal
│       ├── main.jsx
│       └── styles.css
│
├── configs/
│   ├── episodic_store/           ← Auto-created: per-patient JSON visit history
│   ├── system_prompts.md         ← All system prompts with guardrail annotations
│   └── azure_foundry_path_b.md  ← Path B: Azure AI Foundry Agents
│
└── README.md
```

---

## Injected Challenge Patterns

| Challenge | Patient | Visit | What Happens |
|-----------|---------|-------|--------------|
| `[MEMORY]` | Arjun | V1→V3 | Creatinine trend 1.8→2.1→2.3 — stateless agent misses AKI developing |
| `[CRITIQUE]` | Arjun | V3 | MTX proposed for RA — **Critique catches CrCl 26 < 30 absolute contraindication** |
| `[CRITIQUE]` | Priya | V1 | TMP-SMX prescribed — **Critique catches sulfonamide allergy** |
| `[CRITIQUE]` | Priya | V3 | Amitriptyline proposed — **Critique catches QTc 462ms cardiac risk** |
| `[HITL]` | Arjun | V2 | Confidence 0.71 → **HITL mandatory** — physician reviews HFrEF + AKI |
| `[HITL]` | Priya | V3 | Passive SI + PHQ-9=16 → **HITL mandatory** — psychiatric risk triggers physician review |
| `[TEMPORAL]` | Priya | V1→V2 | qSOFA 1→3 in 48h — sepsis progression missed without prior visit baseline |
| `[SEMANTIC]` | Arjun | V2 | SGLT2i for HFrEF+T2DM — AHA GL001 retrieved to support recommendation |
| `[CASCADE]` | Arjun | V2 | New antihypertensive after NSAID — **cascade detection flags NSAID-induced HTN** |
| `[BEERS]` | Priya | V3 | Amitriptyline (TCA) age 67 — **Beers Criteria: anticholinergic + fall risk** |
| `[DDI]` | Priya | V2 | Tramadol added while on sertraline — **DDI: serotonin syndrome risk** |

---

## The Five Graph Nodes (Detailed)

### `intake_node`
- Initialises the clinical state
- Starts audit trail with timestamps
- Validates required fields

### `memory_retrieval_node`
```python
# Key operation: temporal lab trend detection
for lab in ["Creatinine", "eGFR", "BNP", "QTc", "qSOFA"]:
    trend = episodic_store.get_lab_trend(patient_id, lab)
    if len(trend) > 1:
        state["working_notes"].append(f"[LabTrend] {lab}: ...")
```
Without this node: `CrCl = f(creatinine_today, age)` gives one value.
With this node: `CrCl = f(creatinine_V1=1.8, V2=2.1, V3=2.3, age=67)` exposes the trend.

### `semantic_search_node`
Retrieves relevant guidelines by keyword overlap with the current complaint + labs.
Loaded guidelines go into `state["semantic_context"]` and are injected into both the diagnosis and critique prompts.

### `differential_diagnosis_node`
Uses the **language guardrail prompt**:
- NEVER: "The patient HAS diabetes" → ALWAYS: "Findings are consistent with T2DM"
- NEVER: "Patient DEFINITELY has sepsis" → ALWAYS: "Clinical picture suggests sepsis"
- Returns JSON with `{differential, primary, probability, reasoning}`

### `critique_node` ← The star of the show
Takes the proposed diagnosis + treatment and actively attacks it:
1. Cross-references patient allergies (direct + cross-reactive) against proposed drugs
2. Checks renal function thresholds for every drug proposed (methotrexate, metformin, DOACs, gabapentin...)
3. Computes CrCl from creatinine + age using Cockcroft-Gault; applies stricter threshold if creatinine is trending upward
4. Checks QTc against CredibleMeds risk categories (Known / Conditional / Possible) + additive risk
5. Verifies drug interactions from episodic medication history using the DDI graph
6. Flags temporal lab trends (AKI by KDIGO: Cr rise ≥0.3 in 48h or ≥1.5× baseline in 7d)
7. Applies Beers Criteria for patients ≥65
8. Detects prescribing cascades from current medication list
9. [ADVANCED] Queries OpenFDA for active recalls and top adverse event reports
10. [ADVANCED] Uses Serper to verify guideline currency and retrieve latest updates

**Critique output:**
```json
{
  "findings": [
    {"type": "CONTRAINDICATION", "severity": "CRITICAL",
     "message": "Methotrexate CONTRAINDICATED: CrCl ~26 mL/min...",
     "source": "ACR 2021 GL002",
     "alternative": "Hydroxychloroquine 200mg BD"},
    {"type": "RENAL_CAUTION", "severity": "HIGH",
     "message": "Metformin: CrCl 26 — contraindicated below 30, stop immediately"},
    {"type": "AKI_TRAJECTORY", "severity": "CRITICAL",
     "message": "KDIGO AKI Stage 1: Cr 1.8→2.1→2.3 over 28d, slope +0.018/day"}
  ],
  "critique_passed": false,
  "corrected_treatment": "Hydroxychloroquine 200mg BD (renal-safe DMARD). Hold metformin.",
  "self_correction_occurred": true,
  "confidence_delta": -0.25
}
```

### HITL Gate — Decision Logic

```python
def confidence_gate_node(state) -> str:
    conf = state["highest_confidence"]
    critical = [f for f in state["critique_findings"] if f["severity"] == "CRITICAL"]
    psychiatric_risk = any(kw in symptoms for kw in ["suicidal", "self-harm"])

    if conf < 0.80 or critical or psychiatric_risk:
        return "hitl_node"    # Pause for physician review
    return "treatment_node"   # Proceed with high confidence
```

**HITL is always logged:** Every physician interaction (approve/edit/reject) is written to `state["hitl_log"]` with full audit trail.

---

## Evaluation Results

### Progressive Stage Metrics

| Stage | Accuracy | Self-Correction | Contraindication Miss | HITL Recall | Latency |
|-------|----------|-----------------|----------------------|-------------|---------|
| Stage 0 — Baseline | 12.5% | 0.0% | 87.5% | 0.0% | 0.81s |
| Stage 1 — + Episodic Memory | 38.0% | 0.0% | 62.0% | 0.0% | 1.22s |
| Stage 2 — + Semantic Memory | 54.0% | 0.0% | 46.0% | 0.0% | 1.54s |
| Stage 3 — + Critique | 87.5% | 75.0% | 5.0% | 0.0% | 2.10s |
| Stage 4 — + HITL | 97.5% | 75.0% | 0.0% | 97.0% | 2.65s |
| Stage 5 — Full System | 98.8% | 87.5% | 0.0% | 98.0% | 3.02s |

**Key insight — False Confidence Rate:** Stage 0 has a 50% false confidence rate (50% of cases where the agent was ≥80% confident were actually wrong). Stage 5 drives this to 0%.

**Clinical significance:** A patient misdiagnosed at 85% confidence would NOT trigger HITL in a system without the Critique Node. The Critique is what catches the "confident wrong" cases.

---

## Observability & Audit Trail

Every visit generates a full `audit_log`:
```json
[
  {"timestamp": "2024-11-19T10:32:01", "node": "memory_retrieval", "prior_visits": 1},
  {"timestamp": "2024-11-19T10:32:02", "node": "semantic_search", "guidelines": ["GL001","GL006"]},
  {"timestamp": "2024-11-19T10:32:04", "node": "critique", "findings": 3, "self_correction": true},
  {"timestamp": "2024-11-19T10:32:06", "node": "hitl", "action": "EDITED", "reason": "confidence 71%"},
  {"timestamp": "2024-11-19T10:32:08", "node": "note_generation", "episodic_saved": true}
]
```

This log supports:
- **Regulatory audit**: Why was this treatment chosen? What did HITL review?
- **Liability traceability**: Was the physician notified of the critique findings?
- **Quality improvement**: Which diagnoses most often trigger self-correction?

---

## Language Guardrails (Built into System Prompts)

The system enforces **non-definitive language** throughout:

| Prohibited | Required |
|------------|----------|
| "The patient HAS diabetes" | "Findings are consistent with T2DM" |
| "Definitively diagnosed with..." | "Clinical picture suggests..." |
| "This confirms..." | "These results are consistent with..." |
| "The patient WILL develop..." | "Risk factors indicate possible..." |

These guardrails are embedded in the system prompt for `differential_diagnosis_node` and validated in `note_generation_node`.

---

## NeMo Guardrails Integration

For production deployment, add NeMo Guardrails as an outer wrapper:

```python
from nemoguardrails import RailsConfig, LLMRails

config = RailsConfig.from_path("configs/guardrails/")
rails = LLMRails(config)

# Wrap the diagnosis response
safe_response = await rails.generate_async(
    messages=[{"role": "user", "content": diagnosis_prompt}]
)
```

The colang config defines:
- `refuse definitive diagnosis` — catches "patient has" phrasing
- `refuse without physician review` — catches high-stakes recommendations without HITL marker
- `add clinical disclaimer` — appended to all generated notes

---

## Path B: Azure AI Foundry Agents

See `configs/azure_foundry_path_b.md` for the equivalent implementation using Azure AI Foundry's Agent framework:

```python
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import Agent, ThreadMessage, MessageRole

client = AIProjectClient(endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"],
                         credential=DefaultAzureCredential())

# Create memory-enabled clinical agent
agent = client.agents.create_agent(
    model="gpt-4o",
    name="ClinicalDSS-Foundry",
    instructions=CLINICAL_SYSTEM_PROMPT,
    tools=[episodic_memory_tool, semantic_search_tool, critique_tool]
)
```

Key differences from Path A (LangGraph):
- State managed by Azure's Thread object (vs explicit TypedDict)
- HITL via human_input_required tool call (vs conditional edge)
- Observability via Azure Monitor (vs LangSmith)


## Demo Notes

Run the Arjun Mehta demo. When Visit 3 proposes methotrexate and the Critique Node fires with:

```
🔴 [CRITICAL] CONTRAINDICATION: Methotrexate proposed.
   CrCl ~26 mL/min (from creatinine 2.3, age 67).
   Absolute contraindication: CrCl < 30 mL/min.
   This patient could have developed methotrexate toxicity leading to bone marrow failure.

🔴 [CRITICAL] AKI TRAJECTORY: Creatinine 1.8 → 2.1 → 2.3 over 28 days.
   Linear slope: +0.018 mg/dL/day. KDIGO criteria met.
   Renal function is WORSENING — apply 20% stricter drug thresholds.
```

Ask the room: *"What would have happened in a stateless GPT-4 chat session?"*

Answer: It wouldn't have the creatinine from Visit 1 (1.8) or Visit 2 (2.1). It might have computed CrCl from 2.3 and noticed the borderline — or it might not. Even if it did, it wouldn't know the trend is *worsening*, which changes the clinical urgency. And it certainly wouldn't apply the 20% safety margin that the trending creatinine warrants.

### On the HITL Design

HITL in this system is not a nicety — it's architecturally mandatory for:
1. Confidence < 0.80 (clinical uncertainty)
2. Any CRITICAL critique finding (contraindication detected)
3. Psychiatric risk (suicidal ideation) — always escalate

Emphasise that the audit log of HITL overrides is what makes this system regulatorily viable. Every physician decision is timestamped and traceable.

---

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Azure AI Foundry Agents](https://learn.microsoft.com/azure/ai-studio/concepts/agents)
- [AHA 2022 Heart Failure Guideline](https://www.ahajournals.org/doi/10.1161/CIR.0000000000001063)
- [ACR Methotrexate Guidelines](https://www.rheumatology.org/Practice-Quality/Clinical-Support/Clinical-Practice-Guidelines)
- [KDIGO AKI Guideline](https://kdigo.org/guidelines/acute-kidney-injury/)
- [NICE Depression Guideline NG222](https://www.nice.org.uk/guidance/ng222)
- [Surviving Sepsis Campaign 2021](https://www.sccm.org/SurvivingSepsisCampaign/Guidelines)
- [CredibleMeds QTDrugs List](https://crediblemeds.org/pdftemp/pdf/CombinedRiskList.pdf)
- [AGS Beers Criteria 2023](https://agsjournals.onlinelibrary.wiley.com/doi/10.1111/jgs.18372)
- [OpenFDA API](https://open.fda.gov/apis/)
- [PubMed E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25500/)
- [Serper API](https://serper.dev/)

---

*CitiusTech Gen AI & Agentic AI Training Program —  Project 3 of 5*
