"""
ClinicalDSS — Synthetic Patient Dataset Generator
===================================================
Generates multi-visit patient histories deliberately engineered to expose
the value of:
  [MEMORY]    Stateless agent forgets prior visits → wrong differential
  [CRITIQUE]  First-pass diagnosis misses contradicting evidence → Critique catches it
  [HITL]      Low-confidence cases (< 0.80) that MUST surface to physician
  [SEMANTIC]  Guideline lookup changes treatment recommendation
  [TEMPORAL]  Lab trend matters (improving vs worsening) — one visit is misleading

Two detailed patient journeys (for demo) + 8 supporting patient profiles for evaluation.
"""

import json
import random
from pathlib import Path

random.seed(42)
OUT = Path(__file__).parent


# ─────────────────────────────────────────────────────────────────────────────
# MEDICAL GUIDELINES (Semantic Memory KB)
# ─────────────────────────────────────────────────────────────────────────────

GUIDELINES = [
    {
        "id": "GL001",
        "title": "AHA/ACC 2022 Heart Failure Guideline — SGLT2i in HFrEF",
        "body": (
            "Sodium-glucose cotransporter-2 inhibitors (SGLT2i) — dapagliflozin and empagliflozin — "
            "are recommended (Class I, LOE A) for patients with symptomatic chronic heart failure "
            "with reduced ejection fraction (HFrEF, LVEF ≤40%) to reduce the risk of HF hospitalization "
            "and cardiovascular death, irrespective of diabetes status. "
            "CONTRAINDICATION: eGFR < 20 mL/min/1.73m². "
            "Caution: eGFR 20–45 — benefit present but lower. "
            "Do NOT initiate if patient is acutely decompensated."
        ),
        "category": "cardiology",
        "keywords": ["heart failure", "HFrEF", "SGLT2", "dapagliflozin", "empagliflozin", "EF"],
        "challenge_tags": ["SEMANTIC", "CONTRAINDICATION"]
    },
    {
        "id": "GL002",
        "title": "ACR 2021 — Methotrexate Contraindications in RA",
        "body": (
            "Methotrexate (MTX) is first-line DMARD for rheumatoid arthritis. "
            "ABSOLUTE CONTRAINDICATIONS: pregnancy (Category X), breastfeeding, "
            "severe renal impairment (CrCl < 30 mL/min), hepatic cirrhosis, "
            "active infection. "
            "RELATIVE CONTRAINDICATIONS: significant alcohol use (>1 drink/day), "
            "pulmonary disease (ILD risk), obesity (hepatotoxicity risk). "
            "Monitor: CBC, LFTs, creatinine every 4–8 weeks for first 6 months. "
            "Folate supplementation (1 mg/day) reduces mucosal and hematologic toxicity."
        ),
        "category": "rheumatology",
        "keywords": ["methotrexate", "rheumatoid arthritis", "RA", "DMARD", "kidney", "pregnancy"],
        "challenge_tags": ["CRITIQUE", "CONTRAINDICATION"]
    },
    {
        "id": "GL003",
        "title": "ADA 2024 — Type 2 Diabetes: Cardiorenal Protection Algorithm",
        "body": (
            "For T2DM patients with established atherosclerotic cardiovascular disease (ASCVD), "
            "CKD (eGFR 20–60 or albuminuria), or HF: add GLP-1 RA or SGLT2i regardless of HbA1c. "
            "GLP-1 RA preferred if ASCVD predominates (weight loss benefit). "
            "SGLT2i preferred if CKD or HF predominates (renal/cardiac protection). "
            "AVOID metformin if eGFR < 30. HOLD metformin if eGFR 30–45 (use caution). "
            "Insulin required if HbA1c > 10% or symptoms of hyperglycemia."
        ),
        "category": "endocrinology",
        "keywords": ["diabetes", "T2DM", "CKD", "SGLT2", "GLP-1", "HbA1c", "metformin"],
        "challenge_tags": ["SEMANTIC", "MEMORY"]
    },
    {
        "id": "GL004",
        "title": "GOLD 2023 — COPD Exacerbation Management",
        "body": (
            "Acute COPD exacerbation: SABA + SAMA bronchodilators, systemic corticosteroids "
            "(prednisone 40 mg × 5 days), antibiotics if purulent sputum/fever/CRP >10. "
            "Oxygen target: SpO2 88–92% (avoid hyperoxygenation). "
            "CONTRAINDICATION: beta-blockers are RELATIVELY contraindicated in severe COPD "
            "(may worsen bronchospasm); cardioselective beta-1 selective agents preferred if needed "
            "for cardiac indication. "
            "Long-term: LABA+LAMA combination (dual bronchodilation) preferred over LABA alone. "
            "ICS indicated only if eosinophils >300/μL or ≥2 exacerbations/year."
        ),
        "category": "pulmonology",
        "keywords": ["COPD", "exacerbation", "bronchodilator", "corticosteroid", "beta-blocker", "ICS"],
        "challenge_tags": ["CRITIQUE", "CONTRAINDICATION"]
    },
    {
        "id": "GL005",
        "title": "ESC 2023 — Pulmonary Embolism: Diagnosis and Risk Stratification",
        "body": (
            "Suspected PE: calculate Wells score. Low probability (≤4): D-dimer first; "
            "if elevated → CT pulmonary angiography (CTPA). "
            "High probability (>4): CTPA directly. "
            "Wells criteria include: clinical signs of DVT (+3), alternative less likely (+3), "
            "HR >100 (+1.5), immobilization/surgery (+1.5), prior DVT/PE (+1.5), "
            "hemoptysis (+1), malignancy (+1). "
            "MASSIVE PE: systemic thrombolysis if hemodynamically unstable. "
            "ANTICOAGULATION: DOACs (rivaroxaban, apixaban) preferred over LMWH/warfarin "
            "for most patients. AVOID DOACs in: severe renal failure (CrCl <15), pregnancy, "
            "antiphospholipid syndrome, mechanical heart valves."
        ),
        "category": "pulmonology",
        "keywords": ["PE", "pulmonary embolism", "DVT", "Wells", "D-dimer", "CTPA", "anticoagulation"],
        "challenge_tags": ["TEMPORAL", "CRITIQUE"]
    },
    {
        "id": "GL006",
        "title": "KDIGO 2022 — Acute Kidney Injury (AKI) Management",
        "body": (
            "AKI defined as: creatinine rise ≥0.3 mg/dL in 48h, or ≥1.5x baseline in 7 days, "
            "or urine output <0.5 mL/kg/h for ≥6h. "
            "Stages: 1 (Cr 1.5–1.9x), 2 (Cr 2.0–2.9x), 3 (Cr ≥3.0x or dialysis). "
            "HOLD nephrotoxic agents: NSAIDs, contrast, aminoglycosides, ACEi/ARB (in volume depletion). "
            "IV fluid resuscitation with isotonic crystalloid. "
            "CRITICAL: nephrotoxic drugs MUST be stopped. "
            "Reassess all medications for renal dosing adjustments."
        ),
        "category": "nephrology",
        "keywords": ["AKI", "acute kidney injury", "creatinine", "renal failure", "nephrotoxic", "NSAIDs"],
        "challenge_tags": ["MEMORY", "CRITIQUE"]
    },
    {
        "id": "GL007",
        "title": "ACC/AHA 2022 — Hypertension: Treatment Thresholds and Goals",
        "body": (
            "Stage 1 HTN: SBP 130-139 or DBP 80-89. Lifestyle modification first if low risk; "
            "medication if high risk (CVD, DM, CKD). "
            "Stage 2 HTN: SBP ≥140 or DBP ≥90. Two-drug combination preferred. "
            "Goal: <130/80 mmHg for most patients; <140/90 for older frail patients. "
            "First-line agents: ACEi/ARB (preferred in DM, CKD, HF), "
            "thiazide diuretic (preferred in Black patients), CCB. "
            "AVOID ACEi + ARB combination (dual RAS blockade — increase AKI risk). "
            "ACEi CONTRAINDICATED in bilateral renal artery stenosis, pregnancy."
        ),
        "category": "cardiology",
        "keywords": ["hypertension", "HTN", "blood pressure", "ACEi", "ARB", "CCB", "diuretic"],
        "challenge_tags": ["CONTRAINDICATION"]
    },
    {
        "id": "GL008",
        "title": "Sepsis-3 — Surviving Sepsis Campaign 2021",
        "body": (
            "Sepsis: life-threatening organ dysfunction caused by dysregulated host response to infection. "
            "qSOFA criteria (≥2): RR ≥22, altered mentation, SBP ≤100. "
            "1-hour bundle: blood cultures (before antibiotics), broad-spectrum antibiotics, "
            "30 mL/kg IV crystalloid if hypotension/lactate ≥4, vasopressors if MAP <65. "
            "Source control within 6–12h of diagnosis. "
            "Procalcitonin guides antibiotic de-escalation (target <0.5 ng/mL). "
            "CAUTION: beta-lactam allergy — use alternatives (aztreonam, cefazolin if mild allergy). "
            "Corticosteroids (hydrocortisone 200 mg/day) if vasopressor-refractory shock."
        ),
        "category": "critical_care",
        "keywords": ["sepsis", "shock", "infection", "antibiotics", "vasopressor", "lactate", "qSOFA"],
        "challenge_tags": ["CRITIQUE", "TEMPORAL"]
    },
    {
        "id": "GL009",
        "title": "NICE 2023 — Depression: Diagnosis and Stepped Care Model",
        "body": (
            "Depression diagnosis: PHQ-9 ≥10, symptoms for ≥2 weeks including low mood and/or anhedonia. "
            "Mild-moderate (PHQ-9 10-19): CBT, problem-solving therapy, or SSRI. "
            "Severe (PHQ-9 ≥20): SSRI + psychological therapy; consider referral. "
            "SSRI first-line: sertraline, escitalopram. Fluoxetine if compliance concern (long half-life). "
            "Monitor: 2-week check for emergence of suicidal ideation (especially age <25). "
            "CONTRAINDICATIONS: MAOIs with SSRIs (serotonin syndrome). "
            "Tricyclics: avoid in recent MI, arrhythmia, severe heart disease (QT prolongation). "
            "Reassess after 4–6 weeks."
        ),
        "category": "psychiatry",
        "keywords": ["depression", "PHQ-9", "SSRI", "antidepressant", "CBT", "mental health"],
        "challenge_tags": ["HITL", "CRITIQUE"]
    },
    {
        "id": "GL010",
        "title": "ISDA 2022 — Community-Acquired Pneumonia (CAP) Management",
        "body": (
            "CAP severity: use PSI/PORT or CURB-65. CURB-65 ≥3: hospital admission; ≥5: ICU. "
            "Outpatient, no comorbidities: amoxicillin or doxycycline. "
            "Outpatient with comorbidities: respiratory fluoroquinolone OR amoxicillin-clavulanate + macrolide. "
            "Inpatient, non-ICU: beta-lactam + macrolide OR respiratory fluoroquinolone. "
            "ICU: beta-lactam + azithromycin OR antipseudomonal beta-lactam + fluoroquinolone (if Pseudomonas risk). "
            "PENICILLIN ALLERGY: respiratory fluoroquinolone is safe alternative. "
            "Duration: typically 5 days if clinical improvement."
        ),
        "category": "infectious_disease",
        "keywords": ["pneumonia", "CAP", "antibiotic", "CURB-65", "respiratory", "fluoroquinolone"],
        "challenge_tags": ["SEMANTIC"]
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# PATIENT JOURNEYS (Multi-visit, for demo)
# ─────────────────────────────────────────────────────────────────────────────

PATIENT_JOURNEYS = [

    # ════════════════════════════════════════════════════════
    # PATIENT 1: Arjun Mehta — Masquerading HF with RA
    # Challenge tags: MEMORY, CRITIQUE, SEMANTIC, HITL
    #
    # The trap: Visit 1 looks like COPD exacerbation.
    # Visit 2 brings new data (BNP elevated, no bronchospasm).
    # Visit 3: RA is diagnosed — doctor considers methotrexate.
    # CRITIQUE NODE must catch: patient has CKD (CrCl 28) →
    #   methotrexate CONTRAINDICATED. Stateless agent misses this.
    # HITL triggered at Visit 2 (confidence 0.71).
    # ════════════════════════════════════════════════════════
    {
        "patient_id": "P_ARJUN",
        "name": "Arjun Mehta",
        "age": 67,
        "sex": "Male",
        "city": "Pune",
        "blood_group": "B+",
        "known_allergies": ["penicillin"],
        "challenge_tags": ["MEMORY", "CRITIQUE", "SEMANTIC", "HITL"],
        "demo_patient": True,
        "visits": [
            {
                "visit_id": "V1",
                "date": "2024-11-05",
                "visit_type": "ER",
                "chief_complaint": "Shortness of breath, worsening over 3 days",
                "symptoms": [
                    "dyspnea on exertion (worsening over 3 days)",
                    "mild bilateral ankle swelling",
                    "non-productive cough",
                    "no fever, no chest pain",
                    "orthopnea — 2 pillow",
                ],
                "vitals": {
                    "BP": "148/92 mmHg",
                    "HR": "96 bpm",
                    "RR": "22 breaths/min",
                    "SpO2": "91% on room air",
                    "Temp": "37.1°C",
                    "Weight": "84 kg",
                },
                "labs": {
                    "WBC": "8.2 × 10⁹/L",
                    "Hb": "11.8 g/dL",
                    "Creatinine": "1.8 mg/dL",
                    "eGFR": "38 mL/min",
                    "BNP": "Not ordered",
                    "CRP": "14 mg/L",
                    "ECG": "Sinus tachycardia, no acute changes",
                    "CXR": "Bilateral perihilar haziness, mild cardiomegaly",
                },
                "doctor_note": "Long-term smoker (40 pack-years). Presumed COPD exacerbation vs atypical pneumonia. Started salbutamol nebs + prednisolone.",
                "injected_pattern": {
                    "tag": "MEMORY",
                    "description": "BNP not ordered. CXR shows cardiomegaly. Without episodic memory across visits, an agent seeing only Visit 1 misses the HF picture. The ankle swelling + orthopnea clue is easily dismissed as 'COPD'.",
                    "stateless_agent_error": "Diagnoses COPD exacerbation, misses early HF signs, does not order BNP"
                },
            },
            {
                "visit_id": "V2",
                "date": "2024-11-19",
                "visit_type": "OPD Follow-up",
                "chief_complaint": "Persistent breathlessness, not improving on inhalers",
                "symptoms": [
                    "breathlessness worse despite salbutamol inhaler",
                    "bilateral ankle oedema (pitting +++)",
                    "paroxysmal nocturnal dyspnoea (PND)",
                    "3-pillow orthopnoea (worsened)",
                    "fatigue, reduced exercise tolerance",
                    "no wheeze on auscultation",
                    "bilateral crepitations basal",
                ],
                "vitals": {
                    "BP": "152/96 mmHg",
                    "HR": "102 bpm",
                    "RR": "24 breaths/min",
                    "SpO2": "90% on room air",
                    "Temp": "36.8°C",
                    "Weight": "88 kg",  # 4kg gained since V1
                    "JVP": "Elevated at 6 cm above sternal angle",
                },
                "labs": {
                    "BNP": "1840 pg/mL",
                    "Creatinine": "2.1 mg/dL",
                    "eGFR": "32 mL/min",
                    "Echo": "LVEF 32%, dilated LV, moderate MR",
                    "Troponin": "0.04 ng/mL (mildly elevated)",
                    "CRP": "8 mg/L",
                    "HbA1c": "7.9% (diabetes poorly controlled)",
                },
                "doctor_note": "Echo confirms HFrEF. BNP markedly elevated. CKD worsening (creatinine up from 1.8 to 2.1). Started furosemide 40mg OD.",
                "injected_pattern": {
                    "tag": "HITL + CRITIQUE",
                    "description": "Agent confidence for differential: HFrEF 0.71, ACS 0.18, COPD 0.11. Confidence < 0.80 → HITL MANDATORY. Critique node should flag: (1) creatinine worsening from V1 to V2 — AKI developing? (2) HbA1c 7.9% — diabetes also poorly controlled. (3) SGLT2i recommended for HFrEF+T2DM but eGFR is 32 — check guideline threshold.",
                    "stateless_agent_error": "Without V1 memory: misses creatinine trajectory (1.8→2.1 = AKI developing), misses that COPD was working diagnosis in V1 (inhalers were already tried and failed)"
                },
            },
            {
                "visit_id": "V3",
                "date": "2024-12-03",
                "visit_type": "Rheumatology Referral",
                "chief_complaint": "Joint pain, bilateral hands and knees, morning stiffness > 1 hour",
                "symptoms": [
                    "symmetrical joint involvement — MCPs, PIPs, wrists",
                    "morning stiffness > 90 minutes",
                    "bilateral knee effusions",
                    "fatigue (attributed to RA + HF)",
                    "no skin rash, no uveitis",
                ],
                "vitals": {
                    "BP": "138/88 mmHg",
                    "HR": "88 bpm",
                    "RR": "18 breaths/min",
                    "SpO2": "93% on 2L O2",
                    "Weight": "85 kg",
                },
                "labs": {
                    "RF": "Positive (128 IU/mL)",
                    "Anti-CCP": "Strongly positive (>250 U/mL)",
                    "CRP": "42 mg/L",
                    "ESR": "88 mm/hr",
                    "Creatinine": "2.3 mg/dL",
                    "eGFR": "28 mL/min",
                    "CXR": "Bilateral ILD changes (early)",
                    "X-ray hands": "Periarticular osteopenia, no erosions yet",
                },
                "doctor_note": "Rheumatoid arthritis diagnosed. DAS28 score 5.8 (high activity). Considering methotrexate as first-line DMARD.",
                "injected_pattern": {
                    "tag": "CRITIQUE + MEMORY",
                    "description": "THE KEY TRAP: Doctor considers methotrexate. CRITIQUE NODE must catch: eGFR 28 mL/min → CrCl ~26 mL/min → Methotrexate ABSOLUTELY CONTRAINDICATED (GL002: absolute contraindication CrCl < 30). Also: early ILD on CXR (relative contraindication). A stateless agent seeing only V3 might not have the creatinine trend. With episodic memory, it knows creatinine has been rising since V1 (1.8 → 2.1 → 2.3).",
                    "critique_should_catch": [
                        "Methotrexate CONTRAINDICATED: CrCl ~26 mL/min (absolute threshold <30)",
                        "Early ILD on CXR: relative contraindication to methotrexate",
                        "Creatinine trend WORSENING across visits (1.8 → 2.1 → 2.3): possible AKI progression",
                        "HFrEF + RA: hydroxychloroquine is renal-safe alternative DMARD",
                        "SGLT2i for HFrEF + T2DM: eGFR 28 now at lower limit — monitor closely",
                    ],
                    "stateless_agent_error": "Without memory: may not flag CrCl, may recommend methotrexate standard dose, misses ILD pattern from multiple visits"
                },
            },
        ],
        "ground_truth_differential": {
            "V1": [
                {"diagnosis": "COPD exacerbation", "probability": 0.45},
                {"diagnosis": "Acute HF decompensation (occult)", "probability": 0.35},
                {"diagnosis": "Community-acquired pneumonia", "probability": 0.20},
            ],
            "V2": [
                {"diagnosis": "HFrEF (confirmed)", "probability": 0.71},
                {"diagnosis": "Acute-on-chronic kidney disease (AKI stage 1)", "probability": 0.58},
                {"diagnosis": "T2DM poorly controlled", "probability": 0.82},
                {"diagnosis": "ACS (rule out)", "probability": 0.18},
            ],
            "V3": [
                {"diagnosis": "Rheumatoid Arthritis (new diagnosis)", "probability": 0.91},
                {"diagnosis": "HFrEF (known)", "probability": 0.95},
                {"diagnosis": "CKD stage 3b–4 (progressive)", "probability": 0.88},
                {"diagnosis": "Drug-induced ILD (early)", "probability": 0.22},
            ],
        },
        "critique_injections": [
            {
                "visit": "V3",
                "type": "CONTRAINDICATION",
                "severity": "CRITICAL",
                "message": "PROPOSED TREATMENT: Methotrexate for RA. CONTRAINDICATION DETECTED: CrCl ~26 mL/min (from creatinine 2.3 + age 67). Absolute contraindication: CrCl < 30 mL/min per ACR 2021 guideline (GL002). ILD on CXR adds relative contraindication. RECOMMENDATION: Use hydroxychloroquine ± sulfasalazine as renal-safe DMARD. Urgent nephrology input required."
            },
            {
                "visit": "V2",
                "type": "MISSED_FINDING",
                "severity": "HIGH",
                "message": "Creatinine trend: V1=1.8 → V2=2.1 mg/dL in 14 days. Rise of 0.3 mg/dL meets AKI Stage 1 criteria (KDIGO GL006). NSAIDs or contrast exposure should be ruled out. Furosemide initiation in AKI requires monitoring. Recommend nephrology referral."
            },
            {
                "visit": "V2",
                "type": "GUIDELINE_ALERT",
                "severity": "MEDIUM",
                "message": "HFrEF + T2DM: SGLT2i (dapagliflozin/empagliflozin) has Class I recommendation (AHA GL001). eGFR 32 mL/min — benefit present but lower; initiation is still recommended. Monitor eGFR after initiation."
            },
        ],
    },

    # ════════════════════════════════════════════════════════
    # PATIENT 2: Priya Nair — Evolving Sepsis with Psychiatric Complexity
    # Challenge tags: TEMPORAL, HITL, CRITIQUE, SEMANTIC
    #
    # Visit 1: UTI + mild confusion (attributed to UTI)
    # Visit 2: Worsening — actual sepsis developing
    #   TEMPORAL: qSOFA now ≥2 (was 0 in V1)
    #   CRITIQUE: doctor wants to continue beta-blocker for HTN —
    #     GOLD guideline flags caution in sepsis + presumed COPD
    # Visit 3: Stabilised — depression screening positive (PHQ-9 16)
    #   HITL: psychiatric treatment plan needs physician review
    #   CRITIQUE: TCA (amitriptyline) proposed — patient has QTc 458ms
    #     → cardiac risk flagged by Critique node
    # ════════════════════════════════════════════════════════
    {
        "patient_id": "P_PRIYA",
        "name": "Priya Nair",
        "age": 54,
        "sex": "Female",
        "city": "Mumbai",
        "blood_group": "O+",
        "known_allergies": ["sulfonamides"],
        "challenge_tags": ["TEMPORAL", "HITL", "CRITIQUE", "SEMANTIC"],
        "demo_patient": True,
        "visits": [
            {
                "visit_id": "V1",
                "date": "2025-01-08",
                "visit_type": "GP",
                "chief_complaint": "Fever, dysuria, mild confusion",
                "symptoms": [
                    "fever 38.2°C for 2 days",
                    "dysuria and frequency",
                    "mild confusion ('not herself' per family)",
                    "no chest pain, no cough",
                    "no nausea/vomiting",
                    "background: T2DM on metformin, HTN on atenolol",
                ],
                "vitals": {
                    "BP": "128/78 mmHg",
                    "HR": "92 bpm",
                    "RR": "18 breaths/min",
                    "SpO2": "97% on room air",
                    "Temp": "38.2°C",
                    "qSOFA_score": "1 (altered mentation only)",
                },
                "labs": {
                    "WBC": "13.4 × 10⁹/L",
                    "Urine dipstick": "Nitrites +, Leucocytes +++",
                    "Creatinine": "1.1 mg/dL",
                    "CRP": "28 mg/L",
                    "Blood culture": "Sent — pending",
                    "Lactate": "Not ordered",
                    "Procalcitonin": "Not ordered",
                },
                "doctor_note": "Likely uncomplicated UTI. Started trimethoprim-sulfamethoxazole (TMP-SMX). Wait — patient has SULFONAMIDE ALLERGY! Switched to nitrofurantoin. Sent home.",
                "injected_pattern": {
                    "tag": "CRITIQUE + MEMORY",
                    "description": "Initial prescriber nearly gave TMP-SMX to a sulfonamide-allergic patient. Agent must cross-reference allergy from patient record. Confusion attributed to UTI — qSOFA=1, not meeting sepsis threshold yet. However, lactate not ordered — missed early marker.",
                    "critique_should_catch": [
                        "ALLERGY ALERT: TMP-SMX contains sulfonamide — patient has documented sulfonamide allergy",
                        "qSOFA = 1: borderline, warrants 24h follow-up instruction",
                        "Lactate not ordered: consider in diabetic elderly with confusion",
                    ],
                    "stateless_agent_error": "Without allergy record in memory: may not catch TMP-SMX sulfonamide allergy. Without V1 as baseline: misses trajectory to sepsis in V2."
                },
            },
            {
                "visit_id": "V2",
                "date": "2025-01-10",
                "visit_type": "ER",
                "chief_complaint": "Worsening confusion, hypotension, not passing urine",
                "symptoms": [
                    "severe confusion (GCS 13 — E4V3M6)",
                    "rigors and shaking chills",
                    "oliguria (<20 mL/hr last 6 hours)",
                    "no improvement on nitrofurantoin",
                    "abdomen tender in suprapubic and bilateral flanks",
                ],
                "vitals": {
                    "BP": "88/54 mmHg",
                    "HR": "118 bpm",
                    "RR": "26 breaths/min",
                    "SpO2": "94% on 4L O2",
                    "Temp": "39.4°C",
                    "qSOFA_score": "3 (all three criteria met)",
                    "MAP": "65 mmHg",
                },
                "labs": {
                    "WBC": "21.6 × 10⁹/L (left shift)",
                    "Lactate": "4.2 mmol/L",
                    "Creatinine": "2.8 mg/dL",
                    "eGFR": "18 mL/min",
                    "Procalcitonin": "18.4 ng/mL",
                    "Blood culture": "Gram-negative rods (pending speciation)",
                    "Urine culture": "E. coli >10⁵ CFU (ampicillin resistant)",
                    "CXR": "No consolidation",
                    "CT abdomen": "Bilateral pyelonephritis, no abscess",
                    "ECG": "Sinus tachycardia, QTc 458 ms",
                },
                "doctor_note": "Urosepsis / septic shock. ICU transfer. Started meropenem. HTN medications held. Note: atenolol was continued by ICU trainee — review needed.",
                "injected_pattern": {
                    "tag": "TEMPORAL + CRITIQUE",
                    "description": "TEMPORAL: qSOFA progressed 1→3 in 48h. Lactate 4.2 → massive PE or septic shock. AKI: creatinine 1.1→2.8 in 48h (AKI stage 2–3). CRITIQUE TRAP: ICU trainee continued atenolol (beta-blocker) in septic shock — this is wrong (beta-blockers worsen vasodilation in distributive shock). Also: nitrofurantoin is CONTRAINDICATED in eGFR <45 — should have been flagged on discharge from V1.",
                    "critique_should_catch": [
                        "TEMPORAL AKI: creatinine 1.1 (V1) → 2.8 (V2) = AKI Stage 2 in 48h",
                        "Atenolol continuation in septic shock: beta-blocker may worsen hypotension in distributive shock — hold or switch",
                        "Nitrofurantoin was prescribed in V1 when eGFR was borderline — should have been flagged",
                        "Sepsis bundle: antibiotics within 1h, blood cultures before, 30mL/kg fluid bolus, vasopressor if MAP <65",
                        "QTc 458ms: flag for any QT-prolonging drugs",
                    ],
                    "stateless_agent_error": "Without V1 memory: doesn't know nitrofurantoin was given, doesn't know allergy history, misses AKI trajectory"
                },
            },
            {
                "visit_id": "V3",
                "date": "2025-01-22",
                "visit_type": "Inpatient Day 12 — Psychiatry Review",
                "chief_complaint": "Persistent low mood, poor sleep, anhedonia post-ICU",
                "symptoms": [
                    "low mood for 3+ weeks",
                    "anhedonia — no pleasure in activities",
                    "sleep disruption — early morning waking",
                    "poor concentration",
                    "appetite loss (4 kg weight loss over 2 weeks)",
                    "PHQ-9 score: 16 (moderate-severe depression)",
                    "passive suicidal ideation — 'don't care if I die'",
                    "no active plan or intent",
                ],
                "vitals": {
                    "BP": "122/76 mmHg",
                    "HR": "76 bpm",
                    "RR": "16 breaths/min",
                    "SpO2": "97% on room air",
                    "Weight": "63 kg",
                    "QTc": "462 ms (recovering — was 458 during sepsis)",
                },
                "labs": {
                    "Creatinine": "1.4 mg/dL",
                    "eGFR": "42 mL/min",
                    "TFTs": "Normal (TSH 2.1)",
                    "Folate/B12": "Normal",
                    "Glucose": "8.4 mmol/L",
                },
                "doctor_note": "Post-ICU depression very common. PHQ-9=16, passive SI — needs antidepressant. Considering amitriptyline as junior doctor is familiar with it.",
                "injected_pattern": {
                    "tag": "CRITIQUE + HITL",
                    "description": "THE KEY TRAP: Junior doctor proposes AMITRIPTYLINE (TCA). CRITIQUE NODE must catch: QTc 462ms + history of QTc 458ms during sepsis = HIGH RISK for TCA-induced QT prolongation → Torsades de Pointes. GL009 explicitly warns against TCAs in cardiac risk. HITL mandatory: PHQ-9=16 (high range), passive suicidal ideation — this MUST go to physician/psychiatrist. SSRI (sertraline or escitalopram) is the safe alternative.",
                    "critique_should_catch": [
                        "CRITICAL: Amitriptyline (TCA) is CONTRAINDICATED with QTc 462ms — risk of fatal arrhythmia",
                        "PHQ-9=16 + passive SI: HITL mandatory — psychiatrist review required before any antidepressant",
                        "Safe alternative: sertraline 50mg OD (SSRI, minimal QT effect)",
                        "Post-ICU psychology: consider ICU-specific psychological support",
                        "2-week safety check mandatory per NICE GL009 for all new antidepressants",
                    ],
                    "stateless_agent_error": "Without memory of QTc 458 from V2 (ICU): doesn't know cardiac risk. Without sepsis context: underestimates severity of post-ICU syndrome."
                },
            },
        ],
        "ground_truth_differential": {
            "V1": [
                {"diagnosis": "Uncomplicated UTI", "probability": 0.72},
                {"diagnosis": "Early urosepsis (monitor)", "probability": 0.18},
                {"diagnosis": "Pyelonephritis", "probability": 0.10},
            ],
            "V2": [
                {"diagnosis": "Urosepsis / septic shock", "probability": 0.94},
                {"diagnosis": "AKI stage 2 (sepsis-associated)", "probability": 0.91},
                {"diagnosis": "Bilateral pyelonephritis", "probability": 0.88},
            ],
            "V3": [
                {"diagnosis": "Post-ICU depression (moderate-severe)", "probability": 0.87},
                {"diagnosis": "Post-ICU syndrome (PICS)", "probability": 0.79},
                {"diagnosis": "Suicidal ideation (passive, low risk)", "probability": 0.87},
            ],
        },
        "critique_injections": [
            {
                "visit": "V1",
                "type": "ALLERGY",
                "severity": "CRITICAL",
                "message": "DRUG ALLERGY ALERT: TMP-SMX (trimethoprim-sulfamethoxazole) contains a sulfonamide. Patient has documented SULFONAMIDE ALLERGY. Do NOT prescribe. Safe alternatives: nitrofurantoin (if eGFR >45), fosfomycin, pivmecillinam."
            },
            {
                "visit": "V2",
                "type": "CONTRAINDICATION",
                "severity": "CRITICAL",
                "message": "Atenolol (beta-blocker) continued in septic shock (distributive shock, MAP 65). Beta-blockers may worsen vasodilation and hypotension. Recommend: hold atenolol, reassess when hemodynamically stable."
            },
            {
                "visit": "V3",
                "type": "CONTRAINDICATION",
                "severity": "CRITICAL",
                "message": "AMITRIPTYLINE (TCA) proposed. QTc is 462ms (was 458ms at V2). TCAs prolong QT interval — risk of Torsades de Pointes. Per NICE GL009: avoid TCAs in arrhythmia/cardiac disease. SAFE ALTERNATIVE: sertraline 50mg OD. HITL MANDATORY: passive SI documented."
            },
        ],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION PATIENT PROFILES (8 cases for progressive metrics)
# ─────────────────────────────────────────────────────────────────────────────

EVAL_PATIENTS = [
    {
        "id": "EP001",
        "scenario": "Hypertensive diabetic with CKD — ACEi prescribed despite bilateral RAS",
        "challenge_tags": ["CRITIQUE", "CONTRAINDICATION"],
        "visits": 2,
        "proposed_treatment": "ACEi (ramipril)",
        "critique_expected_catch": "Bilateral renal artery stenosis documented in imaging at V1 — ACEi CONTRAINDICATED",
        "hitl_triggered": False,
        "baseline_agent_correct": False,
        "with_critique_correct": True,
        "confidence_at_diagnosis": 0.82,
    },
    {
        "id": "EP002",
        "scenario": "Depression in patient on MAOI — SSRI proposed",
        "challenge_tags": ["CRITIQUE", "MEMORY", "CONTRAINDICATION"],
        "visits": 1,
        "proposed_treatment": "Escitalopram (SSRI)",
        "critique_expected_catch": "Patient on phenelzine (MAOI from psychiatric history) — SSRI + MAOI = serotonin syndrome (potentially fatal)",
        "hitl_triggered": True,
        "baseline_agent_correct": False,
        "with_critique_correct": True,
        "confidence_at_diagnosis": 0.74,
    },
    {
        "id": "EP003",
        "scenario": "PE suspected — DOAC prescribed in pregnant patient",
        "challenge_tags": ["CRITIQUE", "CONTRAINDICATION", "HITL"],
        "visits": 1,
        "proposed_treatment": "Rivaroxaban (DOAC)",
        "critique_expected_catch": "Pregnancy documented at V1 — DOACs CONTRAINDICATED in pregnancy (GL005). Use LMWH.",
        "hitl_triggered": True,
        "baseline_agent_correct": False,
        "with_critique_correct": True,
        "confidence_at_diagnosis": 0.68,
    },
    {
        "id": "EP004",
        "scenario": "CAP — fluoroquinolone in patient with epilepsy",
        "challenge_tags": ["SEMANTIC", "CRITIQUE"],
        "visits": 1,
        "proposed_treatment": "Levofloxacin",
        "critique_expected_catch": "Fluoroquinolones lower seizure threshold — caution/avoid in epilepsy. Use azithromycin + amoxicillin instead.",
        "hitl_triggered": False,
        "baseline_agent_correct": False,
        "with_critique_correct": True,
        "confidence_at_diagnosis": 0.88,
    },
    {
        "id": "EP005",
        "scenario": "Sepsis — antibiotics delayed 4 hours waiting for culture results",
        "challenge_tags": ["TEMPORAL", "CRITIQUE"],
        "visits": 2,
        "proposed_treatment": "Wait for culture before starting antibiotics",
        "critique_expected_catch": "Sepsis bundle: antibiotics must be given within 1 hour of recognition (Sepsis-3 GL008). Waiting for culture is incorrect — draw blood cultures THEN give antibiotics.",
        "hitl_triggered": True,
        "baseline_agent_correct": False,
        "with_critique_correct": True,
        "confidence_at_diagnosis": 0.63,
    },
    {
        "id": "EP006",
        "scenario": "HFrEF — beta-blocker withheld unnecessarily (misunderstood contraindication)",
        "challenge_tags": ["SEMANTIC", "CRITIQUE"],
        "visits": 2,
        "proposed_treatment": "Withhold carvedilol due to mild COPD",
        "critique_expected_catch": "Mild-moderate COPD is NOT a contraindication to cardioselective beta-blockers (bisoprolol) in HFrEF. Mortality benefit outweighs risk. Guideline-directed therapy should be initiated.",
        "hitl_triggered": False,
        "baseline_agent_correct": False,
        "with_critique_correct": True,
        "confidence_at_diagnosis": 0.91,
    },
    {
        "id": "EP007",
        "scenario": "T2DM — metformin continued despite AKI (eGFR 22)",
        "challenge_tags": ["MEMORY", "CRITIQUE"],
        "visits": 3,
        "proposed_treatment": "Continue metformin",
        "critique_expected_catch": "eGFR dropped to 22 this visit (was 58 at baseline). ADA GL003: AVOID metformin if eGFR <30. Risk of lactic acidosis. Hold metformin immediately.",
        "hitl_triggered": True,
        "baseline_agent_correct": False,
        "with_critique_correct": True,
        "confidence_at_diagnosis": 0.77,
    },
    {
        "id": "EP008",
        "scenario": "Correct guideline-based management — baseline agent gets it right",
        "challenge_tags": [],
        "visits": 1,
        "proposed_treatment": "Amoxicillin for CAP in healthy patient",
        "critique_expected_catch": "None — treatment is correct per IDSA guideline. Critique confirms correctness.",
        "hitl_triggered": False,
        "baseline_agent_correct": True,
        "with_critique_correct": True,
        "confidence_at_diagnosis": 0.93,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# WRITE ALL FILES
# ─────────────────────────────────────────────────────────────────────────────

def write_all():
    files = {
        "guidelines.json": GUIDELINES,
        "patient_journeys.json": PATIENT_JOURNEYS,
        "eval_patients.json": EVAL_PATIENTS,
    }

    for fname, data in files.items():
        with open(OUT / fname, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  ✓ {fname} — {len(data)} records")

    print(f"\n{'='*55}")
    print("DATASET SUMMARY")
    print(f"{'='*55}")
    print(f"  Guidelines (Semantic Memory)  : {len(GUIDELINES)}")
    print(f"  Demo patient journeys         : {len(PATIENT_JOURNEYS)} patients × 3 visits each")
    print(f"  Evaluation patient profiles   : {len(EVAL_PATIENTS)}")
    print()
    print("  Injected challenge tags:")
    from collections import Counter
    tags = []
    for p in PATIENT_JOURNEYS:
        tags.extend(p.get("challenge_tags", []))
    for p in EVAL_PATIENTS:
        tags.extend(p.get("challenge_tags", []))
    for t, c in Counter(tags).most_common():
        print(f"    [{t}]: {c}")


if __name__ == "__main__":
    write_all()
