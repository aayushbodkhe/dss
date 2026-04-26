# System Prompts — ClinicalDSS
================================

## Language Guardrails (embedded in all prompts)

MANDATORY PHRASE SUBSTITUTIONS:
| Prohibited → | Required |
|---|---|
| "The patient HAS..." | "Symptoms are consistent with..." |
| "Diagnosed with..." | "Findings suggest..." |
| "This confirms..." | "This is consistent with..." |
| "The patient WILL..." | "Risk factors indicate possible..." |
| "I recommend..." | "Consider..." / "Clinical review recommended..." |

## Confidence Thresholds

- < 0.60: RED — HITL mandatory, specialist referral recommended
- 0.60–0.79: AMBER — HITL mandatory, note amber status in clinical note
- 0.80–0.89: GREEN — Direct treatment node, HITL optional
- ≥ 0.90: HIGH — Direct treatment, auto-generate note

## HITL Override Logging Format (audit requirement)

{
  "timestamp": "ISO-8601",
  "visit_id": "V2",
  "physician_id": "DR_XXX",  # from auth context
  "reason": "confidence 0.71 | CRITICAL: methotrexate contraindication",
  "action": "EDITED|APPROVED|REJECTED",
  "original_treatment": "...",
  "final_treatment": "...",
  "critique_findings": [...],
}
