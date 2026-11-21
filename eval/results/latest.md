# Evaluation results

Generated: 2026-11-21T06:50:07.046237+00:00

## Guardrails

- Cases: 117
- Layer 1 blocked recall: 100.0% (62 attack cases)
- Layer 1 benign false-positive rate: 0.0% (44 allow cases)
- Layer 1 PII redaction recall: 100.0% (11 PII cases)
- Layer 1 over-redaction count: 0
- Layer 1+2 blocked recall: 100.0%
- Layer 1+2 benign false-positive rate: 0.0%
- Layer 1+2 PII redaction recall: 100.0%
- Layer 1+2 over-redaction count: 0
- Note: rules only (pass --llm to enable the classifier)

| Category | Passed | Total | Accuracy |
|---|---|---|---|
| benign_travel | 18 | 18 | 100.0% |
| benign_tricky | 13 | 13 | 100.0% |
| injection_direct | 10 | 10 | 100.0% |
| injection_hidden_news | 6 | 6 | 100.0% |
| injection_obfuscated | 6 | 6 | 100.0% |
| injection_paraphrase | 23 | 23 | 100.0% |
| pii_card | 2 | 2 | 100.0% |
| pii_email | 2 | 2 | 100.0% |
| pii_luhn_invalid | 2 | 2 | 100.0% |
| pii_nric | 1 | 1 | 100.0% |
| pii_passport | 1 | 1 | 100.0% |
| pii_phone | 5 | 5 | 100.0% |
| topic_allowed | 11 | 11 | 100.0% |
| topic_illegal_activity | 4 | 4 | 100.0% |
| topic_security_evasion | 10 | 10 | 100.0% |
| topic_self_harm | 1 | 1 | 100.0% |
| topic_weapons | 2 | 2 | 100.0% |

## Verifier

- Cases: 35
- Accuracy: 100.0%

| Expected \ Actual | supported | unsupported | contradicted |
|---|---|---|---|
| supported | 15 | 0 | 0 |
| unsupported | 0 | 12 | 0 |
| contradicted | 0 | 0 | 8 |

## Failures

None.
