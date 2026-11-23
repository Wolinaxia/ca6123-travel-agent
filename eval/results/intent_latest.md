# Intent routing evaluation (M2)

Generated: 2026-11-22T11:59:17+00:00  
Fixed clock: 2026-11-22T12:00:00-05:00 · cases: `eval/intent_cases.jsonl` · LLM: `gemini-3.5-flash-lite`

## Summary

| Measure | rules | llm |
|---|---|---|
| Intent accuracy | 20/24 (83%) | 24/24 (100%) |
| Intent + all slots correct | 9/24 (38%) | 24/24 (100%) |
| Slot `origin_airport` | 4/13 (31%) | 17/17 (100%) |
| Slot `destination_airport` | 4/13 (31%) | 17/17 (100%) |
| Slot `departure_date` | 13/13 (100%) | 17/17 (100%) |
| Slot `departure_window` | 3/3 (100%) | 4/4 (100%) |
| Slot `arrive_by` | 2/3 (67%) | 3/3 (100%) |
| Slot `flight_number` | 2/2 (100%) | 3/3 (100%) |
| Slot `has_constraints` | 0/2 (0%) | 2/2 (100%) |
| Slot `missing` | 1/1 (100%) | 1/1 (100%) |

## Per intent (rules)

| Intent | Precision | Recall | Cases |
|---|---|---|---|
| assess_trip | 0.60 | 1.00 | 6 |
| plan_trip | 1.00 | 0.67 | 6 |
| rebook | 1.00 | 0.67 | 6 |
| out_of_scope | 1.00 | 1.00 | 6 |

Confusion matrix (rules; rows = expected, columns = predicted):

| | assess_trip | plan_trip | rebook | out_of_scope |
|---|---|---|---|---|
| **assess_trip** | 6 | 0 | 0 | 0 |
| **plan_trip** | 2 | 4 | 0 | 0 |
| **rebook** | 2 | 0 | 4 | 0 |
| **out_of_scope** | 0 | 0 | 0 | 6 |

## Per intent (llm)

| Intent | Precision | Recall | Cases |
|---|---|---|---|
| assess_trip | 1.00 | 1.00 | 6 |
| plan_trip | 1.00 | 1.00 | 6 |
| rebook | 1.00 | 1.00 | 6 |
| out_of_scope | 1.00 | 1.00 | 6 |

Confusion matrix (llm; rows = expected, columns = predicted):

| | assess_trip | plan_trip | rebook | out_of_scope |
|---|---|---|---|---|
| **assess_trip** | 6 | 0 | 0 | 0 |
| **plan_trip** | 0 | 6 | 0 | 0 |
| **rebook** | 0 | 0 | 6 | 0 |
| **out_of_scope** | 0 | 0 | 0 | 6 |

## Failures

- **rules** `as_01`: slots origin_airport, destination_airport. "Are flights out of LaGuardia to O'Hare running on time today?" → `{"origin_airport": null, "destination_airport": null, "departure_date": "2026-11-22", "departure_window": "whole_day", "arrive_by": null, "flight_number": null, "missing": ["origin", "destination"]}`
- **rules** `as_02`: slots origin_airport, destination_airport. "What's the disruption risk on ATL-MIA the day after tomorrow?" → `{"origin_airport": null, "destination_airport": null, "departure_date": "2026-11-24", "departure_window": "whole_day", "arrive_by": null, "flight_number": null, "missing": ["origin", "destination"]}`
- **rules** `as_03`: slots origin_airport, destination_airport. "Is snow likely to cause trouble for my Denver to Seattle trip on Tuesday morning?" → `{"origin_airport": null, "destination_airport": null, "departure_date": "2026-11-24", "departure_window": "morning", "arrive_by": null, "flight_number": null, "missing": ["origin", "destination"]}`
- **rules** `as_04`: slots origin_airport, destination_airport. "check delays for SFO to LAX on 2026-11-30" → `{"origin_airport": null, "destination_airport": null, "departure_date": "2026-11-30", "departure_window": "whole_day", "arrive_by": null, "flight_number": null, "missing": ["origin", "destination"]}`
- **rules** `as_05`: slots origin_airport, destination_airport. "Thinking of flying Boston to Washington on Thursday afternoon. Any weather alerts I should know about?" → `{"origin_airport": null, "destination_airport": null, "departure_date": "2026-11-26", "departure_window": "afternoon", "arrive_by": null, "flight_number": null, "missing": ["origin"]}`
- **rules** `as_06`: slots origin_airport, destination_airport. "How risky is Phoenix to Dallas tomorrow?" → `{"origin_airport": null, "destination_airport": null, "departure_date": "2026-11-23", "departure_window": "whole_day", "arrive_by": null, "flight_number": null, "missing": ["origin"]}`
- **rules** `pl_01`: slots origin_airport, destination_airport. "I must arrive in Seattle by 10am Wednesday, starting from Chicago. What's the safest plan?" → `{"origin_airport": null, "destination_airport": null, "departure_date": "2026-11-25", "departure_window": "whole_day", "arrive_by": "2026-11-25T10:00-08:00", "flight_number": null, "missing": ["origin"]}`
- **rules** `pl_03`: intent assess_trip (expected plan_trip). "Train or plane from New York to Washington DC on Tuesday evening, which is less likely to be disrupted?" → `{"origin_airport": "LGA", "destination_airport": "DCA", "departure_date": "2026-11-24", "departure_window": "evening", "arrive_by": null, "flight_number": null, "missing": []}`
- **rules** `pl_04`: slots has_constraints. "Least risky way to get from Chicago to Detroit on Thursday? I prefer rail if it's under 5 hours." → `{"origin_airport": "ORD", "destination_airport": "DTW", "departure_date": "2026-11-26", "departure_window": "whole_day", "arrive_by": null, "flight_number": null, "missing": []}`
- **rules** `pl_05`: slots origin_airport, destination_airport, arrive_by. "Need to reach Los Angeles from Las Vegas by 8pm today. Drive or fly?" → `{"origin_airport": null, "destination_airport": null, "departure_date": "2026-11-22", "departure_window": "whole_day", "arrive_by": null, "flight_number": null, "missing": ["destination"]}`
- **rules** `pl_06`: intent assess_trip (expected plan_trip). "Flying Houston to New Orleans on Saturday: should I use IAH or Hobby to avoid delays?" → `{"origin_airport": null, "destination_airport": null, "departure_date": "2026-11-28", "departure_window": "whole_day", "arrive_by": null, "flight_number": null, "missing": ["origin"]}`
- **rules** `rb_02`: slots origin_airport, destination_airport. "I missed my connection in Charlotte and need to rebook to Miami today." → `{"origin_airport": null, "destination_airport": null, "departure_date": "2026-11-22", "departure_window": "whole_day", "arrive_by": null, "flight_number": null, "missing": ["origin", "destination"]}`
- **rules** `rb_03`: intent assess_trip (expected rebook). "AA 100 JFK to LAX on Friday is delayed 5 hours, please put me on something earlier." → `{"origin_airport": null, "destination_airport": null, "departure_date": "2026-11-27", "departure_window": "whole_day", "arrive_by": null, "flight_number": "AA 100", "missing": ["origin"]}`
- **rules** `rb_04`: intent assess_trip (expected rebook). "Can you move my Seattle to Portland booking to Saturday instead? The storm looks bad." → `{"origin_airport": null, "destination_airport": null, "departure_date": "2026-11-28", "departure_window": "whole_day", "arrive_by": null, "flight_number": null, "missing": ["origin", "destination"]}`
- **rules** `rb_06`: slots has_constraints. "United cancelled UA 88 from San Francisco to Newark on Monday. Rebook please, no red-eyes." → `{"origin_airport": "SFO", "destination_airport": "EWR", "departure_date": "2026-11-23", "departure_window": "whole_day", "arrive_by": null, "flight_number": "UA 88", "missing": []}`
