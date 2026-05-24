# Architecture

## Goals

Build an automated health-insurance claims processor that is:
1. **Correct** on financial math (regulator-friendly, deterministic numbers).
2. **Intelligent** about semantic understanding (T2DM = diabetes, "Bariatric Consultation" = obesity exclusion).
3. **Explainable** — every decision reconstructable from a structured trace.
4. **Resilient** — individual agent failures degrade confidence; the pipeline does not crash.

## Design philosophy: hybrid LLM + rules, not LLM-only

Pure-LLM decision making fails this assignment for three reasons:

1. **Financial determinism.** TC010 requires exactly ₹3,240 = 4500 × 0.80 (network) × 0.90 (copay). LLMs drift on arithmetic, even at temperature 0. Off-by-one on an insurance approval is a wrong decision.
2. **Auditability.** A trace that says "rule `waiting_period.diabetes=90d` triggered because join_date=2024-09-01, treatment_date=2024-10-15, delta=44 < 90" is regulator-grade. "The LLM said no" is not.
3. **Hallucination on exclusions.** An LLM might pattern-match "Panchakarma Therapy" against wellness/obesity language and reject incorrectly. Explicit allow-lists in a rules engine cannot.

So the LLM is used **narrowly, in three places, each with a deterministic fallback**:
- Semantic classification (diagnoses → policy waiting-period keys; line items → covered/excluded tags)
- Patient identity matching across documents (fuzzy name compare)
- Decision reasoning prose (after the numbers are produced)

Everything financial is pure Python in `services/agents/rules.py`. A separate **DecisionValidator** re-checks every number mentioned in the LLM's prose against the rules engine's authoritative breakdown; mismatches force `MANUAL_REVIEW`. This is the safety net.

## Pipeline (multi-agent)

```
                                  ┌────────────────────┐
                                  │   Intake           │  validate, lookup member→policy
                                  └─────────┬──────────┘
                                            │  claim_id
                                            ▼
                            ┌──────────────────────────────┐
                            │ DocumentClassifier (parallel)│  per file: detect actual type
                            └──────────────┬───────────────┘
                                           │
                            ┌──────────────▼───────────────┐
                            │ DocumentVerifier        [G]  │  TC001 — wrong type
                            └──────────────┬───────────────┘
                                           │
                            ┌──────────────▼───────────────┐
                            │ QualityChecker (parallel)[G] │  TC002 — unreadable
                            └──────────────┬───────────────┘
                                           │
                            ┌──────────────▼───────────────┐
                            │ Extractor (parallel)         │  pdfplumber / Tesseract → fields
                            └──────────────┬───────────────┘
                                           │
                            ┌──────────────▼───────────────┐
                            │ ConsistencyChecker      [G]  │  TC003 — patient mismatch (LLM)
                            └──────────────┬───────────────┘
                                           │
                            ┌──────────────▼───────────────┐
                            │ SemanticClassifier           │  LLM: diagnoses + line items
                            └──────────────┬───────────────┘
                                           │
                       ┌───────────────────┴───────────────────┐
                       ▼                                       ▼
              ┌─────────────────┐                    ┌─────────────────┐
              │ FraudDetector   │                    │ RulesEngine     │
              │ (signals only)  │                    │ (deterministic) │
              └────────┬────────┘                    └────────┬────────┘
                       │                                      │
                       └─────────────────┬────────────────────┘
                                         │
                            ┌────────────▼─────────────┐
                            │ DecisionReasoner (LLM)   │  human-readable explanation
                            └────────────┬─────────────┘
                                         │
                            ┌────────────▼─────────────┐
                            │ DecisionValidator (Py)   │  cross-check LLM numbers
                            └────────────┬─────────────┘
                                         │
                            ┌────────────▼─────────────┐
                            │ assemble_decision        │  final FinalDecision
                            └──────────────────────────┘

       [G] = blocking gate. Returns a BLOCKED FinalDecision with a
             specific, named user_message and short-circuits the pipeline.

       Every agent inherits BaseAgent, which wraps `_run()` in
       try/except + timing + automatic TraceLogger.record(). On
       exception, the agent returns a degraded default and the trace
       records FAILED status. The pipeline continues — that's TC011.
```

## Concurrency

| Stage | Concurrency | Why |
|---|---|---|
| Per-document classification / quality / extraction | `asyncio.gather` (threadpool inside Extractor for OCR I/O) | N docs in parallel; OCR releases GIL |
| Fraud ║ Rules | `asyncio.gather` | independent of each other |
| LLM calls (semantic, patient match, reasoner) | native async via worker thread around the sync Gemini SDK | I/O-bound |
| DB | sync SQLite, single writer | simplest correct model at this scale |

## Persistence

```
policies   (policy_id PK, name, config_json, created_at)
members    (member_id PK, policy_id FK, name, dob, gender, relationship, join_date, primary_member_id)
claims     (claim_id PK, member_id FK, policy_id FK, category, claimed_amount, treatment_date, hospital_name, status, created_at)
documents  (doc_id PK, claim_id FK, file_name, file_path, declared_type, actual_type, quality, extracted_json, ocr_confidence)
decisions  (decision_id PK, claim_id FK, decision, approved_amount, reasons_json, confidence, breakdown_json, llm_reasoning, user_message, created_at)
traces     (trace_id PK, claim_id FK, sequence, agent, action, status, payload_json, confidence, duration_ms, created_at)
```

`traces` is the observability spine: a row per agent step per claim, sequence-numbered, with payload, confidence, and duration. The Streamlit UI renders it as a timeline; `eval_report.md` lists it under each test case.

## Component contracts (the formal interface for every agent)

All agents implement `BaseAgent[IN, OUT]` and declare typed `_run(IN) -> OUT` and `failure_default(IN, exc) -> OUT`. The Pydantic models in `services/models.py` are the precise contracts:

| Agent | Input type | Output type | Errors |
|---|---|---|---|
| `IntakeAgent` | `ClaimSubmission` | `IntakeOutput` | member not found, policy not found |
| `DocumentClassifierAgent` | `UploadedDocument` | `ClassifiedDocument` | falls back to declared_type / UNKNOWN |
| `DocumentVerifierAgent` | `VerifierInput` | `VerifierResult` | DEGRADED on internal failure |
| `QualityCheckerAgent` | `ClassifiedDocument` | `QualityResult` | DEGRADED |
| `ExtractorAgent` | `ClassifiedDocument` | `ExtractedDocument` | DEGRADED, extraction_confidence ↓ |
| `ConsistencyCheckerAgent` | `list[ExtractedDocument]` | `ConsistencyResult` | DEGRADED (does not block on internal failure) |
| `SemanticClassifierAgent` | `SemanticInput` | `SemanticClassification` | confidence ↓ on LLM failure |
| `FraudDetectorAgent` | `FraudInput` | `FraudSignals` | returns fraud_score=0 on failure |
| `RulesEngine` | `RulesEngineInput` | `RulesResult` | falls back to MANUAL_REVIEW |
| `DecisionReasonerAgent` | `ReasonerInput` | `str` (reasoning text) | templated fallback string |
| `DecisionValidatorAgent` | `ValidatorInput` | `dict` (valid + mismatches) | reports as valid on internal failure |

## Failure isolation (TC011)

`BaseAgent.run()` wraps every `_run()` body. On exception:
1. Record a `TraceEvent` with status `FAILED` and the exception class/message.
2. Call `failure_default(payload, exc)` which returns a sane degraded result.
3. Return that result so the pipeline continues.

The pipeline orchestrator adds `degraded_components` to the `FinalDecision`. The decision assembler flags `manual_review_recommended=True` when any component is degraded **without overriding** the rules engine's APPROVED/PARTIAL decision — that's the assignment's exact requirement for TC011 ("APPROVED" with a note recommending manual review).

The `simulate_component_failure` flag in `ClaimSubmission` is honored by the orchestrator to deterministically fail `SemanticClassifier`, proving the pipeline survives.

## Trace-driven explainability (20% of the grade)

Every decision in `eval_report.md` includes the full agent trace. A reader can answer:

- "Why was TC005 rejected?" → trace shows `RulesEngine → OK` with payload containing `notes: ["Diabetes has a 90-day waiting period. Eligible from 2024-11-30."]`.
- "Why is TC011's confidence lower than TC004's?" → trace shows `SemanticClassifier → FAILED` (`simulated_failure` exception); orchestrator marks degraded and reduces confidence.
- "Why exactly ₹3,240 for TC010?" → trace's RulesEngine payload includes the full breakdown: `pre_discount=4500, network_discount=900, after_discount=3600, copay=360, approved=3240, calculation_order=[...]`.

The Streamlit UI renders the trace as an interactive timeline on every decision-review page.

## Edge cases handled

Beyond the 12 test cases, the system handles:

| Edge case | Handled by |
|---|---|
| Diagnosis shorthand ("T2DM", "HTN") | `SemanticClassifier` keyword table + LLM |
| Mixed covered + excluded line items | `_filter_line_items` in rules engine |
| Submission past 30-day deadline | RulesEngine deadline check (uses `submission_date`) |
| Below `minimum_claim_amount` | RulesEngine sanity check |
| Auto-MANUAL_REVIEW above `auto_manual_review_above` | `assemble_decision` |
| Bill total ≠ sum of line items | `ConsistencyCheckerAgent` (DEGRADED, not BLOCKED) |
| Member is a dependent | resolved via `members.primary_member_id` foreign key |
| Family floater | covered (single-member basis in v1, family aggregation noted as future work) |
| Hospital name spelling drift | `_fuzzy_network_match` (RapidFuzz token-set ratio) |
| Pre-auth required (MRI, CT, PET) above threshold | RulesEngine `_needs_pre_auth` |
| Hernia/herniation word-boundary distinction | regex `\b` in `_word_match` (real bug we fixed during eval) |
| Per-claim limit vs sub-limit interaction | `effective_limit = max(per_claim_limit, sub_limit)` on post-filter covered total |
| Gemini API down / no key | every LLM call has a deterministic fallback (`_semantic_keywords`, `_patient_match_fuzzy`, `_template_reasoning`) |
| All LLM calls fail | system still produces a rules-only decision; confidence drops; manual review flagged |
| LLM hallucinated number | `DecisionValidatorAgent` extracts every ₹ from prose and cross-checks against rules breakdown |
| Component failure mid-pipeline | BaseAgent failure isolation + degraded_components tracking |
| Cross-test claim row contamination (eval) | eval clears claims/decisions/traces between cases |

## Scaling to 10x (or 100x) the current load

What this system needs to handle Plum's stated 10M-lives goal:

1. **SQLite → managed Postgres.** SQLite single-writer becomes the bottleneck above ~50 claims/sec.
2. **OCR offloaded to workers.** OCR is CPU-bound. A task queue (Celery / Arq / RQ) with a worker pool, separate from the API. The orchestrator publishes per-document jobs and awaits a future.
3. **LLM rate limiting + batching.** Gemini has request-per-minute caps. Add token-bucket per API key, request batching for semantic-classification (batch N claims into one call when load is high).
4. **Async DB layer.** `aiosqlite`/`asyncpg` lets the gateway handle hundreds of concurrent claims without blocking on I/O.
5. **Tracer queue.** TraceLogger writes synchronously today. At scale, write events to a Kafka/Redis stream consumed by a single writer to Postgres + a search index (Elasticsearch) for ops queries.
6. **Caching of policy/member lookups.** Policies change rarely. Cache the `Policy` object per `policy_id` for the lifetime of an API instance.
7. **Idempotency keys on submission.** Currently each submission creates a fresh `claim_id`. Production needs a client-supplied idempotency key so retries don't duplicate claims.
8. **Per-tenant isolation.** Today everything is one DB. Multi-tenancy (per company) needs per-tenant schemas or strict tenant_id columns + row-level security.
9. **Observability beyond trace.** Add Prometheus metrics: per-agent latency p50/p95/p99, decision-type counters, fraud-score histograms, LLM error rates.
10. **A/B-able rule changes.** Today rules are code. To safely experiment with new rules at 10M lives, externalize the rules engine into a versioned config (or a DSL) so policy changes don't require deploys.

## What I'd change with more time

1. **End-to-end real-document tests.** Currently the eval uses `pre_extracted` JSON content; the Streamlit UI exercises the full OCR path, but it isn't part of the deterministic eval. Adding a small set of generated mock PDFs to `tests/` and running them through `extract_text` would catch OCR-layer regressions.
2. **Better extraction prompts.** The Tesseract path uses raw OCR text + regex. A small vision-LLM step (`Gemini-flash` with image input) would dramatically improve extraction quality on handwritten/scanned documents — that was deliberately out of scope per the user's "OCR for extraction" constraint.
3. **Unit tests for `rules.py`.** Each rule branch deserves a pytest covering exact numerics. The eval suite already covers them implicitly but isolated tests would catch regressions faster.
4. **Family floater aggregation.** Today the YTD/family limits are evaluated per member, not per family. Adding a `family_id` resolver and aggregating dependents' claims would lift this limitation.
5. **Real telemetry.** Trace events are great for per-claim debugging; a metrics layer (counter per decision type, histogram of LLM latency) would surface system-wide trends.

## Component contracts — the formal interface table

See `services/models.py` for the full Pydantic schema of every input/output. Every agent in `services/agents/` has a typed `_run(IN) -> OUT` signature. The schemas there ARE the contracts — another engineer could re-implement any single agent from the contract alone.
