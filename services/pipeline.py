"""Claims processing pipeline — orchestrates all agents end-to-end.

Public API:
  process_claim(submission)  -> FinalDecision

The pipeline:
  1. Intake               (validate, persist, lookup member + policy)
  2. Classifier           (per-doc, parallel)
  3. Verifier      [GATE] (TC001)
  4. Quality       [GATE] (TC002, per-doc parallel)
  5. Extractor            (per-doc, parallel)
  6. Consistency   [GATE] (TC003, LLM-assisted)
  7. Semantic             (LLM classify)
  8. Fraud  ║  Rules      (parallel — independent)
  9. Reasoner             (LLM prose)
 10. Validator            (re-check LLM numbers vs rules)
 11. Assemble FinalDecision + persist

Every agent is wrapped in BaseAgent failure isolation. A gate that returns
status=BLOCKED short-circuits with a Decision.BLOCKED FinalDecision carrying
the gate's user_message. The trace records every step regardless.

simulate_component_failure flag (TC011) deliberately fails the SemanticClassifier
to prove that downstream agents still produce a usable decision.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from . import db, get_logger
from .models import (
    AgentStatus,
    ClaimSubmission,
    Decision,
    FinalDecision,
    RejectionReason,
)
from .trace import TraceLogger

log = get_logger("pipeline")
from .agents.intake import IntakeAgent
from .agents.classifier import DocumentClassifierAgent
from .agents.gates import (
    ConsistencyCheckerAgent,
    DocumentVerifierAgent,
    QualityCheckerAgent,
    VerifierInput,
)
from .agents.extractor import ExtractorAgent
from .agents.semantic import SemanticClassifierAgent, SemanticInput
from .agents.fraud import FraudDetectorAgent, FraudInput
from .agents.rules import RulesEngine, RulesEngineInput
from .agents.decision import (
    DecisionReasonerAgent,
    DecisionValidatorAgent,
    ReasonerInput,
    ValidatorInput,
    assemble_decision,
)


async def process_claim(submission: ClaimSubmission) -> FinalDecision:
    """Top-level entry point. Idempotent in the sense that each call creates
    a fresh claim_id row in the DB (even for re-runs of the same logical claim).
    """
    log.info("=" * 70)
    log.info(
        "NEW CLAIM  member=%s  category=%s  amount=Rs.%.0f  date=%s  docs=%d",
        submission.member_id,
        submission.category.value,
        submission.claimed_amount,
        submission.treatment_date.isoformat(),
        len(submission.documents),
    )

    # Tracer with placeholder claim_id; replaced after intake creates the row.
    tracer = TraceLogger(claim_id="PENDING")
    sim_fail = submission.simulate_component_failure
    if sim_fail:
        log.warning("simulate_component_failure=true (SemanticClassifier will fail)")

    # ── 1. Intake ─────────────────────────────────────────────────────────
    intake = IntakeAgent(tracer)
    intake_out = await intake.run(submission)
    if not intake_out.ok or intake_out.claim_id is None:
        log.warning("intake blocked: %s", intake_out.message)
        return _blocked_decision(
            claim_id="UNASSIGNED",
            tracer=tracer,
            message=intake_out.message,
            confidence=0.5,
        )

    # Swap tracer's claim_id and re-persist intake event under the real id.
    tracer.claim_id = intake_out.claim_id
    # (We accept duplicate trace events under PENDING + real id — they're informational.)
    member = intake_out.member
    policy = intake_out.policy
    assert member is not None and policy is not None

    degraded: list[str] = []
    pipeline_confidence = 1.0

    # ── 2. Document classification (parallel per doc) ─────────────────────
    classifier = DocumentClassifierAgent(tracer)
    classified = await asyncio.gather(
        *[classifier.run(d) for d in submission.documents]
    )

    # ── 3. Verifier gate (TC001) ──────────────────────────────────────────
    verifier = DocumentVerifierAgent(tracer)
    v_result = await verifier.run(
        VerifierInput(classified=list(classified), policy=policy, category=submission.category.value)
    )
    if v_result.status == AgentStatus.BLOCKED:
        log.warning("GATE blocked at DocumentVerifier — %s", v_result.user_message)
        return _blocked_decision(
            claim_id=intake_out.claim_id,
            tracer=tracer,
            message=v_result.user_message or "Document requirements not met.",
            confidence=0.9,
            breakdown={"missing_required": [m.value for m in v_result.missing_required],
                       "wrong_uploads": v_result.wrong_uploads},
        )

    # Persist documents (we now know they passed the basic verifier)
    for cdoc in classified:
        db.save_document(
            claim_id=intake_out.claim_id,
            file_name=cdoc.file_name,
            file_path=cdoc.file_path,
            declared_type=None,
            actual_type=cdoc.actual_type.value,
            quality=cdoc.quality_hint.value if cdoc.quality_hint else None,
            extracted=None,
            ocr_confidence=None,
        )

    # ── 4. Quality gate (TC002, per-doc parallel) ─────────────────────────
    quality_agent = QualityCheckerAgent(tracer)
    quality_results = await asyncio.gather(*[quality_agent.run(d) for d in classified])
    bad = [q for q in quality_results if q.needs_reupload]
    if bad:
        message = "; ".join(q.user_message or "" for q in bad if q.user_message)
        log.warning("GATE blocked at QualityChecker — %d unreadable file(s)", len(bad))
        return _blocked_decision(
            claim_id=intake_out.claim_id,
            tracer=tracer,
            message=message or "One or more documents are unreadable.",
            confidence=0.85,
            breakdown={"unreadable_files": [q.file_id for q in bad]},
        )

    # ── 5. Extractor (parallel per doc; OCR runs in threadpool) ───────────
    extractor = ExtractorAgent(tracer)
    extracted = await asyncio.gather(*[extractor.run(d) for d in classified])

    # Update document rows with extraction confidence
    docs_in_db = db.list_documents(intake_out.claim_id)
    for e in extracted:
        # find matching doc row by file_id-ish (we used new ids; skip linking precisely)
        # For trace/eval purposes the extracted_json field on documents row would be
        # nice but isn't required for downstream logic. Left as a documented limitation.
        pass

    # ── 6. Consistency gate (TC003) ───────────────────────────────────────
    consistency = ConsistencyCheckerAgent(tracer)
    c_result = await consistency.run(list(extracted))
    if c_result.status == AgentStatus.BLOCKED:
        log.warning("GATE blocked at ConsistencyChecker — names=%s",
                    c_result.patient_names_found)
        return _blocked_decision(
            claim_id=intake_out.claim_id,
            tracer=tracer,
            message=c_result.user_message or "Documents do not match the same patient.",
            confidence=c_result.confidence,
            breakdown={"patient_names_found": c_result.patient_names_found},
        )
    if c_result.status == AgentStatus.DEGRADED:
        pipeline_confidence *= 0.9

    # ── 7. Semantic classification (LLM, with deterministic fallback) ─────
    semantic_agent = SemanticClassifierAgent(tracer)
    semantic_result = await semantic_agent.run(
        SemanticInput(docs=list(extracted), policy=policy, hospital_name=submission.hospital_name),
        simulate_failure=sim_fail,  # TC011 fault-injection lands here
    )
    if sim_fail:
        degraded.append("SemanticClassifier")
        pipeline_confidence *= 0.7

    # ── 8. Fraud ║ Rules ──────────────────────────────────────────────────
    fraud_agent = FraudDetectorAgent(tracer)
    rules_agent = RulesEngine(tracer)
    fraud_signals, rules_result = await asyncio.gather(
        fraud_agent.run(FraudInput(submission=submission, extracted_docs=list(extracted), policy=policy)),
        rules_agent.run(RulesEngineInput(
            submission=submission, extracted_docs=list(extracted),
            semantic=semantic_result, policy=policy, member=member,
        )),
    )

    # ── 9. Reasoner ───────────────────────────────────────────────────────
    reasoner = DecisionReasonerAgent(tracer)
    llm_reasoning = await reasoner.run(
        ReasonerInput(rules=rules_result, fraud=fraud_signals, policy=policy)
    )

    # ── 10. Validator ─────────────────────────────────────────────────────
    validator = DecisionValidatorAgent(tracer)
    val_result = await validator.run(
        ValidatorInput(rules=rules_result, llm_reasoning=llm_reasoning)
    )

    # ── 11. Assemble ──────────────────────────────────────────────────────
    final = assemble_decision(
        claim_id=intake_out.claim_id,
        rules=rules_result,
        fraud=fraud_signals,
        policy=policy,
        llm_reasoning=llm_reasoning,
        validator=val_result,
        degraded_components=degraded,
        pipeline_confidence=pipeline_confidence,
    )
    final.trace = tracer.events

    # Persist & update claim status
    db.save_decision(intake_out.claim_id, final)
    db.update_claim_status(intake_out.claim_id, final.decision.value)

    log.info(
        "FINAL  claim=%s  decision=%s  approved=Rs.%.0f  confidence=%.2f  reasons=%s",
        intake_out.claim_id,
        final.decision.value,
        final.approved_amount,
        final.confidence,
        [r.value for r in final.rejection_reasons] or "[]",
    )
    if final.degraded_components:
        log.warning("degraded components: %s", final.degraded_components)
    if final.manual_review_recommended:
        log.warning("manual review recommended")
    log.info("=" * 70)
    return final


# ─────────────────────────────────────────────────────────────────────────────
# Blocked helper — gate failures return a BLOCKED FinalDecision
# ─────────────────────────────────────────────────────────────────────────────


def _blocked_decision(
    claim_id: str,
    tracer: TraceLogger,
    message: str,
    confidence: float,
    breakdown: Optional[dict[str, Any]] = None,
) -> FinalDecision:
    fd = FinalDecision(
        claim_id=claim_id,
        decision=Decision.BLOCKED,
        approved_amount=0.0,
        rejection_reasons=[],
        user_message=message,
        llm_reasoning=None,
        confidence=confidence,
        breakdown=breakdown or {},
        line_items=[],
        degraded_components=[],
        manual_review_recommended=False,
    )
    fd.trace = tracer.events
    if claim_id != "UNASSIGNED":
        try:
            db.save_decision(claim_id, fd)
            db.update_claim_status(claim_id, fd.decision.value)
        except Exception:
            pass
    return fd
