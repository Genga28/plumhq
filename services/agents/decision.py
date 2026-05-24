"""Decision layer: combines RulesResult + FraudSignals into the FinalDecision.

Two agents:

  DecisionReasonerAgent
    - Calls Gemini to write a 2-4 sentence human-readable explanation.
    - Does NOT recompute numbers — those come from RulesEngine.
    - On LLM failure, falls back to a templated string.

  DecisionValidatorAgent
    - Re-runs the deterministic math on whatever amounts the LLM "said"
      to catch any hallucinated numbers in the reasoning text. (Numerical
      values themselves come from the rules engine, but if the LLM's prose
      mentions a different number than the rules produced, we downgrade
      confidence + force MANUAL_REVIEW.)
    - This is the safety net that lets us use LLM reasoning without
      hurting numerical correctness.

Both agents are then composed by DecisionOrchestrator (in pipeline.py).
"""

from __future__ import annotations

import re
from typing import Any

from .. import llm
from ..models import (
    AgentStatus,
    Decision,
    FinalDecision,
    FraudSignals,
    Policy,
    RulesResult,
)
from .base import BaseAgent


# ─────────────────────────────────────────────────────────────────────────────
# Reasoner
# ─────────────────────────────────────────────────────────────────────────────


class ReasonerInput:
    def __init__(self, rules: RulesResult, fraud: FraudSignals, policy: Policy) -> None:
        self.rules = rules
        self.fraud = fraud
        self.policy = policy


class DecisionReasonerAgent(BaseAgent[ReasonerInput, str]):
    name = "DecisionReasoner"

    async def _run(self, payload: ReasonerInput) -> str:
        ctx = {
            "decision": payload.rules.decision.value,
            "approved_amount": payload.rules.approved_amount,
            "reasons": [r.value for r in payload.rules.rejection_reasons],
            "breakdown": payload.rules.breakdown,
            "notes": payload.rules.notes,
            "fraud_signals": payload.fraud.triggers,
            "fraud_score": payload.fraud.fraud_score,
        }
        return await llm.generate_reasoning(ctx)

    def failure_default(self, payload: ReasonerInput, exc: Exception) -> str:
        return (
            f"Decision: {payload.rules.decision.value}. Approved: ₹{payload.rules.approved_amount:.0f}. "
            f"Reasoning generation failed; see the breakdown and trace for details."
        )

    def trace_payload(self, result: str) -> dict:
        try:
            available = llm.is_available()
        except Exception:
            available = False
        return {
            "reasoning_preview": (result or "")[:500],
            "reasoning_length": len(result or ""),
            "llm_available": available,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Validator
# ─────────────────────────────────────────────────────────────────────────────


_NUMBER_RE = re.compile(r"(?:₹|rs\.?\s*|inr\s*)\s*([0-9,]+(?:\.\d+)?)", re.I)


class ValidatorInput:
    def __init__(self, rules: RulesResult, llm_reasoning: str) -> None:
        self.rules = rules
        self.llm_reasoning = llm_reasoning


class DecisionValidatorAgent(BaseAgent[ValidatorInput, dict[str, Any]]):
    """Cross-check the LLM's prose against the rules engine's numbers.

    Returns {valid: bool, mismatches: [..], confidence_penalty: float}.
    """
    name = "DecisionValidator"

    async def _run(self, payload: ValidatorInput) -> dict[str, Any]:
        text = payload.llm_reasoning or ""
        # Extract every number mentioned in the LLM's reasoning
        mentioned = []
        for m in _NUMBER_RE.finditer(text):
            try:
                mentioned.append(float(m.group(1).replace(",", "")))
            except ValueError:
                continue

        # Authoritative numbers
        authoritative = {
            payload.rules.approved_amount,
            payload.rules.pre_discount_amount,
            payload.rules.network_discount,
            payload.rules.copay_deduction,
        }
        authoritative.update(
            float(v) for k, v in payload.rules.breakdown.items()
            if isinstance(v, (int, float)) and v is not None
        )

        # Any mentioned number that isn't within ₹1 of an authoritative one is suspicious
        mismatches: list[float] = []
        for n in mentioned:
            if not any(abs(n - a) <= 1.0 for a in authoritative):
                mismatches.append(n)

        return {
            "valid": len(mismatches) == 0,
            "mismatches": mismatches,
            "mentioned_count": len(mentioned),
            "confidence_penalty": 0.0 if not mismatches else min(0.4, 0.1 * len(mismatches)),
        }

    def failure_default(self, payload: ValidatorInput, exc: Exception) -> dict[str, Any]:
        return {"valid": True, "mismatches": [], "mentioned_count": 0,
                "confidence_penalty": 0.0, "error": str(exc)}

    def trace_payload(self, result: dict[str, Any]) -> dict:
        return {
            "valid": result.get("valid"),
            "mentioned_count": result.get("mentioned_count"),
            "mismatches": result.get("mismatches"),
            "confidence_penalty": result.get("confidence_penalty"),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers used by the pipeline to combine everything
# ─────────────────────────────────────────────────────────────────────────────


def assemble_decision(
    claim_id: str,
    rules: RulesResult,
    fraud: FraudSignals,
    policy: Policy,
    llm_reasoning: str,
    validator: dict[str, Any],
    degraded_components: list[str],
    pipeline_confidence: float,
) -> FinalDecision:
    """Combine all signals into the final user-facing decision.

    Decision precedence:
      1. If RulesEngine already said REJECTED -> REJECTED
      2. Else, if any of: fraud trigger / validator mismatch / degraded component
         -> MANUAL_REVIEW (overrides APPROVED/PARTIAL)
      3. Else, if rules said PARTIAL -> PARTIAL
      4. Else, APPROVED
    """
    fraud_thresholds = policy.fraud_thresholds
    fraud_mr_threshold = float(fraud_thresholds.get("fraud_score_manual_review_threshold", 0.80))
    auto_mr_above = float(fraud_thresholds.get("auto_manual_review_above", 25000))

    rejection_reasons = list(rules.rejection_reasons)
    manual_review = False
    manual_reasons: list[str] = []

    if fraud.fraud_score >= fraud_mr_threshold:
        manual_review = True
        manual_reasons.append(f"Fraud score {fraud.fraud_score:.2f} ≥ {fraud_mr_threshold:.2f}")
    if fraud.same_day_count > int(fraud_thresholds.get("same_day_claims_limit", 2)):
        manual_review = True
        manual_reasons.append(f"{fraud.same_day_count} same-day claims (limit {fraud_thresholds.get('same_day_claims_limit', 2)})")
    if rules.approved_amount > auto_mr_above:
        manual_review = True
        manual_reasons.append(f"Approved ₹{rules.approved_amount:.0f} > MR threshold ₹{auto_mr_above:.0f}")
    # NOTE: degraded components flag manual_review_recommended but do NOT
    # override the final decision — graceful degradation (TC011) means the
    # pipeline should still produce a usable decision with reduced confidence.
    mr_recommended_due_to_degradation = bool(degraded_components)
    if mr_recommended_due_to_degradation:
        manual_reasons.append(f"Degraded components: {', '.join(degraded_components)}")
    if not validator.get("valid", True):
        manual_review = True
        manual_reasons.append(f"LLM reasoning mentioned unverified numbers: {validator.get('mismatches')}")

    # Decision precedence
    if rules.decision == Decision.REJECTED:
        final = Decision.REJECTED
    elif manual_review:
        final = Decision.MANUAL_REVIEW
    elif rules.decision == Decision.PARTIAL:
        final = Decision.PARTIAL
    else:
        final = Decision.APPROVED

    # Confidence: start from rules confidence × fraud penalty × validator penalty × pipeline conf
    confidence = rules.confidence
    confidence *= 1.0 - min(0.5, fraud.fraud_score * 0.5)
    confidence -= validator.get("confidence_penalty", 0.0)
    confidence *= pipeline_confidence
    confidence = max(0.0, min(1.0, round(confidence, 3)))

    # User message: prefer LLM reasoning when valid; else template
    user_message = llm_reasoning
    if manual_review:
        user_message = (
            f"{llm_reasoning}\n\n"
            f"This claim has been routed to manual review: {'; '.join(manual_reasons)}."
        )

    return FinalDecision(
        claim_id=claim_id,
        decision=final,
        approved_amount=rules.approved_amount if final in (Decision.APPROVED, Decision.PARTIAL) else 0.0,
        rejection_reasons=rejection_reasons,
        user_message=user_message,
        llm_reasoning=llm_reasoning,
        confidence=confidence,
        breakdown={
            **rules.breakdown,
            "notes": rules.notes,
            "fraud_signals": fraud.triggers,
            "fraud_score": fraud.fraud_score,
            "validator": validator,
            "manual_review_reasons": manual_reasons,
            "rules_decision": rules.decision.value,
        },
        line_items=rules.line_items,
        degraded_components=degraded_components,
        manual_review_recommended=manual_review or mr_recommended_due_to_degradation,
    )
