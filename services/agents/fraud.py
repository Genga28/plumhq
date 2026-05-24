"""FraudDetectorAgent: emits non-decisional signals about claim risk.

Inputs include the submitted `claims_history` (from the test cases) plus any
prior claims persisted in the DB. The agent does NOT decide rejection on its
own — it returns a FraudSignals object that the DecisionOrchestrator combines
with the rules result to determine MANUAL_REVIEW.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .. import db
from ..models import (
    ClaimSubmission,
    ExtractedDocument,
    FraudSignals,
    Policy,
)
from .base import BaseAgent


class FraudInput:
    def __init__(
        self,
        submission: ClaimSubmission,
        extracted_docs: list[ExtractedDocument],
        policy: Policy,
    ) -> None:
        self.submission = submission
        self.docs = extracted_docs
        self.policy = policy


class FraudDetectorAgent(BaseAgent[FraudInput, FraudSignals]):
    name = "FraudDetector"

    async def _run(self, payload: FraudInput) -> FraudSignals:
        sub = payload.submission
        thresholds = payload.policy.fraud_thresholds
        same_day_limit = int(thresholds.get("same_day_claims_limit", 2))
        monthly_limit = int(thresholds.get("monthly_claims_limit", 6))
        high_value = float(thresholds.get("high_value_claim_threshold", 25000))

        # Same-day count from supplied claims_history (test data) + DB
        history_same_day = [
            h for h in sub.claims_history
            if h.get("date") == sub.treatment_date.isoformat()
        ]
        db_same_day = db.claims_for_member_on_date(sub.member_id, sub.treatment_date.isoformat())
        same_day_count = len(history_same_day) + len(db_same_day) + 1  # +1 = this claim

        # Monthly count
        month_history = [
            h for h in sub.claims_history
            if str(h.get("date", "")).startswith(sub.treatment_date.strftime("%Y-%m"))
        ]
        db_month = db.claims_for_member_in_month(
            sub.member_id, sub.treatment_date.year, sub.treatment_date.month
        )
        monthly_count = len(month_history) + len(db_month) + 1

        is_high_value = sub.claimed_amount > high_value

        # Document alteration heuristic — fields surfaced by the extractor
        alteration = any(
            bool(d.fields.get("document_alteration_detected")) for d in payload.docs
        )

        triggers: list[str] = []
        score = 0.0

        if same_day_count > same_day_limit:
            triggers.append(
                f"Same-day claim count {same_day_count} exceeds limit {same_day_limit}"
            )
            score += 0.5
        if monthly_count > monthly_limit:
            triggers.append(
                f"Monthly claim count {monthly_count} exceeds limit {monthly_limit}"
            )
            score += 0.3
        if is_high_value:
            triggers.append(
                f"High-value claim (₹{sub.claimed_amount:.0f} > ₹{high_value:.0f})"
            )
            score += 0.2
        if alteration:
            triggers.append("Document alteration markers detected")
            score += 0.4

        return FraudSignals(
            same_day_count=same_day_count,
            monthly_count=monthly_count,
            is_high_value=is_high_value,
            alteration_detected=alteration,
            fraud_score=min(1.0, score),
            triggers=triggers,
        )

    def failure_default(self, payload: FraudInput, exc: Exception) -> FraudSignals:
        return FraudSignals(triggers=[f"FraudDetector failed: {exc}"], fraud_score=0.0)

    def trace_payload(self, result: FraudSignals) -> dict:
        return {
            "fraud_score": result.fraud_score,
            "same_day_count": result.same_day_count,
            "monthly_count": result.monthly_count,
            "is_high_value": result.is_high_value,
            "alteration_detected": result.alteration_detected,
            "triggers": result.triggers,
        }
