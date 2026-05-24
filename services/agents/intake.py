"""IntakeAgent: validate submission, fetch member + policy from DB, persist claim row."""

from __future__ import annotations

from typing import Optional

from .. import db
from ..models import (
    AgentStatus,
    ClaimSubmission,
    Member,
    Policy,
)
from .base import BaseAgent


class IntakeOutput:
    def __init__(
        self,
        claim_id: Optional[str],
        member: Optional[Member],
        policy: Optional[Policy],
        ok: bool,
        message: str,
    ) -> None:
        self.claim_id = claim_id
        self.member = member
        self.policy = policy
        self.ok = ok
        self.message = message


class IntakeAgent(BaseAgent[ClaimSubmission, IntakeOutput]):
    name = "Intake"

    async def _run(self, submission: ClaimSubmission) -> IntakeOutput:
        member = db.get_member(submission.member_id)
        if member is None:
            return IntakeOutput(None, None, None, False,
                                f"Member {submission.member_id} not found. Please create the policy first or check the member ID.")

        # Policy: explicit if provided, else from member
        policy_id = submission.policy_id or member.policy_id
        policy = db.get_policy(policy_id)
        if policy is None:
            return IntakeOutput(None, member, None, False,
                                f"Policy {policy_id} not found in DB. Upload the policy first.")

        # Dependents are covered under the primary's policy — already resolved via member.policy_id.

        claim_id = db.create_claim(
            member_id=member.member_id,
            policy_id=policy.policy_id,
            category=submission.category.value,
            claimed_amount=submission.claimed_amount,
            treatment_date=submission.treatment_date.isoformat(),
            hospital_name=submission.hospital_name,
        )
        return IntakeOutput(claim_id, member, policy, True, f"Claim {claim_id} created.")

    def failure_default(self, submission: ClaimSubmission, exc: Exception) -> IntakeOutput:
        return IntakeOutput(None, None, None, False, f"Intake failed: {exc}")

    def trace_payload(self, result: IntakeOutput) -> dict:
        return {
            "claim_id": result.claim_id,
            "ok": result.ok,
            "message": result.message,
            "member_id": result.member.member_id if result.member else None,
            "policy_id": result.policy.policy_id if result.policy else None,
        }
