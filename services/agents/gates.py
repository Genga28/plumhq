"""Three gate agents that can block the pipeline before decisioning.

DocumentVerifierAgent  -> TC001 (wrong document type) — blocks if required types missing
QualityCheckerAgent    -> TC002 (unreadable doc) — blocks if any doc is unreadable
ConsistencyCheckerAgent -> TC003 (different patients) — blocks if names mismatch

Each gate produces a user_message that NAMES the specific problem (assignment
requirement: "the message must name the uploaded document type and the
required document type").
"""

from __future__ import annotations

from typing import Optional

from .. import llm
from ..models import (
    AgentStatus,
    ClassifiedDocument,
    ConsistencyResult,
    DocumentQuality,
    DocumentType,
    ExtractedDocument,
    Policy,
    QualityResult,
    VerifierResult,
)
from .base import BaseAgent


# ─────────────────────────────────────────────────────────────────────────────
# Verifier (TC001)
# ─────────────────────────────────────────────────────────────────────────────


class VerifierInput:
    def __init__(self, classified: list[ClassifiedDocument], policy: Policy, category: str) -> None:
        self.classified = classified
        self.policy = policy
        self.category = category


class DocumentVerifierAgent(BaseAgent[VerifierInput, VerifierResult]):
    name = "DocumentVerifier"

    async def _run(self, payload: VerifierInput) -> VerifierResult:
        req = payload.policy.document_requirements.get(payload.category, {})
        required: list[str] = list(req.get("required", []))
        if not required:
            return VerifierResult(status=AgentStatus.OK)

        uploaded_types: list[DocumentType] = [d.actual_type for d in payload.classified]
        uploaded_set = set(uploaded_types)

        missing = [DocumentType(t) for t in required if DocumentType(t) not in uploaded_set]

        if not missing:
            return VerifierResult(status=AgentStatus.OK)

        # Build a SPECIFIC user message — the assignment is explicit that we must
        # name what was uploaded AND what is required.
        uploaded_names = ", ".join(sorted({d.actual_type.value for d in payload.classified}))
        missing_names = ", ".join(m.value for m in missing)
        wrong_uploads = []
        # If the user uploaded only one type but multiple of it (e.g. 2 prescriptions),
        # surface that clearly.
        if len(missing) == 1 and len(uploaded_set) == 1:
            only = next(iter(uploaded_set))
            wrong_uploads.append({"uploaded": only.value, "required_instead": missing[0].value})
            message = (
                f"For a {payload.category} claim we need: {missing_names}. "
                f"You uploaded only {only.value} (×{len(payload.classified)}). "
                f"Please upload the missing {missing_names} and resubmit."
            )
        else:
            message = (
                f"For a {payload.category} claim the required documents are: "
                f"{', '.join(required)}. We received: {uploaded_names}. "
                f"Missing: {missing_names}. Please upload these and resubmit."
            )

        return VerifierResult(
            status=AgentStatus.BLOCKED,
            missing_required=missing,
            wrong_uploads=wrong_uploads,
            user_message=message,
        )

    def failure_default(self, payload: VerifierInput, exc: Exception) -> VerifierResult:
        return VerifierResult(
            status=AgentStatus.DEGRADED,
            user_message=f"Document verification could not complete: {exc}. Routed to manual review.",
        )

    def trace_payload(self, result: VerifierResult) -> dict:
        return {
            "status": result.status.value,
            "missing_required": [m.value for m in result.missing_required],
            "wrong_uploads": result.wrong_uploads,
            "user_message": result.user_message,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Quality (TC002)
# ─────────────────────────────────────────────────────────────────────────────


class QualityCheckerAgent(BaseAgent[ClassifiedDocument, QualityResult]):
    name = "QualityChecker"

    async def _run(self, doc: ClassifiedDocument) -> QualityResult:
        # Test cases provide quality_hint = "UNREADABLE" for TC002
        if doc.quality_hint == DocumentQuality.UNREADABLE:
            return QualityResult(
                file_id=doc.file_id,
                quality=DocumentQuality.UNREADABLE,
                ocr_confidence=0.05,
                needs_reupload=True,
                user_message=(
                    f"The document '{doc.file_name}' ({doc.actual_type.value}) is too "
                    f"blurry/unreadable. Please upload a clearer photo of this specific "
                    f"document and resubmit."
                ),
            )

        # Real-file path: re-run OCR confidence pass
        if doc.file_path:
            from ..ocr import extract_text
            result = extract_text(doc.file_path)
            conf = float(result.get("confidence", 0.0))
            quality = (
                DocumentQuality.UNREADABLE if conf < 0.2
                else DocumentQuality.DEGRADED if conf < 0.5
                else DocumentQuality.GOOD
            )
            needs_reupload = quality == DocumentQuality.UNREADABLE
            return QualityResult(
                file_id=doc.file_id,
                quality=quality,
                ocr_confidence=conf,
                needs_reupload=needs_reupload,
                user_message=(
                    f"The document '{doc.file_name}' is unreadable (OCR confidence {conf:.2f}). "
                    f"Please upload a clearer image and resubmit." if needs_reupload else None
                ),
            )

        # Pre-extracted (eval mode without quality_hint=UNREADABLE) — trust the data
        return QualityResult(
            file_id=doc.file_id,
            quality=DocumentQuality.GOOD,
            ocr_confidence=1.0,
        )

    def failure_default(self, doc: ClassifiedDocument, exc: Exception) -> QualityResult:
        return QualityResult(
            file_id=doc.file_id,
            quality=DocumentQuality.DEGRADED,
            ocr_confidence=0.4,
            needs_reupload=False,
        )

    def trace_payload(self, result: QualityResult) -> dict:
        return {
            "file_id": result.file_id,
            "quality": result.quality.value,
            "ocr_confidence": result.ocr_confidence,
            "needs_reupload": result.needs_reupload,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Consistency (TC003)
# ─────────────────────────────────────────────────────────────────────────────


class ConsistencyCheckerAgent(BaseAgent[list[ExtractedDocument], ConsistencyResult]):
    name = "ConsistencyChecker"

    async def _run(self, docs: list[ExtractedDocument]) -> ConsistencyResult:
        # Collect all distinct patient names found
        names = [d.patient_name for d in docs if d.patient_name]
        unique_names = list({n for n in names if n})

        if len(unique_names) <= 1:
            patient_match = True
            confidence = 1.0
            method = "trivial"
        else:
            match_result = await llm.patient_identity_match(unique_names)
            patient_match = bool(match_result.get("match", False))
            confidence = float(match_result.get("confidence", 0.5))
            method = str(match_result.get("method", "unknown"))

        # Also check total vs sum of line items (per doc)
        delta: Optional[float] = None
        for d in docs:
            if d.total_amount is not None and d.line_items:
                summed = sum(float(li.get("amount", 0) or 0) for li in d.line_items)
                if summed > 0:
                    diff = abs(d.total_amount - summed)
                    if diff > 1:  # tolerate rounding
                        delta = round(diff, 2)
                        break

        if not patient_match:
            return ConsistencyResult(
                status=AgentStatus.BLOCKED,
                patient_match=False,
                patient_names_found=unique_names,
                total_vs_lineitems_delta=delta,
                user_message=(
                    f"The documents you uploaded appear to be for different patients: "
                    f"{', '.join(unique_names)}. All documents in a single claim must be "
                    f"for the same person. Please check and resubmit."
                ),
                confidence=confidence,
            )

        status = AgentStatus.OK if delta is None else AgentStatus.DEGRADED
        msg = (
            f"Note: total on the bill differs from line-item sum by ₹{delta:.2f}."
            if delta is not None else None
        )
        return ConsistencyResult(
            status=status,
            patient_match=True,
            patient_names_found=unique_names,
            total_vs_lineitems_delta=delta,
            user_message=msg,
            confidence=confidence,
        )

    def failure_default(self, docs: list[ExtractedDocument], exc: Exception) -> ConsistencyResult:
        return ConsistencyResult(
            status=AgentStatus.DEGRADED,
            patient_match=True,  # don't block on consistency-check failure
            confidence=0.5,
            user_message=f"Consistency check degraded: {exc}",
        )

    def trace_payload(self, result: ConsistencyResult) -> dict:
        return {
            "status": result.status.value,
            "patient_match": result.patient_match,
            "patient_names_found": result.patient_names_found,
            "total_vs_lineitems_delta": result.total_vs_lineitems_delta,
            "user_message": result.user_message,
            "confidence": result.confidence,
        }
