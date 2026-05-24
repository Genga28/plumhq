"""SemanticClassifierAgent: maps free-text extraction into policy categories.

Why this exists: the rules engine needs structured categories ("diabetes",
"obesity_treatment", "covered" / "excluded" per line item) to make decisions.
Free-text diagnoses like "T2DM" or "Bariatric Consultation" need semantic
understanding — that's what the LLM does here. Falls back to keyword tables
when no LLM key is configured (see services/llm.py).
"""

from __future__ import annotations

from typing import Any

from .. import llm
from ..models import (
    ExtractedDocument,
    Policy,
    SemanticClassification,
)
from .base import BaseAgent


class SemanticInput:
    def __init__(self, docs: list[ExtractedDocument], policy: Policy, hospital_name: str | None) -> None:
        self.docs = docs
        self.policy = policy
        self.hospital_name = hospital_name


class SemanticClassifierAgent(BaseAgent[SemanticInput, SemanticClassification]):
    name = "SemanticClassifier"

    async def _run(self, payload: SemanticInput) -> SemanticClassification:
        # Aggregate everything the LLM should look at
        diagnosis = next((d.diagnosis for d in payload.docs if d.diagnosis), "") or ""
        treatment = ""
        line_items: list[dict[str, Any]] = []
        for d in payload.docs:
            t = d.fields.get("treatment")
            if t and not treatment:
                treatment = str(t)
            for li in d.line_items or []:
                line_items.append({"description": li.get("description", ""), "amount": li.get("amount", 0)})

        # Resolve hospital: prefer submission-provided, else extracted
        hospital = payload.hospital_name or next(
            (d.hospital_name for d in payload.docs if d.hospital_name), None
        )

        extracted_blob = {
            "diagnosis": diagnosis,
            "treatment": treatment,
            "line_items": line_items,
            "hospital_name": hospital,
        }

        result = await llm.semantic_classify(extracted_blob, payload.policy.config)
        return SemanticClassification(
            diagnosis_keys=result.get("diagnosis_keys", []),
            excluded_matches=result.get("excluded_matches", []),
            line_item_tags=result.get("line_item_tags", []),
            network_hospital_match=result.get("network_hospital_match"),
            confidence=float(result.get("confidence", 0.7)),
        )

    def failure_default(self, payload: SemanticInput, exc: Exception) -> SemanticClassification:
        return SemanticClassification(
            diagnosis_keys=[],
            excluded_matches=[],
            line_item_tags=[],
            confidence=0.4,
        )

    def trace_payload(self, result: SemanticClassification) -> dict:
        return {
            "diagnosis_keys": result.diagnosis_keys,
            "excluded_matches": result.excluded_matches,
            "network_hospital_match": result.network_hospital_match,
            "line_item_tags": result.line_item_tags,
            "confidence": result.confidence,
        }
