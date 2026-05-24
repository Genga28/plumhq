"""Pydantic types used across the pipeline. These are the formal component contracts."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────


class ClaimCategory(str, Enum):
    CONSULTATION = "CONSULTATION"
    DIAGNOSTIC = "DIAGNOSTIC"
    PHARMACY = "PHARMACY"
    DENTAL = "DENTAL"
    VISION = "VISION"
    ALTERNATIVE_MEDICINE = "ALTERNATIVE_MEDICINE"


class DocumentType(str, Enum):
    PRESCRIPTION = "PRESCRIPTION"
    HOSPITAL_BILL = "HOSPITAL_BILL"
    PHARMACY_BILL = "PHARMACY_BILL"
    LAB_REPORT = "LAB_REPORT"
    DIAGNOSTIC_REPORT = "DIAGNOSTIC_REPORT"
    DENTAL_REPORT = "DENTAL_REPORT"
    DISCHARGE_SUMMARY = "DISCHARGE_SUMMARY"
    UNKNOWN = "UNKNOWN"


class DocumentQuality(str, Enum):
    GOOD = "GOOD"
    DEGRADED = "DEGRADED"
    UNREADABLE = "UNREADABLE"


class Decision(str, Enum):
    APPROVED = "APPROVED"
    PARTIAL = "PARTIAL"
    REJECTED = "REJECTED"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    BLOCKED = "BLOCKED"  # pre-decision gate failure (TC001/TC002/TC003)


class RejectionReason(str, Enum):
    WAITING_PERIOD = "WAITING_PERIOD"
    PRE_AUTH_MISSING = "PRE_AUTH_MISSING"
    PER_CLAIM_EXCEEDED = "PER_CLAIM_EXCEEDED"
    SUB_LIMIT_EXCEEDED = "SUB_LIMIT_EXCEEDED"
    YTD_EXHAUSTED = "YTD_EXHAUSTED"
    EXCLUDED_CONDITION = "EXCLUDED_CONDITION"
    EXCLUDED_PROCEDURE = "EXCLUDED_PROCEDURE"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    BELOW_MINIMUM = "BELOW_MINIMUM"
    CATEGORY_NOT_COVERED = "CATEGORY_NOT_COVERED"
    POLICY_INACTIVE = "POLICY_INACTIVE"


class AgentStatus(str, Enum):
    OK = "OK"
    BLOCKED = "BLOCKED"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


# ─────────────────────────────────────────────────────────────────────────────
# Submission (input to pipeline)
# ─────────────────────────────────────────────────────────────────────────────


class UploadedDocument(BaseModel):
    """A document as it arrives at the pipeline (file already saved to disk)."""

    file_id: str
    file_name: str
    file_path: Optional[str] = None  # absolute path on disk (None when pre-extracted)
    declared_type: Optional[DocumentType] = None  # what the user said it is
    # Used by eval.py only: bypass OCR with pre-extracted content (from test_cases.json)
    pre_extracted: Optional[dict[str, Any]] = None
    quality_hint: Optional[DocumentQuality] = None  # used by test cases


class ClaimSubmission(BaseModel):
    """Top-level input to the claims pipeline."""

    member_id: str
    policy_id: Optional[str] = None  # resolved from member if absent
    category: ClaimCategory
    claimed_amount: float = Field(gt=0)
    treatment_date: date
    submission_date: Optional[date] = None  # when the claim was submitted (eval uses treatment_date)
    hospital_name: Optional[str] = None
    documents: list[UploadedDocument]
    claims_history: list[dict[str, Any]] = Field(default_factory=list)
    simulate_component_failure: bool = False  # TC011

    @field_validator("documents")
    @classmethod
    def at_least_one_doc(cls, v: list[UploadedDocument]) -> list[UploadedDocument]:
        if not v:
            raise ValueError("At least one document is required")
        return v


# ─────────────────────────────────────────────────────────────────────────────
# Per-agent inputs / outputs
# ─────────────────────────────────────────────────────────────────────────────


class ClassifiedDocument(BaseModel):
    file_id: str
    file_name: str
    file_path: Optional[str] = None
    actual_type: DocumentType
    classification_confidence: float = Field(ge=0, le=1)
    pre_extracted: Optional[dict[str, Any]] = None
    quality_hint: Optional[DocumentQuality] = None


class VerifierResult(BaseModel):
    status: AgentStatus
    missing_required: list[DocumentType] = Field(default_factory=list)
    wrong_uploads: list[dict[str, str]] = Field(default_factory=list)  # {uploaded, required_instead}
    user_message: Optional[str] = None  # specific, names types


class QualityResult(BaseModel):
    file_id: str
    quality: DocumentQuality
    ocr_confidence: float = Field(ge=0, le=1)
    needs_reupload: bool = False
    user_message: Optional[str] = None


class ExtractedDocument(BaseModel):
    file_id: str
    actual_type: DocumentType
    quality: DocumentQuality
    raw_text: str = ""
    fields: dict[str, Any] = Field(default_factory=dict)
    # common normalized fields, optional:
    patient_name: Optional[str] = None
    doctor_name: Optional[str] = None
    doctor_registration: Optional[str] = None
    hospital_name: Optional[str] = None
    diagnosis: Optional[str] = None
    line_items: list[dict[str, Any]] = Field(default_factory=list)
    total_amount: Optional[float] = None
    document_date: Optional[date] = None
    extraction_confidence: float = Field(ge=0, le=1, default=0.0)


class ConsistencyResult(BaseModel):
    status: AgentStatus
    patient_match: bool = True
    patient_names_found: list[str] = Field(default_factory=list)
    total_vs_lineitems_delta: Optional[float] = None
    user_message: Optional[str] = None
    confidence: float = Field(ge=0, le=1, default=1.0)


class SemanticClassification(BaseModel):
    """LLM-produced semantic tags grounded in policy categories."""

    diagnosis_keys: list[str] = Field(default_factory=list)  # e.g. ["diabetes"]
    excluded_matches: list[str] = Field(default_factory=list)  # e.g. ["obesity"]
    line_item_tags: list[dict[str, Any]] = Field(default_factory=list)
    # [{description, category: "covered"|"excluded"|"cosmetic", reason}]
    network_hospital_match: Optional[str] = None  # canonical name if matched
    confidence: float = Field(ge=0, le=1, default=1.0)


class FraudSignals(BaseModel):
    same_day_count: int = 0
    monthly_count: int = 0
    is_high_value: bool = False
    duplicate_hash_match: bool = False
    alteration_detected: bool = False
    fraud_score: float = Field(ge=0, le=1, default=0.0)
    triggers: list[str] = Field(default_factory=list)


class LineItemDecision(BaseModel):
    description: str
    claimed_amount: float
    approved_amount: float
    status: str  # APPROVED | REJECTED
    reason: Optional[str] = None


class RulesResult(BaseModel):
    """Output of the deterministic rules engine — the source of truth on numbers."""

    decision: Decision
    approved_amount: float = 0.0
    pre_discount_amount: float = 0.0
    network_discount: float = 0.0
    copay_deduction: float = 0.0
    rejection_reasons: list[RejectionReason] = Field(default_factory=list)
    line_items: list[LineItemDecision] = Field(default_factory=list)
    breakdown: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1, default=1.0)


class FinalDecision(BaseModel):
    """The final claim decision returned to the user / UI."""

    claim_id: str
    decision: Decision
    approved_amount: float = 0.0
    rejection_reasons: list[RejectionReason] = Field(default_factory=list)
    user_message: str
    llm_reasoning: Optional[str] = None
    confidence: float = Field(ge=0, le=1)
    breakdown: dict[str, Any] = Field(default_factory=dict)
    line_items: list[LineItemDecision] = Field(default_factory=list)
    degraded_components: list[str] = Field(default_factory=list)
    manual_review_recommended: bool = False
    trace: list["TraceEvent"] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Trace
# ─────────────────────────────────────────────────────────────────────────────


class TraceEvent(BaseModel):
    sequence: int
    agent: str
    action: str
    status: AgentStatus
    payload: dict[str, Any] = Field(default_factory=dict)
    confidence: Optional[float] = None
    duration_ms: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Policy + Member (loaded from DB)
# ─────────────────────────────────────────────────────────────────────────────


class Member(BaseModel):
    member_id: str
    policy_id: str
    name: str
    date_of_birth: date
    gender: str
    relationship: str
    join_date: Optional[date] = None
    primary_member_id: Optional[str] = None


class Policy(BaseModel):
    """Thin wrapper over the policy_terms.json structure."""

    policy_id: str
    name: str
    config: dict[str, Any]  # the full JSON; queried via helpers in services/policy

    @property
    def opd_categories(self) -> dict[str, Any]:
        return self.config.get("opd_categories", {})

    @property
    def waiting_periods(self) -> dict[str, Any]:
        return self.config.get("waiting_periods", {})

    @property
    def exclusions(self) -> dict[str, Any]:
        return self.config.get("exclusions", {})

    @property
    def network_hospitals(self) -> list[str]:
        return self.config.get("network_hospitals", [])

    @property
    def document_requirements(self) -> dict[str, Any]:
        return self.config.get("document_requirements", {})

    @property
    def fraud_thresholds(self) -> dict[str, Any]:
        return self.config.get("fraud_thresholds", {})

    @property
    def submission_rules(self) -> dict[str, Any]:
        return self.config.get("submission_rules", {})

    @property
    def coverage(self) -> dict[str, Any]:
        return self.config.get("coverage", {})

    @property
    def pre_authorization(self) -> dict[str, Any]:
        return self.config.get("pre_authorization", {})


FinalDecision.model_rebuild()
