"""ExtractorAgent: OCR text -> structured ExtractedDocument.

Two modes:
  1. Pre-extracted (eval): the test case provided a `content` dict — we just
     normalize it into ExtractedDocument fields. No OCR runs.
  2. Real file: extract_text() -> raw text -> regex/heuristic parser -> fields.

Heuristics are conservative; anything not confidently extracted is left None
and surfaced via the trace + extraction_confidence. Downstream agents
(semantic classifier, consistency checker) handle the imperfection.
"""

from __future__ import annotations

import asyncio
import re
from datetime import date, datetime
from typing import Any, Optional

from .. import get_logger, llm
from ..models import (
    ClassifiedDocument,
    DocumentQuality,
    DocumentType,
    ExtractedDocument,
)
from ..ocr import extract_text
from .base import BaseAgent

log = get_logger("extractor")


_DATE_PATTERNS = [
    "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d-%b-%Y", "%d %b %Y", "%d-%B-%Y",
    "%d %B %Y", "%d.%m.%Y", "%Y/%m/%d",
]


class ExtractorAgent(BaseAgent[ClassifiedDocument, ExtractedDocument]):
    name = "Extractor"

    async def _run(self, doc: ClassifiedDocument) -> ExtractedDocument:
        # Path 1: pre-extracted content (eval / test data)
        if doc.pre_extracted is not None:
            return _from_pre_extracted(doc)

        # Path 2: real file via OCR (CPU-bound -> threadpool)
        raw = await asyncio.to_thread(extract_text, doc.file_path or "")
        text = raw.get("raw_text", "")
        conf = float(raw.get("confidence", 0.0))

        fields = _parse_text(text, doc.actual_type)
        line_items = fields.get("line_items") or []
        total = fields.get("total_amount")

        # ── LLM fallback for line items ──
        # The regex parser handles standard table layouts but can miss messier
        # real-world bills. If we got nothing and the document type is a bill,
        # ask Gemini to pull line items out of the raw OCR text. Empty list
        # comes back when Gemini is unavailable / quota-exhausted, so the
        # pipeline still works without it.
        is_bill = doc.actual_type in (
            DocumentType.HOSPITAL_BILL,
            DocumentType.PHARMACY_BILL,
        )
        if is_bill and not line_items and text and len(text) > 100:
            log.info("Extractor: 0 line items from regex on %s, trying LLM fallback",
                     doc.file_name)
            llm_items = await llm.extract_line_items(text)
            if llm_items:
                line_items = llm_items
                fields["line_items"] = llm_items
                fields["line_items_source"] = "llm_fallback"

        quality = (
            DocumentQuality.UNREADABLE if conf < 0.2
            else DocumentQuality.DEGRADED if conf < 0.5
            else DocumentQuality.GOOD
        )

        return ExtractedDocument(
            file_id=doc.file_id,
            actual_type=doc.actual_type,
            quality=quality,
            raw_text=text[:4000],
            fields=fields,
            patient_name=fields.get("patient_name"),
            doctor_name=fields.get("doctor_name"),
            doctor_registration=fields.get("doctor_registration"),
            hospital_name=fields.get("hospital_name"),
            diagnosis=fields.get("diagnosis"),
            line_items=line_items,
            total_amount=total,
            document_date=fields.get("document_date"),
            extraction_confidence=conf,
        )

    def failure_default(self, doc: ClassifiedDocument, exc: Exception) -> ExtractedDocument:
        return ExtractedDocument(
            file_id=doc.file_id,
            actual_type=doc.actual_type,
            quality=DocumentQuality.DEGRADED,
            extraction_confidence=0.2,
            fields={"extraction_error": str(exc)},
        )

    def trace_payload(self, result: ExtractedDocument) -> dict:
        return {
            "file_id": result.file_id,
            "actual_type": result.actual_type.value,
            "quality": result.quality.value,
            "extraction_confidence": result.extraction_confidence,
            "patient_name": result.patient_name,
            "doctor_name": result.doctor_name,
            "hospital_name": result.hospital_name,
            "diagnosis": result.diagnosis,
            "document_date": result.document_date.isoformat() if result.document_date else None,
            "line_items_count": len(result.line_items or []),
            "line_items": result.line_items or [],
            "total_amount": result.total_amount,
            "line_items_source": result.fields.get("line_items_source", "regex"),
            "raw_text": (result.raw_text or "")[:5000],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Pre-extracted (eval) path
# ─────────────────────────────────────────────────────────────────────────────


def _from_pre_extracted(doc: ClassifiedDocument) -> ExtractedDocument:
    c = doc.pre_extracted or {}
    line_items = c.get("line_items") or []
    total = c.get("total") or c.get("total_amount")
    quality = doc.quality_hint or DocumentQuality.GOOD
    diag = c.get("diagnosis") or c.get("treatment")
    return ExtractedDocument(
        file_id=doc.file_id,
        actual_type=doc.actual_type,
        quality=quality,
        raw_text="",
        fields=c,
        patient_name=c.get("patient_name"),
        doctor_name=c.get("doctor_name"),
        doctor_registration=c.get("doctor_registration"),
        hospital_name=c.get("hospital_name"),
        diagnosis=diag,
        line_items=line_items,
        total_amount=float(total) if total is not None else None,
        document_date=_parse_date(c.get("date")) if c.get("date") else None,
        extraction_confidence=0.95,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Real-text parser
# ─────────────────────────────────────────────────────────────────────────────


def _parse_text(text: str, doc_type: DocumentType) -> dict[str, Any]:
    """Heuristic extractor for OCR'd Indian medical documents.

    Targets the formats described in sample_documents_guide.md:
      * Prescriptions with "Patient: ...", "Diagnosis: ...", "Reg. No: ..."
      * Hospital/clinic bills with "DESCRIPTION ... AMOUNT" tabular line items
      * Pharmacy bills with "MEDICINE BATCH EXP QTY MRP AMT" lines
      * Lab reports with test/result columns
    """
    if not text:
        return {}

    fields: dict[str, Any] = {}

    # ── Patient name ────────────────────────────────────────────────────────
    # Common label variants. Stop at common end-of-segment cues so we don't
    # capture the next field as part of the name.
    m = re.search(
        r"(?:patient\s*(?:name)?|name\s*of\s*patient)\s*[:\-]\s*"
        r"([A-Z][A-Za-z. ]{2,40}?)"
        r"\s*(?:\n|date\b|age\b|gender\b|sex\b|dob\b|referring\b|dr\.?|ref\.?\b|\s{3,})",
        text, re.I,
    )
    if m:
        fields["patient_name"] = _clean(m.group(1))

    # ── Doctor name ─────────────────────────────────────────────────────────
    # Match a "Dr." prefix and capture the name up to credentials (MBBS, MD) or end of line.
    m = re.search(
        r"\b(Dr\.?\s+[A-Z][A-Za-z. ]{2,40}?)"
        r"(?=,?\s*(?:MBBS|MD|MS|DM|DGO|DNB|BAMS|BDS|PhD|Reg|MRCP|Ph\.?D|FRCS|\(|\n|$))",
        text,
    )
    if m:
        fields["doctor_name"] = _clean(m.group(1))

    # ── Doctor registration number (state codes from sample_documents_guide) ─
    m = re.search(r"\b(?:[A-Z]{2}|AYUR/[A-Z]{2})/\d{2,6}/\d{4}\b", text)
    if m:
        fields["doctor_registration"] = m.group(0)

    # ── Hospital / clinic / pharmacy ────────────────────────────────────────
    # Greedy match to the LAST recognized suffix on the line, so
    # "City Medical Centre" doesn't truncate to "City Medical".
    m = re.search(
        r"^\s*([A-Z][A-Za-z0-9&. ]{2,60}\b"
        r"(?:Hospital|Hospitals|Clinic|Centre|Center|Diagnostics|Pharmacy|"
        r"Laboratory|Healthcare|Lab))\b",
        text, re.M | re.I,
    )
    if m:
        fields["hospital_name"] = _clean(m.group(1))

    # ── Diagnosis ───────────────────────────────────────────────────────────
    # Prefer an explicit "Diagnosis:" / "Impression:" line. "Chief Complaint"
    # is the patient's stated symptom, NOT the doctor's diagnosis — don't
    # confuse the two.
    m = re.search(
        r"(?:diagnosis|impression)\s*[:\-]\s*([^\n]{2,120})",
        text, re.I,
    )
    if m:
        fields["diagnosis"] = _clean(m.group(1))
    else:
        # Fallback: chief complaint, if no explicit diagnosis was given
        m = re.search(r"chief\s*complaint\s*[:\-]\s*([^\n]{2,120})", text, re.I)
        if m:
            fields["diagnosis"] = _clean(m.group(1))

    # ── Treatment (alternative-medicine path uses this key) ────────────────
    m = re.search(r"(?:treatment|procedure|therapy)\s*[:\-]\s*([^\n]{2,120})", text, re.I)
    if m:
        fields["treatment"] = _clean(m.group(1))

    # ── Date ────────────────────────────────────────────────────────────────
    m = re.search(
        r"(?:bill\s*date|date\s*of\s*(?:bill|visit|treatment)|date)\s*[:\-]?\s*"
        r"(\d{1,4}[-/. ][A-Za-z\d]{1,4}[-/. ]\d{2,4})",
        text, re.I,
    )
    if m:
        d = _parse_date(m.group(1))
        if d:
            fields["document_date"] = d

    # ── Tests ordered (used by pre-auth check, e.g. MRI) ───────────────────
    m = re.search(r"(?:investigations?|tests?\s*ordered|advised)\s*[:\-]\s*([^\n]{2,200})",
                  text, re.I)
    if m:
        tests = [t.strip() for t in re.split(r"[,;]", m.group(1)) if t.strip()]
        if tests:
            fields["tests_ordered"] = tests

    # ── Line items + total ──────────────────────────────────────────────────
    line_items = _parse_line_items(text)
    if line_items:
        fields["line_items"] = line_items

    # Total handles "Total Amount", "Grand Total", "Net Amount", "Net Payable", "Total"
    m = re.search(
        r"(?:total\s*amount|grand\s*total|net\s*amount|net\s*payable|total)"
        r"\s*[:\-]?\s*(?:rs\.?|inr|₹)?\s*([0-9][0-9,]*(?:\.\d{1,2})?)",
        text, re.I,
    )
    if m:
        try:
            fields["total_amount"] = float(m.group(1).replace(",", ""))
        except ValueError:
            pass

    return fields


# Lines whose description we should NEVER treat as a billable line item
_BAD_DESC = {
    "subtotal", "sub total", "sub-total", "discount", "gst", "tax", "cgst", "sgst",
    "igst", "total", "total amount", "grand total", "net amount", "net payable",
    "amount", "description", "qty", "rate", "qty rate amount", "test name",
    "medicine", "items", "patient", "patient name", "patient name age",
    "bill", "bill no", "date", "received by", "remarks", "signature",
}


def _parse_line_items(text: str) -> list[dict[str, Any]]:
    """Extract billable line items: tabular rows with a description + an amount.

    Handles:
      * Description with parentheses, hyphens, slashes, hash markers
        (e.g. "Root Canal Treatment (Tooth #36)").
      * Indian-format amounts with thousand separators (e.g. "8,000.00").
      * Multi-column rows (Qty / Rate / Amount) — we capture the LAST number.
    """
    # A currency-style number: digits, optional thousand-comma groups,
    # optional .XX decimal. E.g. matches "8,000.00", "12345", "1500.50".
    NUMBER = r"\d+(?:,\d{3})*(?:\.\d{1,2})?"

    pattern = re.compile(
        rf"^(?P<desc>[A-Za-z][A-Za-z0-9 #()\-,./&%'\"]+?)"
        rf"(?:\s+{NUMBER}){{0,3}}"        # 0-3 optional intermediate columns
        rf"\s+(?P<amt>{NUMBER})\s*$"
    )

    items: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.rstrip()
        if not line or len(line) < 6:
            continue
        m = pattern.match(line)
        if not m:
            continue
        desc = _clean(m.group("desc"))
        if not desc or len(desc) < 3:
            continue
        normalized = re.sub(r"[^a-z ]", "", desc.lower()).strip()
        if normalized in _BAD_DESC:
            continue
        if any(b in normalized for b in ("total", "subtotal", "gst ", "discount", " tax")):
            continue
        try:
            amt = float(m.group("amt").replace(",", ""))
        except ValueError:
            continue
        if amt <= 0:
            continue
        items.append({"description": desc, "amount": amt})
    return items


def _clean(s: str) -> str:
    """Tidy a captured string — collapse whitespace and trim punctuation."""
    return re.sub(r"\s+", " ", s).strip(" .,:-")


def _parse_date(s: Any) -> Optional[date]:
    if s is None:
        return None
    if isinstance(s, date):
        return s
    s = str(s).strip()
    for fmt in _DATE_PATTERNS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None
