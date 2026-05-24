"""DocumentClassifierAgent: detects the actual type of each uploaded document.

Strategy:
  1. If the user / test gave a declared_type, trust it unless the content
     strongly contradicts it.
  2. Else infer from filename keywords (prescription, bill, lab, pharmacy...).
  3. Else infer from extracted text content (cheap regex pass).

This runs BEFORE the OCR extractor — the classifier only peeks at the file
name + a tiny content sample so we can do the doc-verification gate (TC001)
cheaply.
"""

from __future__ import annotations

import re
from typing import Optional

from ..models import (
    ClassifiedDocument,
    DocumentType,
    UploadedDocument,
)
from ..ocr import extract_text, is_pdf, is_image
from .base import BaseAgent


# filename keyword -> type (most-specific first)
# NOTE: bare "report" intentionally dropped — too many docs have "report" in
# the filename without being lab reports (e.g. "doctor_report.pdf"). Use
# "lab_report" or "lab" if you want lab-specific matching.
_FILENAME_HINTS: list[tuple[str, DocumentType]] = [
    ("pharmacy_bill", DocumentType.PHARMACY_BILL),
    ("pharmacy", DocumentType.PHARMACY_BILL),
    ("hospital_bill", DocumentType.HOSPITAL_BILL),
    ("bill", DocumentType.HOSPITAL_BILL),
    ("invoice", DocumentType.HOSPITAL_BILL),
    ("receipt", DocumentType.HOSPITAL_BILL),
    ("prescription", DocumentType.PRESCRIPTION),
    ("rx", DocumentType.PRESCRIPTION),
    ("dental_report", DocumentType.DENTAL_REPORT),
    ("dental", DocumentType.DENTAL_REPORT),
    ("diagnostic", DocumentType.DIAGNOSTIC_REPORT),
    ("discharge", DocumentType.DISCHARGE_SUMMARY),
    ("lab_report", DocumentType.LAB_REPORT),
    ("lab", DocumentType.LAB_REPORT),
]


# Pattern order matters — first match wins.
# Strategy: check the MOST SPECIFIC documents (with unambiguous markers) first.
# A prescription mentioning "CBC" in an Investigations section should NOT be
# misclassified as a lab report, so the prescription pattern runs first.
_CONTENT_PATTERNS: list[tuple[str, DocumentType]] = [
    # Pharmacy bill — drug license + MRP + batch is a very specific combo
    (r"\b(drug\s*lic(?:ence|ense)?|mrp\b.*\bbatch|\bbatch\s*no\s*[:.])",
     DocumentType.PHARMACY_BILL),

    # Prescription — state-coded doctor registration is the strongest signal
    # (KA/45678/2015 etc.), plus prescription-specific section headers.
    (r"\b[A-Z]{2}/\d{2,6}/\d{4}\b"          # state reg number
     r"|\brx\s*[:.]"                         # "Rx:" header
     r"|^rx\s*[:.]"
     r"|chief\s*complaint\s*[:.]"
     r"|\bfollow\s*-?\s*up\s*[:.]"
     r"|\btab\.?\s+[A-Z]"                    # "Tab. Paracetamol"
     r"|\bcap\.?\s+[A-Z]"                    # "Cap. Amoxicillin"
     r"|\bsig\b"                             # prescription shorthand "sig"
     r"|\d\s*mg\b\s*[-—]\s*\d"               # "650mg — 1-1-1"
     r"|\d-\d-\d\s*x\s*\d",                  # "1-0-1 x 5 days" dosing
     DocumentType.PRESCRIPTION),

    # Lab report — NABL/Pathologist/Sample ID/Reference Range are lab-exclusive.
    # Removed bare "cbc", "hemoglobin", "wbc count" because those frequently
    # appear in prescriptions' Investigations section.
    (r"\b(nabl\b|pathologist|sample\s*id\s*[:.]|reference\s*range|"
     r"normal\s*range\s*[\d(])",
     DocumentType.LAB_REPORT),

    # Hospital bill — GSTIN/Bill No/Subtotal/Payment Mode are bill-specific.
    (r"\b(gstin|bill\s*no\s*[:.]|sub\s*total\s*[:.]|"
     r"received\s*by|payment\s*mode|consultation\s*fee\s+\d|"
     r"description\s+(?:qty|rate|amount))",
     DocumentType.HOSPITAL_BILL),

    (r"\bdischarge\s*summary\b", DocumentType.DISCHARGE_SUMMARY),
    (r"\bdental\s*(report|x-?ray|treatment)", DocumentType.DENTAL_REPORT),
]


class DocumentClassifierAgent(BaseAgent[UploadedDocument, ClassifiedDocument]):
    name = "DocumentClassifier"

    async def _run(self, doc: UploadedDocument) -> ClassifiedDocument:
        # Priority of signals (strongest first):
        #   1. User explicitly declared a type     -> trust them (0.95)
        #   2. Pre-extracted JSON + declared type  -> eval mode (1.0)
        #   3. Filename keywords AND content agree -> high confidence (0.9)
        #   4. Content-based OCR pattern match     -> moderate (0.75)
        #   5. Filename-only match                 -> low-moderate (0.6)
        #   6. Nothing matched                     -> UNKNOWN (0.2)

        # 1. Pre-extracted (eval mode): we already know the type
        if doc.pre_extracted is not None and doc.declared_type:
            return ClassifiedDocument(
                file_id=doc.file_id,
                file_name=doc.file_name,
                file_path=doc.file_path,
                actual_type=doc.declared_type,
                classification_confidence=1.0,
                pre_extracted=doc.pre_extracted,
                quality_hint=doc.quality_hint,
            )

        # 2. User explicitly told us the type (UI dropdown).
        # Trust them by default, BUT cross-check against OCR content. If the
        # content strongly indicates a different type, the user likely
        # mis-labeled — we override and downgrade confidence so the trace
        # records the disagreement.
        if doc.declared_type:
            content_type = None
            if doc.file_path and (is_pdf(doc.file_path) or is_image(doc.file_path)):
                result = extract_text(doc.file_path)
                content_type = _from_content(result.get("raw_text", ""))

            if content_type and content_type != doc.declared_type:
                # Mismatch: trust content over the user, but flag the conflict.
                return ClassifiedDocument(
                    file_id=doc.file_id,
                    file_name=doc.file_name,
                    file_path=doc.file_path,
                    actual_type=content_type,
                    classification_confidence=0.6,
                    pre_extracted=doc.pre_extracted,
                    quality_hint=doc.quality_hint,
                )

            return ClassifiedDocument(
                file_id=doc.file_id,
                file_name=doc.file_name,
                file_path=doc.file_path,
                actual_type=doc.declared_type,
                classification_confidence=0.95,
                pre_extracted=doc.pre_extracted,
                quality_hint=doc.quality_hint,
            )

        # 3. Filename keyword scan (cheap, common case)
        inferred = _from_filename(doc.file_name)
        confidence = 0.6 if inferred else 0.0

        # 4. Content-based via OCR — the fallback for filenames like
        # "prs.jpg", "8819.jpg", or other meaningless names. We extract text
        # and look for prescription/bill/lab signatures in the content.
        if doc.file_path and (is_pdf(doc.file_path) or is_image(doc.file_path)):
            result = extract_text(doc.file_path)
            content_type = _from_content(result.get("raw_text", ""))
            if content_type and inferred is None:
                inferred = content_type
                confidence = 0.75
            elif content_type and content_type == inferred:
                confidence = 0.9  # filename + content agree
            elif content_type and content_type != inferred:
                inferred = content_type  # content wins
                confidence = 0.7

        if inferred is None:
            inferred = DocumentType.UNKNOWN
            confidence = 0.2

        return ClassifiedDocument(
            file_id=doc.file_id,
            file_name=doc.file_name,
            file_path=doc.file_path,
            actual_type=inferred,
            classification_confidence=confidence,
            pre_extracted=doc.pre_extracted,
            quality_hint=doc.quality_hint,
        )

    def failure_default(self, doc: UploadedDocument, exc: Exception) -> ClassifiedDocument:
        return ClassifiedDocument(
            file_id=doc.file_id,
            file_name=doc.file_name,
            file_path=doc.file_path,
            actual_type=doc.declared_type or DocumentType.UNKNOWN,
            classification_confidence=0.2,
            pre_extracted=doc.pre_extracted,
            quality_hint=doc.quality_hint,
        )

    def trace_payload(self, result: ClassifiedDocument) -> dict:
        return {
            "file_name": result.file_name,
            "actual_type": result.actual_type.value,
            "classification_confidence": result.classification_confidence,
            "via": "user_declared" if result.classification_confidence >= 0.95 else
                   "filename+content" if result.classification_confidence >= 0.85 else
                   "content_ocr" if result.classification_confidence >= 0.7 else
                   "filename_only" if result.classification_confidence >= 0.6 else
                   "fallback",
        }


def _from_filename(name: str) -> Optional[DocumentType]:
    n = name.lower()
    for needle, t in _FILENAME_HINTS:
        if needle in n:
            return t
    return None


def _from_content(text: str) -> Optional[DocumentType]:
    if not text:
        return None
    t = text.lower()
    for pattern, dt in _CONTENT_PATTERNS:
        if re.search(pattern, t):
            return dt
    return None
