"""OCR + text extraction.

Routing:
  * .pdf  -> pdfplumber (fast, accurate on native PDFs). If no text extracted,
            falls through to image-OCR by rasterizing each page.
  * image -> Tesseract via pytesseract.

Both paths return (raw_text, ocr_confidence). Confidence is heuristic, not
calibrated probability, but it's monotonic with quality so the QualityChecker
gate can act on it sensibly.

All blocking I/O is wrapped in run_in_executor at the agent layer so the
pipeline stays async-friendly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from . import get_logger

log = get_logger("ocr")

# Lazy imports — these are only needed when real files are processed.
# Tests that work with pre_extracted dicts shouldn't pay the import cost.


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
_PDF_SUFFIXES = {".pdf"}


def _suffix(path: str) -> str:
    return Path(path).suffix.lower()


def is_pdf(path: str) -> bool:
    return _suffix(path) in _PDF_SUFFIXES


def is_image(path: str) -> bool:
    return _suffix(path) in _IMAGE_SUFFIXES


# ─────────────────────────────────────────────────────────────────────────────
# Public extractor (sync; agent layer wraps with executor)
# ─────────────────────────────────────────────────────────────────────────────


def extract_text(path: str) -> dict[str, Any]:
    """Return {raw_text, confidence, method}.

    Confidence is a soft signal in [0, 1]:
      * empty / very short text  -> 0.0–0.2
      * short text, low char ratio -> ~0.4
      * normal text               -> 0.7
      * clean structured text     -> 0.9+
    """
    fname = os.path.basename(path) if path else "?"
    if not os.path.exists(path):
        log.warning("extract_text: file not found: %s", path)
        return {"raw_text": "", "confidence": 0.0, "method": "missing"}

    if is_pdf(path):
        result = _extract_pdf(path)
        if not result["raw_text"].strip():
            log.info("ocr: pdfplumber found no text in %s — falling back to image OCR", fname)
            result = _extract_pdf_via_ocr(path)
        log.info("ocr: %s → method=%s, chars=%d, conf=%.2f",
                 fname, result["method"], len(result["raw_text"]), result["confidence"])
        return result

    if is_image(path):
        result = _extract_image(path)
        log.info("ocr: %s → method=%s, chars=%d, conf=%.2f",
                 fname, result["method"], len(result["raw_text"]), result["confidence"])
        return result

    # Unknown file type — try reading as text
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return {"raw_text": text, "confidence": _confidence_from_text(text), "method": "text"}
    except Exception as e:
        log.warning("ocr: could not read %s as text: %s", fname, e)
        return {"raw_text": "", "confidence": 0.0, "method": "unknown"}


# ─────────────────────────────────────────────────────────────────────────────
# PDF path
# ─────────────────────────────────────────────────────────────────────────────


def _extract_pdf(path: str) -> dict[str, Any]:
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        return {"raw_text": "", "confidence": 0.0, "method": "pdfplumber_unavailable"}

    text_parts: list[str] = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                if t:
                    text_parts.append(t)
    except Exception as e:
        return {"raw_text": "", "confidence": 0.0, "method": f"pdfplumber_error:{e}"}

    raw = "\n".join(text_parts)
    return {"raw_text": raw, "confidence": _confidence_from_text(raw), "method": "pdfplumber"}


def _extract_pdf_via_ocr(path: str) -> dict[str, Any]:
    """Fallback: rasterize PDF pages and OCR each. Used when pdfplumber finds no text."""
    try:
        from pdf2image import convert_from_path  # type: ignore
    except ImportError:
        # pdf2image not installed — we just return empty rather than crash.
        return {"raw_text": "", "confidence": 0.0, "method": "pdf_ocr_unavailable"}

    try:
        images = convert_from_path(path, dpi=200)
    except Exception as e:
        return {"raw_text": "", "confidence": 0.0, "method": f"pdf_rasterize_error:{e}"}

    text_parts: list[str] = []
    for img in images:
        # save to temp + run tesseract
        text_parts.append(_ocr_image(img))
    raw = "\n".join(text_parts)
    return {"raw_text": raw, "confidence": _confidence_from_text(raw) * 0.85, "method": "pdf_ocr"}


# ─────────────────────────────────────────────────────────────────────────────
# Image path
# ─────────────────────────────────────────────────────────────────────────────


def _extract_image(path: str) -> dict[str, Any]:
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        return {"raw_text": "", "confidence": 0.0, "method": "pillow_unavailable"}

    try:
        img = Image.open(path)
    except Exception as e:
        return {"raw_text": "", "confidence": 0.0, "method": f"image_open_error:{e}"}

    raw = _ocr_image(img)
    # Image-based extraction is intrinsically lower confidence than native PDF.
    return {"raw_text": raw, "confidence": _confidence_from_text(raw) * 0.9, "method": "tesseract"}


def _ocr_image(img: Any) -> str:
    try:
        import pytesseract  # type: ignore
    except ImportError:
        return ""
    tesseract_cmd = os.environ.get("TESSERACT_CMD")
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    try:
        return pytesseract.image_to_string(img) or ""
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic confidence
# ─────────────────────────────────────────────────────────────────────────────


def _confidence_from_text(text: str) -> float:
    """Cheap proxy for OCR quality."""
    t = text.strip()
    if not t:
        return 0.0
    n = len(t)
    if n < 20:
        return 0.15
    if n < 80:
        return 0.4
    # ratio of alpha characters — pure garbage OCR is dominated by symbols
    alpha = sum(1 for c in t if c.isalpha())
    ratio = alpha / max(n, 1)
    base = 0.6 + 0.3 * min(1.0, ratio)
    # boost a bit if the text contains common keywords we expect
    keywords = ("date", "patient", "amount", "total", "doctor", "diagnosis",
                "rs", "₹", "mg", "tab", "hospital", "clinic")
    hits = sum(1 for k in keywords if k.lower() in t.lower())
    base = min(1.0, base + 0.02 * hits)
    return round(base, 3)
