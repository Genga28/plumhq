"""Gemini wrapper — narrow API for the three places the LLM is used.

Design:
  * Three public coroutines, one per call site:
      - patient_identity_match(names)             -> {match, canonical_name, confidence}
      - semantic_classify(extracted, policy)      -> dict matching SemanticClassification
      - generate_reasoning(decision_context)      -> str (human-readable)
  * Each function has a deterministic fallback when GEMINI_API_KEY is missing
    or the API errors. The pipeline never blocks on LLM failure (TC011).
  * Output is parsed as JSON and validated against the Pydantic model; on
    parse failure we fall back rather than crash.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Optional

from . import get_logger

log = get_logger("llm")

try:
    from google import genai  # type: ignore
    from google.genai import types as genai_types  # type: ignore
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


# Configurable via GEMINI_MODEL env var. Default chosen for the most generous
# free-tier daily allowance at time of writing. If the primary model 429s
# (quota exhausted), we automatically try the fallbacks below in order.
_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
_FALLBACK_MODELS = [
    m for m in [
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash-lite",
    ]
    if m != _MODEL
]


def _is_quota_error(exc: Exception) -> bool:
    """Detect Google's 429/RESOURCE_EXHAUSTED for cleaner logging."""
    s = str(exc).lower()
    return "429" in s or "resource_exhausted" in s or "quota" in s


def _client() -> Optional[Any]:
    if not _GENAI_AVAILABLE:
        return None
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    try:
        return genai.Client(api_key=key)
    except Exception:
        return None


def is_available() -> bool:
    return _client() is not None


# ─────────────────────────────────────────────────────────────────────────────
# 1. Patient identity match (cross-document)
# ─────────────────────────────────────────────────────────────────────────────


async def patient_identity_match(names: list[str]) -> dict[str, Any]:
    """Decide if a set of patient names found across documents refer to the same person.

    Falls back to RapidFuzz similarity when no LLM is configured. Result is
    always shaped the same way.
    """
    names = [n for n in (n.strip() for n in names) if n]
    if len(names) < 2:
        return {"match": True, "canonical_name": names[0] if names else "", "confidence": 1.0,
                "method": "trivial"}

    cli = _client()
    if cli is None:
        log.info("patient_identity_match: no Gemini key, using rapidfuzz fallback for %s", names)
        return _patient_match_fuzzy(names)
    log.info("patient_identity_match: calling Gemini for %s", names)

    prompt = (
        "Decide whether these patient names from different medical documents refer to the SAME PERSON.\n"
        "Indian names may have initials, missing surnames, or spelling variations. Treat 'Rajesh Kumar' "
        "and 'R. Kumar' as the same; treat 'Rajesh Kumar' and 'Arjun Mehta' as different people.\n\n"
        f"Names: {json.dumps(names)}\n\n"
        "Respond as STRICT JSON: "
        '{"match": true|false, "canonical_name": "<best canonical name>", "confidence": 0.0-1.0, "reason": "..."}'
    )

    try:
        text = await _call_text(cli, prompt)
        data = _parse_json(text)
        if not isinstance(data, dict) or "match" not in data:
            raise ValueError("bad shape")
        log.info("patient_identity_match: gemini result match=%s conf=%.2f",
                 data["match"], data.get("confidence", 0.8))
        return {
            "match": bool(data["match"]),
            "canonical_name": str(data.get("canonical_name", names[0])),
            "confidence": float(data.get("confidence", 0.8)),
            "reason": str(data.get("reason", "")),
            "method": "gemini",
        }
    except Exception as e:
        if _is_quota_error(e):
            log.warning("patient_identity_match: gemini quota exhausted, using rapidfuzz")
        else:
            log.warning("patient_identity_match: gemini failed (%s: %s), using rapidfuzz",
                        type(e).__name__, str(e)[:200])
        return _patient_match_fuzzy(names)


def _patient_match_fuzzy(names: list[str]) -> dict[str, Any]:
    """Deterministic fallback using token-set similarity."""
    try:
        from rapidfuzz import fuzz
    except ImportError:
        # last-resort: case-insensitive equality
        same = len({n.lower() for n in names}) == 1
        return {"match": same, "canonical_name": names[0], "confidence": 1.0 if same else 0.0,
                "method": "exact_fallback"}

    ref = names[0]
    scores = [fuzz.token_set_ratio(ref, n) for n in names[1:]]
    min_score = min(scores) if scores else 100
    match = min_score >= 75  # token-set ratio above 75 = likely same person
    return {
        "match": match,
        "canonical_name": ref,
        "confidence": round(min_score / 100, 2),
        "reason": f"token_set_ratio min={min_score}",
        "method": "rapidfuzz",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Semantic classification (diagnosis + line-items -> policy categories)
# ─────────────────────────────────────────────────────────────────────────────


_DIAGNOSIS_KEYWORDS = {
    "diabetes": ["diabetes", "t2dm", "type 2 dm", "type ii diabetes", "dm", "hyperglycemia"],
    "hypertension": ["hypertension", "htn", "high bp", "high blood pressure"],
    "thyroid_disorders": ["thyroid", "hypothyroid", "hyperthyroid"],
    "joint_replacement": ["joint replacement", "knee replacement", "hip replacement"],
    "maternity": ["maternity", "pregnancy", "antenatal", "postnatal"],
    "mental_health": ["depression", "anxiety", "psychiatric", "mental health"],
    "obesity_treatment": ["obesity", "bariatric", "morbid obesity", "weight loss", "diet plan",
                          "nutrition program", "diet program"],
    "hernia": ["hernia"],
    "cataract": ["cataract"],
}

# Diagnosis-level exclusion keywords (whole claim non-coverable).
# Note "cosmetic" by itself is intentionally NOT here — it's too easy to match
# secondary notes like "cosmetic concern alongside pulpitis". For diagnosis-
# level cosmetic exclusion we require unambiguous procedure language.
_EXCLUDED_KEYWORDS = {
    "obesity": ["bariatric", "morbid obesity", "weight loss program", "diet program",
                "nutrition program", "personalised diet"],
    "cosmetic": ["cosmetic procedure", "cosmetic surgery", "cosmetic treatment",
                 "aesthetic procedure", "aesthetic surgery"],
    "experimental": ["experimental"],
    "substance_abuse": ["substance abuse", "deaddiction", "rehab"],
    "infertility": ["infertility", "ivf", "assisted reproduction"],
    "self_inflicted": ["self-inflicted", "self inflicted", "suicide attempt"],
}

# Separate keyword set used ONLY for tagging individual line items.
# Here single words like "whitening" or "cosmetic" are appropriate, because
# a line item description ("Teeth Whitening", "Cosmetic Dental Procedure")
# is the procedure name itself — there's no surrounding context to confuse.
_LINE_ITEM_EXCLUDED_KEYWORDS = {
    "cosmetic": ["teeth whitening", "whitening", "veneers", "lasik", "refractive",
                 "cosmetic", "aesthetic", "bleaching"],
    "obesity":  ["bariatric", "obesity", "weight loss", "diet program",
                 "nutrition program", "personalised diet"],
}


async def semantic_classify(extracted: dict[str, Any], policy_config: dict[str, Any]) -> dict[str, Any]:
    """Tag the diagnosis + each line item against the policy's known categories.

    LLM gives semantic understanding (e.g. "Bariatric Consultation" -> obesity).
    Falls back to keyword matching so the pipeline works without a key.
    """
    cli = _client()
    if cli is None:
        log.info("semantic_classify: no Gemini key, using keyword fallback")
        return _semantic_keywords(extracted, policy_config)
    log.info(
        "semantic_classify: calling Gemini · diagnosis=%r line_items=%d hospital=%r",
        extracted.get("diagnosis"),
        len(extracted.get("line_items", []) or []),
        extracted.get("hospital_name"),
    )

    exclusion_conditions = policy_config.get("exclusions", {}).get("conditions", [])
    dental_exclusions = policy_config.get("exclusions", {}).get("dental_exclusions", [])
    vision_exclusions = policy_config.get("exclusions", {}).get("vision_exclusions", [])
    waiting_keys = list(policy_config.get("waiting_periods", {}).get("specific_conditions", {}).keys())

    prompt = (
        "You classify Indian health insurance claim data against a known policy. "
        "Be CONSERVATIVE: only flag something as excluded when the diagnosis or "
        "line-item DIRECTLY and CLEARLY matches one of the listed exclusions. "
        "If in doubt, do NOT exclude. Normal diagnostic tests (MRI, CT scan, "
        "X-ray, blood tests, etc.) are NEVER excluded — only the specific "
        "conditions listed below are.\n\n"
        f"POLICY EXCLUDED CONDITIONS (the ONLY allowed values for excluded_matches): "
        f"{json.dumps(exclusion_conditions)}\n"
        f"POLICY DENTAL EXCLUSIONS: {json.dumps(dental_exclusions)}\n"
        f"POLICY VISION EXCLUSIONS: {json.dumps(vision_exclusions)}\n"
        f"WAITING-PERIOD KEYS (the ONLY allowed values for diagnosis_keys): "
        f"{json.dumps(waiting_keys)}\n"
        f"NETWORK HOSPITALS: {json.dumps(policy_config.get('network_hospitals', []))}\n\n"
        f"CLAIM DATA:\n"
        f"  diagnosis: {extracted.get('diagnosis')}\n"
        f"  treatment: {extracted.get('treatment')}\n"
        f"  hospital: {extracted.get('hospital_name')}\n"
        f"  line_items: {json.dumps(extracted.get('line_items', []))}\n\n"
        "Rules:\n"
        " - excluded_matches MUST be a subset of POLICY EXCLUDED CONDITIONS (exact strings).\n"
        " - diagnosis_keys MUST be a subset of WAITING-PERIOD KEYS (exact strings).\n"
        " - If the diagnosis is a normal/diagnostic procedure with no policy exclusion match, "
        "return excluded_matches as an empty list [].\n"
        " - line_item_tags.category is one of: covered | excluded | cosmetic.\n\n"
        "Return STRICT JSON with this exact shape:\n"
        "{\n"
        '  "diagnosis_keys": [],\n'
        '  "excluded_matches": [],\n'
        '  "line_item_tags": [{"description":"...","category":"covered|excluded|cosmetic","reason":"..."}],\n'
        '  "network_hospital_match": null,\n'
        '  "confidence": 0.0-1.0\n'
        "}\n"
    )

    try:
        text = await _call_text(cli, prompt)
        data = _parse_json(text)
        if not isinstance(data, dict):
            raise ValueError("bad shape")

        # Validate LLM output against the policy. This is the safety net for
        # hallucinated exclusions — only keep matches that actually appear in
        # the policy's exclusion list (case-insensitive partial match).
        raw_excluded = list(data.get("excluded_matches", []))
        validated_excluded = _validate_excluded_matches(raw_excluded, exclusion_conditions)

        raw_diag = list(data.get("diagnosis_keys", []))
        validated_diag = [k for k in raw_diag if k in waiting_keys]

        if raw_excluded != validated_excluded:
            log.warning(
                "semantic_classify: dropped invalid excluded_matches from LLM "
                "(raw=%s, kept=%s)", raw_excluded, validated_excluded,
            )
        log.info(
            "semantic_classify: gemini result · diagnosis_keys=%s excluded=%s network=%s",
            validated_diag, validated_excluded, data.get("network_hospital_match"),
        )
        return {
            "diagnosis_keys": validated_diag,
            "excluded_matches": validated_excluded,
            "line_item_tags": list(data.get("line_item_tags", [])),
            "network_hospital_match": data.get("network_hospital_match"),
            "confidence": float(data.get("confidence", 0.8)),
            "method": "gemini",
            "_llm_raw_excluded": raw_excluded,  # for trace debug
            "_llm_raw_diagnosis": raw_diag,
        }
    except Exception as e:
        if _is_quota_error(e):
            log.warning("semantic_classify: gemini quota exhausted, using keyword fallback")
        else:
            log.warning("semantic_classify: gemini failed (%s: %s), using keyword fallback",
                        type(e).__name__, str(e)[:200])
        return _semantic_keywords(extracted, policy_config)


def _validate_excluded_matches(
    llm_returned: list[str], policy_exclusions: list[str]
) -> list[str]:
    """Drop any LLM-returned excluded_matches that don't actually correspond
    to a policy exclusion. Match case-insensitively, accepting either the full
    policy phrase or a substring match (so "obesity" passes when the policy
    has "Obesity and weight loss programs").
    """
    policy_lower = [p.lower() for p in policy_exclusions]
    kept: list[str] = []
    for entry in llm_returned:
        if not entry:
            continue
        e = entry.lower().strip()
        # Must match (or be a substring of, or contain) a real policy exclusion.
        if any(e == p or e in p or p in e for p in policy_lower):
            kept.append(entry)
    return kept


def _word_match(text: str, term: str) -> bool:
    """True if `term` appears in `text` as a whole word/phrase.

    Needed because substring matching turns 'hernia' into a false positive on
    'lumbar disc herniation'. We anchor on word boundaries.
    """
    pattern = r"\b" + re.escape(term.strip()) + r"\b"
    return bool(re.search(pattern, text, re.I))


def _semantic_keywords(extracted: dict[str, Any], policy_config: dict[str, Any]) -> dict[str, Any]:
    """Deterministic fallback using keyword tables above + fuzzy hospital match."""
    diagnosis = (extracted.get("diagnosis") or "")
    treatment = (extracted.get("treatment") or "")
    combo = f"{diagnosis} {treatment}".lower()

    diagnosis_keys = [k for k, terms in _DIAGNOSIS_KEYWORDS.items()
                      if any(_word_match(combo, t) for t in terms)]

    # excluded_matches at the whole-claim level should ONLY trigger from the
    # diagnosis/treatment text, not from individual line items. Line items get
    # tagged separately below and filtered by the rules engine.
    excluded_matches: list[str] = []
    for k, terms in _EXCLUDED_KEYWORDS.items():
        if any(_word_match(combo, t) for t in terms):
            excluded_matches.append(k)

    # Tag each line item independently using the LINE-ITEM keyword set (which
    # includes single words like "whitening" — appropriate here because line
    # item descriptions ARE the procedure name, not surrounding context).
    line_tags: list[dict[str, Any]] = []
    dental_exclusions = [s.lower() for s in policy_config.get("exclusions", {}).get("dental_exclusions", [])]
    vision_exclusions = [s.lower() for s in policy_config.get("exclusions", {}).get("vision_exclusions", [])]
    for item in extracted.get("line_items", []) or []:
        desc = str(item.get("description", "")).lower()
        category = "covered"
        reason = ""
        if any(_word_match(desc, t) for t in _LINE_ITEM_EXCLUDED_KEYWORDS["cosmetic"]):
            category, reason = "cosmetic", "matches cosmetic-exclusion keyword"
        elif any(_word_match(desc, t) for t in dental_exclusions):
            category, reason = "excluded", "matches dental-exclusion list"
        elif any(_word_match(desc, t) for t in vision_exclusions):
            category, reason = "excluded", "matches vision-exclusion list"
        elif any(_word_match(desc, t) for t in _LINE_ITEM_EXCLUDED_KEYWORDS["obesity"]):
            category, reason = "excluded", "matches obesity-exclusion"
        line_tags.append({**item, "category": category, "reason": reason})

    network_match = _fuzzy_network_match(
        extracted.get("hospital_name") or "",
        policy_config.get("network_hospitals", []),
    )

    return {
        "diagnosis_keys": diagnosis_keys,
        "excluded_matches": excluded_matches,
        "line_item_tags": line_tags,
        "network_hospital_match": network_match,
        "confidence": 0.75,
        "method": "keywords",
    }


def _fuzzy_network_match(hospital: str, network: list[str]) -> Optional[str]:
    if not hospital:
        return None
    try:
        from rapidfuzz import process, fuzz
        best = process.extractOne(hospital, network, scorer=fuzz.token_set_ratio)
        if best and best[1] >= 80:
            return best[0]
    except ImportError:
        for n in network:
            if n.lower() in hospital.lower() or hospital.lower() in n.lower():
                return n
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. Decision reasoning (human-readable summary)
# ─────────────────────────────────────────────────────────────────────────────


async def extract_line_items(raw_text: str) -> list[dict[str, Any]]:
    """Use Gemini to extract billable line items from messy OCR text.

    Called as a fallback when the regex line-item parser returns nothing.
    Returns a list of {description, amount} dicts. Empty list on any failure
    (no key, quota exhausted, parse error, etc.).
    """
    if not raw_text or len(raw_text) < 50:
        return []

    cli = _client()
    if cli is None:
        return []

    log.info("extract_line_items: calling Gemini on %d chars of OCR text", len(raw_text))

    prompt = (
        "You extract BILLABLE LINE ITEMS from an Indian medical bill (raw OCR text). "
        "Return ONLY the line items — not subtotals, taxes, discounts, totals, or header rows.\n\n"
        "EACH line item must have:\n"
        "  - description: a short phrase like 'Consultation Fee' or 'Root Canal Treatment'\n"
        "  - amount: a positive number (rupees, no symbol)\n\n"
        "Skip lines like 'Total', 'Subtotal', 'GST', 'Discount', 'Net Amount', "
        "'Grand Total', 'Bill No', 'Date', 'Patient', 'Doctor'.\n\n"
        f"OCR TEXT:\n{raw_text[:3000]}\n\n"
        "Respond as STRICT JSON: "
        '{"line_items": [{"description": "...", "amount": 0.0}, ...]}\n'
        "If you find no billable items, respond: "
        '{"line_items": []}'
    )

    try:
        text = await _call_text(cli, prompt)
        data = _parse_json(text)
        items = data.get("line_items") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return []
        cleaned: list[dict[str, Any]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            desc = str(it.get("description", "")).strip()
            try:
                amt = float(it.get("amount", 0))
            except (TypeError, ValueError):
                continue
            if desc and amt > 0:
                cleaned.append({"description": desc, "amount": amt})
        log.info("extract_line_items: gemini returned %d items", len(cleaned))
        return cleaned
    except Exception as e:
        if _is_quota_error(e):
            log.warning("extract_line_items: gemini quota exhausted")
        else:
            log.warning("extract_line_items: gemini failed (%s)", type(e).__name__)
        return []


async def generate_reasoning(context: dict[str, Any]) -> str:
    """Produce a 2-4 sentence explanation given the structured decision context.

    Crucially: the LLM does NOT decide here. It writes prose around the numbers
    that the rules engine already produced. The validator agent re-checks the
    numbers afterwards.
    """
    cli = _client()
    if cli is None:
        log.info("generate_reasoning: no Gemini key, using template fallback")
        return _template_reasoning(context)
    log.info("generate_reasoning: calling Gemini for decision=%s amt=%s",
             context.get("decision"), context.get("approved_amount"))

    prompt = (
        "You write a short, clear explanation for a health insurance claim decision.\n"
        "DO NOT invent numbers — use exactly what is given. Be specific about WHY.\n\n"
        f"DECISION CONTEXT (authoritative, do not contradict):\n{json.dumps(context, default=str)}\n\n"
        "Write 2-4 sentences for the member. Mention the decision, the approved amount if any, "
        "the specific reason (waiting period date, exclusion category, calculation breakdown), "
        "and what the member should do next if applicable."
    )

    try:
        text = await _call_text(cli, prompt)
        result = text.strip()[:2000] if text else _template_reasoning(context)
        log.info("generate_reasoning: gemini returned %d chars", len(result))
        return result
    except Exception as e:
        if _is_quota_error(e):
            log.warning("generate_reasoning: gemini quota exhausted, using template")
        else:
            log.warning("generate_reasoning: gemini failed (%s: %s), using template",
                        type(e).__name__, str(e)[:200])
        return _template_reasoning(context)


def _template_reasoning(context: dict[str, Any]) -> str:
    d = context.get("decision", "MANUAL_REVIEW")
    amt = context.get("approved_amount", 0)
    reasons = context.get("reasons", [])
    if d == "APPROVED":
        return f"Claim approved for ₹{amt:.0f}. {context.get('breakdown_note','')}"
    if d == "PARTIAL":
        return (f"Partially approved for ₹{amt:.0f}. Some line items were excluded under the policy: "
                f"{', '.join(reasons) or 'see breakdown'}.")
    if d == "REJECTED":
        return f"Claim rejected. Reason: {', '.join(reasons) or 'see policy terms'}."
    return "This claim has been routed to manual review. An operations agent will get back to you."


# ─────────────────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────────────────


async def _call_text(cli: Any, prompt: str) -> str:
    """Run a Gemini generation in a worker thread (the SDK is sync).

    If the primary model returns 429 (quota exhausted), iterate through the
    fallback models — each free-tier model has its own quota pool.
    """
    def _run(model: str) -> str:
        resp = cli.models.generate_content(
            model=model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1024,
            ),
        )
        return getattr(resp, "text", "") or ""

    last_exc: Optional[Exception] = None
    for model in [_MODEL, *_FALLBACK_MODELS]:
        try:
            return await asyncio.to_thread(_run, model)
        except Exception as e:
            last_exc = e
            if _is_quota_error(e):
                log.info("llm: %s quota exhausted, trying next model", model)
                continue
            # Non-quota error — don't iterate, let caller handle it
            raise
    # All models failed (likely all quota-exhausted)
    raise last_exc if last_exc else RuntimeError("All Gemini models failed")


_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_json(text: str) -> Any:
    """Best-effort JSON extraction — handles raw JSON or fenced code blocks."""
    if not text:
        raise ValueError("empty")
    s = text.strip()
    # try direct
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # try fenced
    m = _JSON_FENCE.search(s)
    if m:
        return json.loads(m.group(1))
    # try slicing first {...} block
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(s[start:end + 1])
    raise ValueError("no json")
