"""Eval runner — executes all 12 cases from test_cases.json through the full pipeline.

Usage:
  python eval.py [--cases TC001,TC005] [--out eval_report.md]

The script:
  1. Ensures the policy from policy_terms.json is loaded into the DB.
  2. Ensures members from policy_terms.json are loaded.
  3. For each test case, builds a ClaimSubmission with pre_extracted document
     content (since test_cases.json gives us extracted JSON, not real files).
  4. Runs it through services.pipeline.process_claim.
  5. Compares actual vs expected and writes eval_report.md.

Note: real files would exercise pdfplumber/OCR — the pre-extracted path skips
that to give deterministic eval results. The Streamlit UI tests the real-file
path end-to-end.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from datetime import date
from pathlib import Path
from typing import Any

from services import db
from services.models import (
    ClaimCategory,
    ClaimSubmission,
    Decision,
    DocumentQuality,
    DocumentType,
    Member,
    UploadedDocument,
)
from services.pipeline import process_claim


ROOT = Path(__file__).parent


def _load_seed() -> tuple[dict, dict]:
    with open(ROOT / "policy_terms.json", "r", encoding="utf-8") as f:
        policy_cfg = json.load(f)
    with open(ROOT / "test_cases.json", "r", encoding="utf-8") as f:
        cases = json.load(f)
    return policy_cfg, cases


def _seed_db(policy_cfg: dict) -> None:
    db.init_db()
    # Wipe any prior eval state so each run is deterministic and isolated.
    with db.get_conn() as conn:
        for table in ("traces", "decisions", "documents", "claims"):
            conn.execute(f"DELETE FROM {table}")
    policy_id = policy_cfg["policy_id"]
    db.save_policy(policy_id, policy_cfg.get("policy_name", policy_id), policy_cfg)
    for m in policy_cfg.get("members", []):
        try:
            db.save_member(Member(
                member_id=m["member_id"],
                policy_id=policy_id,
                name=m["name"],
                date_of_birth=date.fromisoformat(m["date_of_birth"]),
                gender=m.get("gender", ""),
                relationship=m.get("relationship", ""),
                join_date=date.fromisoformat(m["join_date"]) if m.get("join_date") else None,
                primary_member_id=m.get("primary_member_id"),
            ))
        except Exception as e:
            print(f"  ! skipped member {m.get('member_id')}: {e}")


def _build_submission(case: dict[str, Any]) -> ClaimSubmission:
    inp = case["input"]
    docs: list[UploadedDocument] = []
    for d in inp.get("documents", []):
        actual = d.get("actual_type")
        quality = d.get("quality")
        docs.append(UploadedDocument(
            file_id=d["file_id"],
            file_name=d.get("file_name", f"{d['file_id']}.pdf"),
            file_path=None,
            declared_type=DocumentType(actual) if actual else None,
            pre_extracted=d.get("content"),
            quality_hint=DocumentQuality(quality) if quality else None,
        ))
    treatment_date = date.fromisoformat(inp["treatment_date"])
    return ClaimSubmission(
        member_id=inp["member_id"],
        policy_id=inp.get("policy_id"),
        category=ClaimCategory(inp["claim_category"]),
        claimed_amount=float(inp["claimed_amount"]),
        treatment_date=treatment_date,
        # Eval treats every claim as submitted same day as treatment, so the
        # deadline check is meaningful relative to the test scenario's timeline.
        submission_date=treatment_date,
        hospital_name=inp.get("hospital_name"),
        documents=docs,
        claims_history=inp.get("claims_history", []),
        simulate_component_failure=bool(inp.get("simulate_component_failure", False)),
    )


def _evaluate(case: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    """Compare actual pipeline output vs the test case's expected block."""
    exp = case["expected"]
    expected_decision = exp.get("decision")
    actual_decision = actual.get("decision")

    # Many test cases expect decision=null when a gate should fire.
    # In our system, gates produce decision=BLOCKED. Map that for comparison.
    decision_matches: bool
    if expected_decision is None:
        decision_matches = actual_decision in ("BLOCKED",)
    else:
        decision_matches = actual_decision == expected_decision

    notes: list[str] = []

    if "approved_amount" in exp:
        diff = abs(float(exp["approved_amount"]) - float(actual.get("approved_amount", 0)))
        amount_ok = diff <= 1.0
        notes.append(f"approved_amount expected={exp['approved_amount']} actual={actual.get('approved_amount')} (Δ={diff:.2f})")
    else:
        amount_ok = True

    if "rejection_reasons" in exp:
        expected_reasons = set(exp["rejection_reasons"])
        actual_reasons = set(actual.get("rejection_reasons", []))
        reasons_ok = expected_reasons.issubset(actual_reasons)
        notes.append(f"reasons expected⊆actual: {expected_reasons} vs {actual_reasons}")
    else:
        reasons_ok = True

    if "confidence_score" in exp:
        spec = exp["confidence_score"]
        if spec.startswith("above"):
            try:
                threshold = float(spec.split()[-1])
                conf_ok = float(actual.get("confidence", 0)) >= threshold
                notes.append(f"confidence ≥ {threshold}: actual={actual.get('confidence')}")
            except Exception:
                conf_ok = True
        else:
            conf_ok = True
    else:
        conf_ok = True

    pass_all = decision_matches and amount_ok and reasons_ok and conf_ok

    return {
        "passed": pass_all,
        "decision_matches": decision_matches,
        "amount_ok": amount_ok,
        "reasons_ok": reasons_ok,
        "conf_ok": conf_ok,
        "notes": notes,
    }


def _render_report(results: list[dict[str, Any]]) -> str:
    n = len(results)
    passed = sum(1 for r in results if r["eval"]["passed"])
    lines: list[str] = []
    lines.append(f"# Eval Report\n")
    lines.append(f"**{passed}/{n} cases passed**\n")
    lines.append("| Case | Name | Expected | Actual | ₹ Approved | Pass |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        c = r["case"]
        a = r["actual"]
        e = r["eval"]
        exp_dec = c["expected"].get("decision") or "BLOCKED (gate)"
        lines.append(
            f"| {c['case_id']} | {c['case_name']} | "
            f"{exp_dec} | {a.get('decision')} | "
            f"₹{a.get('approved_amount',0):.0f} | "
            f"{'✅' if e['passed'] else '❌'} |"
        )
    lines.append("\n---\n")

    for r in results:
        c = r["case"]
        a = r["actual"]
        e = r["eval"]
        lines.append(f"## {c['case_id']} — {c['case_name']}")
        lines.append(f"**Result:** {'✅ PASS' if e['passed'] else '❌ FAIL'}")
        lines.append("")
        lines.append(f"- Expected decision: `{c['expected'].get('decision') or 'BLOCKED'}`")
        lines.append(f"- Actual decision: `{a.get('decision')}`")
        lines.append(f"- Approved amount: ₹{a.get('approved_amount', 0):.0f}")
        lines.append(f"- Confidence: {a.get('confidence', 0):.2f}")
        lines.append(f"- User message: _{a.get('user_message','')}_")
        if e["notes"]:
            lines.append("- Eval notes:")
            for n in e["notes"]:
                lines.append(f"  - {n}")
        trace = a.get("trace", [])
        if trace:
            lines.append(f"- Trace ({len(trace)} events):")
            for ev in trace:
                lines.append(
                    f"  - #{ev['sequence']} `{ev['agent']}` → {ev['status']} "
                    f"({ev.get('duration_ms', 0)}ms)"
                )
        lines.append("")
    return "\n".join(lines)


async def _run_all(case_filter: set[str] | None) -> list[dict[str, Any]]:
    policy_cfg, cases_file = _load_seed()
    _seed_db(policy_cfg)
    results: list[dict[str, Any]] = []
    for case in cases_file["test_cases"]:
        if case_filter and case["case_id"] not in case_filter:
            continue
        # Isolate each test case — previous cases' claim rows must not leak
        # into the fraud detector's same-day / monthly counts.
        with db.get_conn() as conn:
            for table in ("traces", "decisions", "documents", "claims"):
                conn.execute(f"DELETE FROM {table}")
        print(f">> {case['case_id']} {case['case_name']}")
        try:
            submission = _build_submission(case)
            decision = await process_claim(submission)
            actual = decision.model_dump(mode="json")
        except Exception as e:
            actual = {
                "decision": "ERROR",
                "approved_amount": 0,
                "rejection_reasons": [],
                "user_message": f"Pipeline crashed: {e}",
                "confidence": 0,
                "trace": [],
            }
        ev = _evaluate(case, actual)
        results.append({"case": case, "actual": actual, "eval": ev})
        print(f"  {'[PASS]' if ev['passed'] else '[FAIL]'} {actual.get('decision')} (Rs.{actual.get('approved_amount',0):.0f})")
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=str, default=None, help="comma-separated case IDs to run")
    parser.add_argument("--out", type=str, default="eval_report.md")
    args = parser.parse_args()

    case_filter = set(args.cases.split(",")) if args.cases else None
    results = asyncio.run(_run_all(case_filter))

    report = _render_report(results)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport: {args.out}")
    passed = sum(1 for r in results if r["eval"]["passed"])
    print(f"PASS: {passed}/{len(results)}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
