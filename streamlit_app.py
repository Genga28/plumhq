"""Plum Claims — Streamlit UI.

Two things the assignment asks for:
  1. Submit a claim.
  2. Review the decision (with a trace).

So the UI has three sections, navigated via the sidebar:
  • Setup       — load policy + members from JSON (one-time)
  • Submit      — pick member, fill claim details, upload documents, get decision
  • History     — browse prior claims; click one to see its decision + trace
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import date
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from services import db, llm
from services.models import (
    ClaimCategory,
    ClaimSubmission,
    DocumentType,
    Member,
    UploadedDocument,
)
from services.pipeline import process_claim

load_dotenv()

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Plum Claims", layout="wide", page_icon="🩺")
db.init_db()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────


def _sidebar() -> str:
    st.sidebar.markdown("### 🩺 Plum Claims")
    st.sidebar.caption("Multi-agent claims processor")

    policies = db.list_policies()
    members = db.list_members()

    pages = ["Policy Upload", "Claim Upload", "History"]
    default = "Claim Upload" if policies else "Policy Upload"
    page = st.sidebar.radio(
        "Pages", pages,
        index=pages.index(default),
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.caption(
        f"Policies: **{len(policies)}**  ·  Members: **{len(members)}**"
    )
    st.sidebar.caption(
        f"Gemini: {'✓ on' if llm.is_available() else '⚠ stub fallback'}"
    )
    return page


# ─────────────────────────────────────────────────────────────────────────────
# Setup page
# ─────────────────────────────────────────────────────────────────────────────


def page_setup() -> None:
    st.title("Policy Upload")
    st.write("Upload a policy JSON to load the policy config and its member roster.")

    policies = db.list_policies()
    if policies:
        st.success(f"✓ {len(policies)} policy loaded · {len(db.list_members())} members")

    f = st.file_uploader("Policy JSON", type=["json"], label_visibility="collapsed")
    if f is None:
        return
    try:
        cfg = json.loads(f.read().decode("utf-8"))
    except Exception as e:
        st.error(f"Could not parse JSON: {e}")
        return

    c1, c2 = st.columns(2)
    c1.write(f"**Policy ID:** `{cfg.get('policy_id','?')}`")
    c2.write(f"**Members:** {len(cfg.get('members', []))}")

    if st.button("Save policy", type="primary"):
        _seed_policy(cfg)


def _seed_policy(cfg: dict[str, Any]) -> None:
    policy_id = cfg.get("policy_id") or f"POL_{uuid.uuid4().hex[:8]}"
    name = cfg.get("policy_name", policy_id)
    db.save_policy(policy_id, name, cfg)
    saved = 0
    skipped = 0
    for m in cfg.get("members", []):
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
            saved += 1
        except Exception:
            skipped += 1
    st.success(f"✓ Saved policy `{policy_id}` · {saved} members loaded" + (f" · {skipped} skipped" if skipped else ""))


# ─────────────────────────────────────────────────────────────────────────────
# Submit page
# ─────────────────────────────────────────────────────────────────────────────


def page_submit() -> None:
    st.title("Claim Upload")

    members = db.list_members()
    if not members:
        st.warning("Upload a policy first (Policy Upload page).")
        return

    # ── Member + policy ──
    member_opts = {f"{m['member_id']} · {m['name']} ({m['relationship']})": m["member_id"] for m in members}
    label = st.selectbox("Member", list(member_opts.keys()))
    member_id = member_opts[label]
    member = db.get_member(member_id)
    policy = db.get_policy(member.policy_id) if member else None

    if member and policy:
        with st.expander("👤 Member & coverage details", expanded=False):
            _render_member_coverage(member, policy)

    # ── Claim details ──
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        category = st.selectbox("Category", [c.value for c in ClaimCategory])
        treatment_date = st.date_input("Treatment date", value=date.today())
    with c2:
        claimed_amount = st.number_input("Claimed amount (₹)", min_value=0.0, value=1500.0, step=100.0)
        hospital_name = st.text_input("Hospital (optional, free-text)", placeholder="e.g. Apollo Hospitals")

    # ── Documents ──
    st.write("")
    st.write("**Upload documents** (PDF or image — the system will OCR and classify them)")
    uploaded = st.file_uploader(
        "Files",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    declared_types: dict[str, str] = {}
    if uploaded:
        st.caption("Optional: declare a type per file (or leave on auto-detect)")
        cols = st.columns(min(3, len(uploaded)))
        for i, uf in enumerate(uploaded):
            with cols[i % len(cols)]:
                declared_types[uf.name] = st.selectbox(
                    uf.name,
                    ["(auto-detect)"] + [t.value for t in DocumentType if t != DocumentType.UNKNOWN],
                    key=f"dt_{i}",
                )

    # ── Submit ──
    st.write("")
    submit_disabled = not uploaded
    if st.button("Submit claim", type="primary", disabled=submit_disabled):
        # Persist uploads to disk
        claim_tmp = uuid.uuid4().hex[:10]
        claim_dir = UPLOAD_DIR / claim_tmp
        claim_dir.mkdir(parents=True, exist_ok=True)
        docs: list[UploadedDocument] = []
        for uf in uploaded:
            target = claim_dir / uf.name
            with open(target, "wb") as out:
                out.write(uf.getbuffer())
            dt = declared_types.get(uf.name, "(auto-detect)")
            docs.append(UploadedDocument(
                file_id=uuid.uuid4().hex[:8],
                file_name=uf.name,
                file_path=str(target),
                declared_type=DocumentType(dt) if dt != "(auto-detect)" else None,
            ))

        submission = ClaimSubmission(
            member_id=member_id,
            category=ClaimCategory(category),
            claimed_amount=claimed_amount,
            treatment_date=treatment_date,
            submission_date=date.today(),
            hospital_name=hospital_name or None,
            documents=docs,
        )

        with st.spinner("Running pipeline…"):
            decision = asyncio.run(process_claim(submission))

        st.session_state["last_decision"] = decision.model_dump(mode="json")
        st.divider()
        _render_decision(decision.model_dump(mode="json"))

    # If a decision was rendered earlier in this session, keep it visible after rerun.
    elif "last_decision" in st.session_state and not uploaded:
        st.caption("Previous decision (session):")
        _render_decision(st.session_state["last_decision"])


# ─────────────────────────────────────────────────────────────────────────────
# Member + coverage details (shown on Submit page)
# ─────────────────────────────────────────────────────────────────────────────


def _render_member_coverage(member: Member, policy) -> None:
    """Show member info + the full coverage map (sub-limits, copay, exclusions,
    waiting periods, network hospitals) so the user can see at a glance what
    the member is covered for."""

    # ── Member info ──
    st.markdown("**Member**")
    c1, c2, c3 = st.columns(3)
    c1.caption(f"**ID:** {member.member_id}")
    c1.caption(f"**Name:** {member.name}")
    c2.caption(f"**Gender:** {member.gender}")
    c2.caption(f"**DOB:** {member.date_of_birth.isoformat() if member.date_of_birth else '-'}")
    c3.caption(f"**Relationship:** {member.relationship}")
    c3.caption(f"**Join date:** {member.join_date.isoformat() if member.join_date else '-'}")
    if member.primary_member_id:
        st.caption(f"_Dependent of: `{member.primary_member_id}`_")

    st.divider()

    # ── Policy headline numbers ──
    st.markdown("**Policy coverage**")
    st.caption(f"**{policy.name}** · `{policy.policy_id}`")
    cov = policy.coverage
    m1, m2, m3 = st.columns(3)
    m1.metric("Sum insured", f"₹{float(cov.get('sum_insured_per_employee', 0)):,.0f}")
    m2.metric("Annual OPD limit", f"₹{float(cov.get('annual_opd_limit', 0)):,.0f}")
    m3.metric("Per-claim limit", f"₹{float(cov.get('per_claim_limit', 0)):,.0f}")

    # ── Categories with sub-limits, copay, network discount ──
    st.markdown("**Per-category coverage**")
    cats = policy.opd_categories or {}
    rows = []
    for name, c in cats.items():
        if not c.get("covered", False):
            continue
        rows.append({
            "Category": name.replace("_", " ").title(),
            "Sub-limit": f"₹{float(c.get('sub_limit', 0)):,.0f}",
            "Co-pay": f"{c.get('copay_percent', 0)}%",
            "Network discount": (f"{c.get('network_discount_percent', 0)}%"
                                 if c.get("network_discount_percent") else "—"),
            "Pre-auth": (f"above ₹{c.get('pre_auth_threshold','-')}"
                         if c.get("pre_auth_threshold") else
                         ("required" if c.get("requires_pre_auth") else "—")),
        })
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)

    # ── Waiting periods ──
    wp = policy.waiting_periods or {}
    st.markdown("**Waiting periods**")
    initial = wp.get("initial_waiting_period_days", 0)
    st.caption(f"Initial (all conditions): **{initial} days** from policy join")
    specific = wp.get("specific_conditions") or {}
    if specific:
        wp_rows = [{"Condition": k.replace("_", " ").title(), "Days": v} for k, v in specific.items()]
        st.dataframe(wp_rows, hide_index=True, use_container_width=True)

    # ── Exclusions ──
    excl = policy.exclusions or {}
    conds = excl.get("conditions") or []
    if conds:
        st.markdown("**General exclusions**")
        for c in conds:
            st.caption(f"• {c}")

    cols = st.columns(2)
    dental_ex = excl.get("dental_exclusions") or []
    vision_ex = excl.get("vision_exclusions") or []
    if dental_ex:
        with cols[0]:
            st.markdown("**Dental exclusions**")
            for c in dental_ex:
                st.caption(f"• {c}")
    if vision_ex:
        with cols[1]:
            st.markdown("**Vision exclusions**")
            for c in vision_ex:
                st.caption(f"• {c}")

    # ── Network hospitals ──
    hospitals = policy.network_hospitals or []
    if hospitals:
        st.markdown(f"**Network hospitals** ({len(hospitals)})")
        st.caption(" · ".join(hospitals))

    # ── Document requirements per category ──
    reqs = policy.document_requirements or {}
    if reqs:
        st.markdown("**Required documents per claim type**")
        req_rows = []
        for cat, r in reqs.items():
            req_rows.append({
                "Category": cat,
                "Required": ", ".join(r.get("required", [])),
                "Optional": ", ".join(r.get("optional", [])) or "—",
            })
        st.dataframe(req_rows, hide_index=True, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# History page
# ─────────────────────────────────────────────────────────────────────────────


def page_history() -> None:
    st.title("History")
    claims = db.list_claims(limit=100)
    if not claims:
        st.info("No claims yet. Submit one to see it here.")
        return

    # Compact summary table
    rows = [{
        "Claim": c["claim_id"],
        "Member": c.get("member_name") or c["member_id"],
        "Category": c["category"],
        "Amount": f"₹{float(c['claimed_amount']):,.0f}",
        "Decision": c.get("decision") or c.get("status"),
        "Approved": f"₹{float(c.get('approved_amount') or 0):,.0f}",
        "When": (c.get("created_at") or "")[:19],
    } for c in claims]
    st.dataframe(rows, hide_index=True, use_container_width=True)

    st.divider()
    claim_id = st.selectbox("Open a claim", [""] + [c["claim_id"] for c in claims])
    if not claim_id:
        return

    dec = db.get_decision(claim_id)
    trace = db.get_trace(claim_id)
    if not dec:
        st.warning("No decision found for that claim.")
        return

    full = {
        "claim_id": claim_id,
        "decision": dec["decision"],
        "approved_amount": dec["approved_amount"],
        "rejection_reasons": json.loads(dec["reasons_json"] or "[]"),
        "user_message": dec["user_message"],
        "llm_reasoning": dec["llm_reasoning"],
        "confidence": dec["confidence"],
        "breakdown": json.loads(dec["breakdown_json"] or "{}"),
        "line_items": [],
        "trace": [t.model_dump(mode="json") for t in trace],
    }
    _render_decision(full)


# ─────────────────────────────────────────────────────────────────────────────
# Decision rendering — single, focused layout
# ─────────────────────────────────────────────────────────────────────────────


def _render_decision(d: dict[str, Any]) -> None:
    dec = d.get("decision", "UNKNOWN")

    # 1. Banner
    _banner(dec, d.get("user_message") or "")

    # 2. Key numbers (only what makes sense for this decision)
    _metrics(dec, d)

    # 3. Why (reasons / MR triggers / notes)
    _why(dec, d)

    # 4. Calculation breakdown (only when there's actual math to show)
    if dec in ("APPROVED", "PARTIAL"):
        _calc(d)

    # 5. Line items (only when non-trivial)
    line_items = d.get("line_items") or []
    if line_items and any(li.get("status") == "REJECTED" for li in line_items):
        # Always show when there's a partial — that's the point of TC006
        st.markdown("**Line items**")
        _line_items(line_items)
    elif line_items and len(line_items) > 1:
        with st.expander(f"Line items ({len(line_items)})"):
            _line_items(line_items)

    # 6. Trace (collapsed by default)
    trace = d.get("trace") or []
    if trace:
        with st.expander(f"📜 Agent trace · {len(trace)} steps", expanded=False):
            _trace(trace)


def _banner(dec: str, msg: str) -> None:
    mapping = {
        "APPROVED":      ("✅ APPROVED",                st.success),
        "PARTIAL":       ("🟡 PARTIALLY APPROVED",     st.warning),
        "REJECTED":      ("❌ REJECTED",                st.error),
        "MANUAL_REVIEW": ("🔍 MANUAL REVIEW",          st.warning),
        "BLOCKED":       ("🚫 BLOCKED — fix and resubmit", st.error),
    }
    title, fn = mapping.get(dec, (dec, st.info))
    fn(f"**{title}**\n\n{msg or '(no message)'}")


def _metrics(dec: str, d: dict[str, Any]) -> None:
    """Two metrics only: approved amount and confidence. Decision is in the banner."""
    if dec == "BLOCKED":
        # nothing to show — the banner already says it all
        return
    c1, c2 = st.columns(2)
    c1.metric("Approved", f"₹{float(d.get('approved_amount', 0) or 0):,.0f}")
    c2.metric("Confidence", f"{(d.get('confidence', 0) or 0):.0%}")


def _why(dec: str, d: dict[str, Any]) -> None:
    """Show rejection reasons, MR triggers, and policy notes."""
    reasons = d.get("rejection_reasons") or []
    if reasons:
        st.markdown("**Rejected because:**")
        for r in reasons:
            st.markdown(f"- `{r}`")

    b = d.get("breakdown") or {}
    notes = b.get("notes") or []
    mr_reasons = b.get("manual_review_reasons") or []
    fraud_triggers = b.get("fraud_signals") or []

    if mr_reasons:
        st.markdown("**Routed to manual review because:**")
        for r in mr_reasons:
            st.markdown(f"- {r}")

    if fraud_triggers:
        st.markdown("**Fraud signals:**")
        for t in fraud_triggers:
            st.markdown(f"- ⚠ {t}")

    if notes:
        # Notes are the rules engine's human-readable explanations
        # (e.g. "Diabetes has a 90-day waiting period. Eligible from 2024-11-30")
        for n in notes:
            st.info(n)


def _calc(d: dict[str, Any]) -> None:
    """Show the arithmetic chain inline, only for approvals/partials."""
    b = d.get("breakdown") or {}
    if not b:
        return

    claimed   = b.get("claimed_amount")
    covered   = b.get("covered_total")
    rejected  = b.get("rejected_line_items_total") or 0
    discount  = b.get("network_discount_amount") or 0
    disc_pct  = b.get("network_discount_percent") or 0
    network   = b.get("network_hospital")
    copay     = b.get("copay_amount") or 0
    copay_pct = b.get("copay_percent") or 0
    approved  = d.get("approved_amount", 0)

    # If nothing interesting changed, skip
    if not discount and not copay and not rejected:
        return

    st.markdown("**Calculation**")
    lines = []
    if claimed is not None:
        lines.append(f"Claimed: ₹{float(claimed):,.0f}")
    if rejected and float(rejected) > 0:
        lines.append(f"− Excluded line items: ₹{float(rejected):,.0f}")
    if covered is not None and rejected:
        lines.append(f"= Covered: ₹{float(covered):,.0f}")
    if discount and float(discount) > 0:
        label = f"({disc_pct:.0f}% network"
        if network:
            label += f" — {network}"
        label += ")"
        lines.append(f"− Network discount {label}: ₹{float(discount):,.0f}")
    if copay and float(copay) > 0:
        lines.append(f"− Co-pay ({copay_pct:.0f}%): ₹{float(copay):,.0f}")
    lines.append(f"**= Approved: ₹{float(approved):,.0f}**")
    st.markdown("\n".join(lines))


def _line_items(items: list[dict[str, Any]]) -> None:
    rows = []
    for li in items:
        status = li.get("status", "")
        rows.append({
            "✓/✗": "✓" if status == "APPROVED" else "✗",
            "Item":     li.get("description", ""),
            "Claimed":  f"₹{float(li.get('claimed_amount', 0) or 0):,.0f}",
            "Approved": f"₹{float(li.get('approved_amount', 0) or 0):,.0f}",
            "Reason":   li.get("reason") or "",
        })
    st.dataframe(rows, hide_index=True, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Trace renderer — clean per-agent cards, no raw JSON unless toggled
# ─────────────────────────────────────────────────────────────────────────────


def _trace(events: list[dict[str, Any]]) -> None:
    show_raw = st.toggle("Show raw payload (debug)", value=False, key="trace_raw")
    badges = {"OK": "🟢", "BLOCKED": "🚫", "DEGRADED": "🟡", "FAILED": "🔴"}
    for ev in events:
        status = ev.get("status", "OK")
        badge = badges.get(status, "•")
        seq = ev.get("sequence", "?")
        agent = ev.get("agent", "?")
        ms = ev.get("duration_ms", 0)
        payload = ev.get("payload") or {}

        st.markdown(f"**{badge} `#{seq}` {agent}** · {ms}ms · {status}")
        _agent_card(agent, payload)
        if show_raw and payload:
            st.code(json.dumps(payload, indent=2, default=str), language="json")
        st.divider()


def _agent_card(agent: str, p: dict[str, Any]) -> None:
    if not p:
        return

    if agent == "Intake":
        st.caption(f"{p.get('message','')} · member `{p.get('member_id','-')}` · policy `{p.get('policy_id','-')}`")
        return

    if agent == "DocumentClassifier":
        st.caption(
            f"📄 **{p.get('file_name','file')}** → `{p.get('actual_type','?')}` "
            f"({(p.get('classification_confidence',0) or 0):.0%}, via {p.get('via','?')})"
        )
        return

    if agent == "DocumentVerifier":
        missing = p.get("missing_required") or []
        if missing:
            st.error("Missing: " + ", ".join(missing))
        else:
            st.caption("✓ All required documents present")
        if p.get("user_message"):
            st.caption(p["user_message"])
        return

    if agent == "QualityChecker":
        q = p.get("quality", "?")
        oc = p.get("ocr_confidence", 0) or 0
        if p.get("needs_reupload"):
            st.error(f"{p.get('file_id','file')}: {q} — OCR {oc:.0%}, re-upload needed")
        else:
            st.caption(f"✓ {p.get('file_id','file')}: {q} (OCR {oc:.0%})")
        return

    if agent == "Extractor":
        bits = []
        if p.get("patient_name"):  bits.append(f"👤 {p['patient_name']}")
        if p.get("doctor_name"):   bits.append(f"👨‍⚕️ {p['doctor_name']}")
        if p.get("hospital_name"): bits.append(f"🏥 {p['hospital_name']}")
        if p.get("diagnosis"):     bits.append(f"🩺 {p['diagnosis']}")
        if bits:
            st.caption(" · ".join(bits))

        n_items = p.get("line_items_count", 0)
        source = p.get("line_items_source", "regex")
        total = p.get("total_amount")
        total_str = f"₹{float(total):,.0f}" if total is not None else "—"
        st.caption(
            f"line items: {n_items} (source: {source}) · "
            f"total: {total_str} · "
            f"extraction confidence {(p.get('extraction_confidence',0) or 0):.0%}"
        )

        # Show extracted line items inline (when present) so the user can see
        # exactly what the extractor found.
        items = p.get("line_items") or []
        if items:
            st.dataframe(
                [{"item": it.get("description", ""),
                  "amount": f"₹{float(it.get('amount', 0) or 0):,.0f}"} for it in items],
                hide_index=True, use_container_width=True,
            )

        # Full OCR text in a popover so users can verify what was extracted
        raw = p.get("raw_text") or ""
        if raw.strip():
            with st.popover(f"📝 view OCR text ({len(raw)} chars)"):
                st.text(raw)
        return

    if agent == "ConsistencyChecker":
        names = p.get("patient_names_found") or []
        if p.get("patient_match"):
            st.caption("✓ Patient match" + (f": {', '.join(names)}" if names else ""))
        else:
            st.error(f"Patient mismatch: {', '.join(names)}")
        return

    if agent == "SemanticClassifier":
        bits = []
        for k in p.get("diagnosis_keys") or []:
            bits.append(f"🏷 `{k}`")
        for k in p.get("excluded_matches") or []:
            bits.append(f"🚫 `{k}`")
        if p.get("network_hospital_match"):
            bits.append(f"🏥 `{p['network_hospital_match']}`")
        if bits:
            st.markdown(" ".join(bits))
        tags = p.get("line_item_tags") or []
        if tags and any(t.get("category") != "covered" for t in tags):
            for t in tags:
                cat = (t.get("category") or "covered").lower()
                if cat == "covered":
                    continue
                emoji = {"excluded": "✗", "cosmetic": "💄"}.get(cat, "•")
                st.caption(f"{emoji} {t.get('description','')} → `{cat}` ({t.get('reason','')})")
        return

    if agent == "FraudDetector":
        score = p.get("fraud_score", 0) or 0
        c1, c2, c3 = st.columns(3)
        c1.caption(f"Fraud score: **{score:.2f}**")
        c2.caption(f"Same-day: **{p.get('same_day_count', 0)}**")
        c3.caption(f"Monthly: **{p.get('monthly_count', 0)}**")
        for t in p.get("triggers") or []:
            st.caption(f"⚠ {t}")
        return

    if agent == "RulesEngine":
        st.caption(f"Decision: **{p.get('decision','?')}** · Approved: **₹{float(p.get('approved_amount', 0) or 0):,.0f}**")
        for r in p.get("rejection_reasons") or []:
            st.caption(f"❌ {r}")
        return

    if agent == "DecisionReasoner":
        preview = (p.get("reasoning_preview") or "").strip()
        if preview:
            st.caption(preview[:300])
        st.caption(
            f"_via {'Gemini' if p.get('llm_available') else 'stub fallback'}, "
            f"{p.get('reasoning_length',0)} chars_"
        )
        return

    if agent == "DecisionValidator":
        if p.get("valid"):
            st.caption(f"✓ Verified {p.get('mentioned_count',0)} ₹ amount(s) in reasoning")
        else:
            st.error(f"Mismatch in reasoning: {p.get('mismatches')}")
        return

    if p.get("error"):
        st.error(f"💥 {p.get('exception_type','Error')}: {p.get('error')}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    page = _sidebar()
    if page == "Policy Upload":
        page_setup()
    elif page == "Claim Upload":
        page_submit()
    elif page == "History":
        page_history()


if __name__ == "__main__":
    main()
