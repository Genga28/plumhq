"""SQLite persistence layer. Schema + thin repository functions.

Design notes:
  * SQLite is enough for this assignment's scale; the schema is normalized.
  * Trace events are stored row-per-event so they can be queried in order.
  * Connection is created per-call; SQLite handles concurrent reads fine.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator, Optional

from .models import (
    AgentStatus,
    Decision,
    FinalDecision,
    Member,
    Policy,
    TraceEvent,
)

DB_PATH = os.environ.get("DB_PATH", "claims.db")


SCHEMA = """
CREATE TABLE IF NOT EXISTS policies (
    policy_id   TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    config_json TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS members (
    member_id          TEXT PRIMARY KEY,
    policy_id          TEXT NOT NULL,
    name               TEXT NOT NULL,
    date_of_birth      TEXT,
    gender             TEXT,
    relationship       TEXT,
    join_date          TEXT,
    primary_member_id  TEXT,
    FOREIGN KEY(policy_id) REFERENCES policies(policy_id)
);

CREATE TABLE IF NOT EXISTS claims (
    claim_id        TEXT PRIMARY KEY,
    member_id       TEXT NOT NULL,
    policy_id       TEXT NOT NULL,
    category        TEXT NOT NULL,
    claimed_amount  REAL NOT NULL,
    treatment_date  TEXT NOT NULL,
    hospital_name   TEXT,
    status          TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    FOREIGN KEY(member_id) REFERENCES members(member_id),
    FOREIGN KEY(policy_id) REFERENCES policies(policy_id)
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    claim_id        TEXT NOT NULL,
    file_name       TEXT NOT NULL,
    file_path       TEXT,
    declared_type   TEXT,
    actual_type     TEXT,
    quality         TEXT,
    extracted_json  TEXT,
    ocr_confidence  REAL,
    FOREIGN KEY(claim_id) REFERENCES claims(claim_id)
);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id      TEXT PRIMARY KEY,
    claim_id         TEXT NOT NULL,
    decision         TEXT NOT NULL,
    approved_amount  REAL NOT NULL,
    reasons_json     TEXT,
    confidence       REAL,
    breakdown_json   TEXT,
    llm_reasoning    TEXT,
    user_message     TEXT,
    created_at       TEXT NOT NULL,
    FOREIGN KEY(claim_id) REFERENCES claims(claim_id)
);

CREATE TABLE IF NOT EXISTS traces (
    trace_id     TEXT PRIMARY KEY,
    claim_id     TEXT NOT NULL,
    sequence     INTEGER NOT NULL,
    agent        TEXT NOT NULL,
    action       TEXT NOT NULL,
    status       TEXT NOT NULL,
    payload_json TEXT,
    confidence   REAL,
    duration_ms  INTEGER,
    created_at   TEXT NOT NULL,
    FOREIGN KEY(claim_id) REFERENCES claims(claim_id)
);

CREATE INDEX IF NOT EXISTS idx_traces_claim ON traces(claim_id, sequence);
CREATE INDEX IF NOT EXISTS idx_claims_member ON claims(member_id);
CREATE INDEX IF NOT EXISTS idx_members_policy ON members(policy_id);
"""


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def _now() -> str:
    return datetime.utcnow().isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ─────────────────────────────────────────────────────────────────────────────
# Policy + Member
# ─────────────────────────────────────────────────────────────────────────────


def save_policy(policy_id: str, name: str, config: dict[str, Any]) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO policies(policy_id, name, config_json, created_at) VALUES(?,?,?,?)",
            (policy_id, name, json.dumps(config), _now()),
        )


def get_policy(policy_id: str) -> Optional[Policy]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT policy_id, name, config_json FROM policies WHERE policy_id=?",
            (policy_id,),
        ).fetchone()
    if not row:
        return None
    return Policy(policy_id=row["policy_id"], name=row["name"], config=json.loads(row["config_json"]))


def list_policies() -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute("SELECT policy_id, name, created_at FROM policies ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def save_member(member: Member) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO members
               (member_id, policy_id, name, date_of_birth, gender, relationship, join_date, primary_member_id)
               VALUES(?,?,?,?,?,?,?,?)""",
            (
                member.member_id,
                member.policy_id,
                member.name,
                member.date_of_birth.isoformat() if member.date_of_birth else None,
                member.gender,
                member.relationship,
                member.join_date.isoformat() if member.join_date else None,
                member.primary_member_id,
            ),
        )


def get_member(member_id: str) -> Optional[Member]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM members WHERE member_id=?", (member_id,)).fetchone()
    if not row:
        return None
    from datetime import date as _date

    def _d(v: Optional[str]) -> Optional[_date]:
        return _date.fromisoformat(v) if v else None

    return Member(
        member_id=row["member_id"],
        policy_id=row["policy_id"],
        name=row["name"],
        date_of_birth=_d(row["date_of_birth"]),
        gender=row["gender"],
        relationship=row["relationship"],
        join_date=_d(row["join_date"]),
        primary_member_id=row["primary_member_id"],
    )


def list_members(policy_id: Optional[str] = None) -> list[dict[str, Any]]:
    with get_conn() as conn:
        if policy_id:
            rows = conn.execute(
                "SELECT member_id, policy_id, name, relationship FROM members WHERE policy_id=?",
                (policy_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT member_id, policy_id, name, relationship FROM members"
            ).fetchall()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Claim + Documents
# ─────────────────────────────────────────────────────────────────────────────


def create_claim(
    member_id: str,
    policy_id: str,
    category: str,
    claimed_amount: float,
    treatment_date: str,
    hospital_name: Optional[str],
) -> str:
    claim_id = new_id("CLM")
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO claims(claim_id, member_id, policy_id, category, claimed_amount,
                                  treatment_date, hospital_name, status, created_at)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (
                claim_id,
                member_id,
                policy_id,
                category,
                claimed_amount,
                treatment_date,
                hospital_name,
                "PROCESSING",
                _now(),
            ),
        )
    return claim_id


def update_claim_status(claim_id: str, status: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE claims SET status=? WHERE claim_id=?", (status, claim_id))


def save_document(
    claim_id: str,
    file_name: str,
    file_path: Optional[str],
    declared_type: Optional[str],
    actual_type: Optional[str],
    quality: Optional[str],
    extracted: Optional[dict[str, Any]],
    ocr_confidence: Optional[float],
) -> str:
    doc_id = new_id("DOC")
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO documents(doc_id, claim_id, file_name, file_path, declared_type,
                                     actual_type, quality, extracted_json, ocr_confidence)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (
                doc_id,
                claim_id,
                file_name,
                file_path,
                declared_type,
                actual_type,
                quality,
                json.dumps(extracted) if extracted is not None else None,
                ocr_confidence,
            ),
        )
    return doc_id


def list_documents(claim_id: str) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM documents WHERE claim_id=?", (claim_id,)).fetchall()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Decision
# ─────────────────────────────────────────────────────────────────────────────


def save_decision(claim_id: str, decision: FinalDecision) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO decisions(decision_id, claim_id, decision, approved_amount,
                                     reasons_json, confidence, breakdown_json, llm_reasoning,
                                     user_message, created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                new_id("DEC"),
                claim_id,
                decision.decision.value,
                decision.approved_amount,
                json.dumps([r.value for r in decision.rejection_reasons]),
                decision.confidence,
                json.dumps(decision.breakdown),
                decision.llm_reasoning,
                decision.user_message,
                _now(),
            ),
        )


def get_decision(claim_id: str) -> Optional[dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM decisions WHERE claim_id=? ORDER BY created_at DESC LIMIT 1",
            (claim_id,),
        ).fetchone()
    return dict(row) if row else None


def list_claims(limit: int = 50) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT c.claim_id, c.member_id, m.name AS member_name, c.category,
                      c.claimed_amount, c.treatment_date, c.status, c.created_at,
                      d.decision, d.approved_amount
               FROM claims c
               LEFT JOIN members m ON m.member_id = c.member_id
               LEFT JOIN decisions d ON d.claim_id = c.claim_id
               ORDER BY c.created_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def claims_for_member_on_date(member_id: str, on_date: str) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT claim_id, claimed_amount, treatment_date FROM claims WHERE member_id=? AND treatment_date=?",
            (member_id, on_date),
        ).fetchall()
    return [dict(r) for r in rows]


def claims_for_member_in_month(member_id: str, year: int, month: int) -> list[dict[str, Any]]:
    prefix = f"{year:04d}-{month:02d}"
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT claim_id, treatment_date FROM claims WHERE member_id=? AND substr(treatment_date,1,7)=?",
            (member_id, prefix),
        ).fetchall()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Trace
# ─────────────────────────────────────────────────────────────────────────────


def save_trace_event(claim_id: str, event: TraceEvent) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO traces(trace_id, claim_id, sequence, agent, action, status,
                                  payload_json, confidence, duration_ms, created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                new_id("TRC"),
                claim_id,
                event.sequence,
                event.agent,
                event.action,
                event.status.value if isinstance(event.status, AgentStatus) else str(event.status),
                json.dumps(event.payload, default=str),
                event.confidence,
                event.duration_ms,
                event.created_at.isoformat(),
            ),
        )


def get_trace(claim_id: str) -> list[TraceEvent]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM traces WHERE claim_id=? ORDER BY sequence ASC",
            (claim_id,),
        ).fetchall()
    return [
        TraceEvent(
            sequence=r["sequence"],
            agent=r["agent"],
            action=r["action"],
            status=AgentStatus(r["status"]),
            payload=json.loads(r["payload_json"]) if r["payload_json"] else {},
            confidence=r["confidence"],
            duration_ms=r["duration_ms"] or 0,
            created_at=datetime.fromisoformat(r["created_at"]),
        )
        for r in rows
    ]
