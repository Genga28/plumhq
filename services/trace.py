"""TraceLogger: append-only structured log per claim. The observability backbone.

Every agent calls `tracer.record(...)` to write an event. Events are persisted
to the `traces` table and also kept in-memory so the final FinalDecision can
return the full trace alongside the decision.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Any, Optional

from .db import save_trace_event
from .models import AgentStatus, TraceEvent


class TraceLogger:
    """One instance per claim. Thread-safe sequence counter."""

    def __init__(self, claim_id: str, persist: bool = True) -> None:
        self.claim_id = claim_id
        self._persist = persist
        self._lock = threading.Lock()
        self._sequence = 0
        self.events: list[TraceEvent] = []

    def _next_seq(self) -> int:
        with self._lock:
            self._sequence += 1
            return self._sequence

    def record(
        self,
        agent: str,
        action: str,
        status: AgentStatus,
        payload: Optional[dict[str, Any]] = None,
        confidence: Optional[float] = None,
        duration_ms: int = 0,
    ) -> TraceEvent:
        event = TraceEvent(
            sequence=self._next_seq(),
            agent=agent,
            action=action,
            status=status,
            payload=payload or {},
            confidence=confidence,
            duration_ms=duration_ms,
            created_at=datetime.utcnow(),
        )
        self.events.append(event)
        if self._persist and self.claim_id:
            try:
                save_trace_event(self.claim_id, event)
            except Exception:
                # Trace persistence must never crash the pipeline.
                pass
        return event


class Timer:
    """Context manager for measuring agent duration. Use inside BaseAgent.run."""

    def __init__(self) -> None:
        self.elapsed_ms = 0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed_ms = int((time.perf_counter() - self._start) * 1000)
