"""BaseAgent: failure isolation + timing + tracing for every agent.

Every agent inherits and overrides `_run(input)`. The public `run(input)` wraps
that body in try/except, records a trace event, and (on failure) returns a
DegradedResult or whatever the subclass declares as `failure_default()`.

This is how TC011 (graceful degradation) is implemented uniformly.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from .. import get_logger
from ..models import AgentStatus
from ..trace import TraceLogger

log = get_logger("agent")


IN = TypeVar("IN")
OUT = TypeVar("OUT")


class BaseAgent(ABC, Generic[IN, OUT]):
    """Abstract base. Subclasses implement `_run`, `name`, and `failure_default`."""

    #: Short name used in trace events (e.g. "DocumentVerifier").
    name: str = "BaseAgent"

    def __init__(self, tracer: TraceLogger) -> None:
        self.tracer = tracer

    @abstractmethod
    async def _run(self, payload: IN) -> OUT:
        """Subclass body. May raise; the wrapper handles it."""

    @abstractmethod
    def failure_default(self, payload: IN, exc: Exception) -> OUT:
        """What to return when _run raises. Should produce a degraded but usable result."""

    def trace_payload(self, result: OUT) -> dict[str, Any]:
        """Override to customize the trace event payload. Default: dump model."""
        try:
            from pydantic import BaseModel
            if isinstance(result, BaseModel):
                return result.model_dump(mode="json")
        except Exception:
            pass
        return {"result": str(result)[:500]}

    async def run(self, payload: IN, *, simulate_failure: bool = False) -> OUT:
        log.info("%s > started", self.name)
        start = time.perf_counter()
        try:
            if simulate_failure:
                raise RuntimeError(f"Simulated failure in {self.name}")
            result = await self._run(payload)
            elapsed_ms = _ms_since(start)
            payload_dump = self.trace_payload(result)
            self.tracer.record(
                agent=self.name,
                action="run",
                status=AgentStatus.OK,
                payload=payload_dump,
                duration_ms=elapsed_ms,
            )
            log.info("%s OK (%dms) %s", self.name, elapsed_ms, _summary(payload_dump))
            return result
        except Exception as exc:
            elapsed_ms = _ms_since(start)
            degraded = self.failure_default(payload, exc)
            self.tracer.record(
                agent=self.name,
                action="run",
                status=AgentStatus.FAILED,
                payload={"error": str(exc), "exception_type": type(exc).__name__},
                duration_ms=elapsed_ms,
            )
            log.warning("%s FAILED (%dms): %s: %s",
                        self.name, elapsed_ms, type(exc).__name__, exc)
            return degraded


def _ms_since(start: float) -> int:
    """Elapsed time in milliseconds; clamps to 1ms minimum for completed work
    so the UI doesn't show a misleading 0ms for sub-millisecond operations."""
    elapsed = (time.perf_counter() - start) * 1000
    return max(1, int(round(elapsed)))


# Fields we surface in the one-line log summary per agent. Keys are matched
# against whatever the agent's trace_payload returned. Only a small subset is
# logged so the terminal stays readable.
_SUMMARY_KEYS = (
    "claim_id", "decision", "approved_amount", "actual_type",
    "classification_confidence", "quality", "ocr_confidence",
    "patient_name", "diagnosis", "total_amount", "line_items_count",
    "missing_required", "patient_match", "fraud_score", "same_day_count",
    "diagnosis_keys", "excluded_matches", "network_hospital_match",
)


def _summary(payload: dict[str, Any]) -> str:
    """Compact one-line summary of an agent's result for the terminal log."""
    if not payload:
        return ""
    bits: list[str] = []
    for k in _SUMMARY_KEYS:
        if k in payload and payload[k] not in (None, "", [], {}):
            v = payload[k]
            if isinstance(v, list):
                v = f"[{len(v)}]" if len(v) > 3 else v
            bits.append(f"{k}={v}")
    return " ".join(bits)


async def run_parallel(*coros: Any) -> list[Any]:
    """Tiny helper around asyncio.gather with return_exceptions=False.

    Kept here so importers don't need to know about asyncio at the call site.
    """
    return list(await asyncio.gather(*coros))
