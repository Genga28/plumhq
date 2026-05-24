"""Plum claims services package.

Configures logging on import so every module under `services.*` shares a
consistent, terminal-friendly logger.
"""

from __future__ import annotations

import logging
import os
import sys


def _setup_logging() -> None:
    """One-time logger setup. Idempotent — safe to call multiple times
    (Streamlit re-imports the module on every rerun)."""
    root = logging.getLogger("plum")
    if root.handlers:
        return  # already configured

    # Windows consoles default to cp1252 and can't encode characters like
    # Rs/=/etc. Force UTF-8 on stdout (Python 3.7+). errors='replace' is the
    # last-resort fallback if even UTF-8 reconfiguration is rejected.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    level_name = os.environ.get("PLUM_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s.%(msecs)03d  %(levelname)-5s  %(name)-22s  %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False


_setup_logging()


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the `plum.*` namespace."""
    return logging.getLogger(f"plum.{name}")
