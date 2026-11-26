"""
MossObservationStore — MOSS's OWN cross-session accumulation store (Mode B).

Separate from the Spec Store: this holds the compact structured SIGNALS MOSS
accumulates across sessions to detect recurring patterns, plus an L3-alert
log for the dev team. NOT raw traces — only small structured rows.

SYNCHRONOUS sqlite3 (check_same_thread=False): tiny fast writes, no
concurrency in our tests/usage.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_DDL = """
CREATE TABLE IF NOT EXISTS signals (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT,
    agent        TEXT,
    reason       TEXT,
    task_mode    TEXT,
    detail_json  TEXT,
    created_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_signals_agent_reason ON signals(agent, reason);
CREATE INDEX IF NOT EXISTS idx_signals_session ON signals(session_id);
CREATE TABLE IF NOT EXISTS l3_alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT,
    alert_kind  TEXT,
    detail_json TEXT,
    created_at  TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MossObservationStore:
    """SQLite-backed store for MOSS's cross-session signals + L3 alerts."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA busy_timeout=5000")
        for stmt in _DDL.split(";"):
            s = stmt.strip()
            if s:
                self._db.execute(s)
        self._db.commit()

    def record_signal(
        self,
        session_id: str,
        agent: str,
        reason: str,
        detail: Optional[dict] = None,
        task_mode: Optional[str] = None,
    ) -> None:
        self._db.execute(
            "INSERT INTO signals (session_id, agent, reason, task_mode, detail_json, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (session_id, agent, reason, task_mode, json.dumps(detail or {}), _now()),
        )
        self._db.commit()

    def distinct_sessions(self, agent: str, reason: str) -> int:
        row = self._db.execute(
            "SELECT COUNT(DISTINCT session_id) FROM signals WHERE agent=? AND reason=?",
            (agent, reason),
        ).fetchone()
        return int(row[0]) if row else 0

    def signal_count(self, agent: str, reason: str) -> int:
        row = self._db.execute(
            "SELECT COUNT(*) FROM signals WHERE agent=? AND reason=?",
            (agent, reason),
        ).fetchone()
        return int(row[0]) if row else 0

    def flag_l3(self, session_id: str, alert_kind: str, detail: Optional[dict] = None) -> None:
        self._db.execute(
            "INSERT INTO l3_alerts (session_id, alert_kind, detail_json, created_at) "
            "VALUES (?,?,?,?)",
            (session_id, alert_kind, json.dumps(detail or {}), _now()),
        )
        self._db.commit()

    def list_alerts(self) -> list[dict]:
        rows = self._db.execute(
            "SELECT id, session_id, alert_kind, detail_json, created_at FROM l3_alerts ORDER BY id"
        ).fetchall()
        return [
            {
                "id": r[0],
                "session_id": r[1],
                "alert_kind": r[2],
                "detail": json.loads(r[3]),
                "created_at": r[4],
            }
            for r in rows
        ]

    def close(self) -> None:
        self._db.close()


__all__ = ["MossObservationStore"]
