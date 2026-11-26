"""AuthStore — the AutoResearch web login layer (SQLite + bcrypt).

Ported from vibe_programming's ``web/backend/app/auth_store.py`` and trimmed to what
AutoResearch needs: the engine runs ONE session at a time (single-session-serial), so
vibe's multi-tenant ``session_owners`` table is dropped (YAGNI). Three tables:

- ``users``          (username PK, password_hash, is_admin, created_at)  -- approved accounts
- ``pending_users``  (username PK, password_hash, created_at)            -- awaiting admin approval
- ``auth_sessions``  (token PK, username, created_at, expires_at)        -- login sessions

Passwords are bcrypt-hashed directly (no passlib). All methods are synchronous. Thread
safety: a single ``check_same_thread=False`` connection serialised through an ``RLock``
(FastAPI runs sync routes on a threadpool; concurrent cursor use on one sqlite3
connection otherwise raises ``InterfaceError``). bcrypt's slow hash runs OUTSIDE the
lock so a login never blocks concurrent auth checks.
"""
from __future__ import annotations

import os
import secrets
import sqlite3
import threading
import time
from datetime import datetime, timezone

import bcrypt

# Run one bcrypt.checkpw even for a non-existent user to blunt username-enumeration
# timing differences; a fixed valid dummy hash for that path.
_DUMMY_HASH = b"$2b$12$OmZwH2rdS6gbNVr7fUeXP.uDj82bs1TsNi6kBKRXMIPPOWuWhO04C"

# Default login-token lifetime (seconds): 7 days.
DEFAULT_TOKEN_TTL = 7 * 24 * 3600
MAX_PASSWORD_BYTES = 72


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def _now() -> float:
    return time.time()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


class AuthStore:
    """SQLite auth data layer. Single connection + RLock (threadpool-safe)."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        parent = os.path.dirname(os.path.abspath(db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    is_admin INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )"""
            )
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS pending_users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )"""
            )
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS auth_sessions (
                    token TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at REAL NOT NULL
                )"""
            )

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ users
    def create_user(self, username: str, password: str, *, is_admin: bool = False) -> None:
        """Create an approved account. Existing username -> sqlite3.IntegrityError."""
        password_hash = _hash_password(password)  # slow bcrypt outside the lock
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO users (username, password_hash, is_admin, created_at) VALUES (?, ?, ?, ?)",
                (username, password_hash, 1 if is_admin else 0, _iso(_now())),
            )

    def user_exists(self, username: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM users WHERE username = ?", (username,)
            ).fetchone()
        return row is not None

    def pending_user_exists(self, username: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM pending_users WHERE username = ?", (username,)
            ).fetchone()
        return row is not None

    def create_pending_user(self, username: str, password: str) -> None:
        """Submit a registration request. Existing username/pending -> IntegrityError."""
        password_hash = _hash_password(password)  # slow bcrypt outside the lock
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO pending_users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, password_hash, _iso(_now())),
            )

    def list_pending_users(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT username, created_at FROM pending_users ORDER BY created_at ASC"
            ).fetchall()
        return [{"username": r["username"], "created_at": r["created_at"]} for r in rows]

    def approve_pending_user(self, username: str) -> bool:
        """Move a pending user into ``users``. Missing -> False; name clash -> IntegrityError."""
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT password_hash, created_at FROM pending_users WHERE username = ?",
                (username,),
            ).fetchone()
            if not row:
                return False
            self._conn.execute(
                "INSERT INTO users (username, password_hash, is_admin, created_at) VALUES (?, ?, 0, ?)",
                (username, row["password_hash"], row["created_at"]),
            )
            self._conn.execute("DELETE FROM pending_users WHERE username = ?", (username,))
        return True

    def reject_pending_user(self, username: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM pending_users WHERE username = ?", (username,))
        return cur.rowcount > 0

    def get_user(self, username: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT username, is_admin FROM users WHERE username = ?", (username,)
            ).fetchone()
        if not row:
            return None
        return {"username": row["username"], "is_admin": bool(row["is_admin"])}

    def is_admin(self, username: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT is_admin FROM users WHERE username = ?", (username,)
            ).fetchone()
        return bool(row["is_admin"]) if row else False

    def verify_password(self, username: str, password: str) -> bool:
        if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
            return False
        with self._lock:
            row = self._conn.execute(
                "SELECT password_hash FROM users WHERE username = ?", (username,)
            ).fetchone()
        # bcrypt check outside the lock (slow)
        if not row:
            # run a checkpw for a missing user too, to blunt enumeration timing.
            bcrypt.checkpw(password.encode("utf-8"), _DUMMY_HASH)
            return False
        return _verify_password(password, row["password_hash"])

    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """Old password correct -> update hash, return True; else no-op, return False."""
        if not self.verify_password(username, old_password):
            return False
        new_hash = _hash_password(new_password)
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE users SET password_hash = ? WHERE username = ?", (new_hash, username)
            )
        return True

    # ------------------------------------------------------------------ tokens
    def create_token(self, username: str, *, ttl: float = DEFAULT_TOKEN_TTL) -> str:
        token = secrets.token_urlsafe(32)
        now = _now()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO auth_sessions (token, username, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (token, username, _iso(now), now + ttl),
            )
        return token

    def resolve_token(self, token: str | None) -> str | None:
        """Valid + unexpired -> username; invalid/expired -> None (expired is cleaned up)."""
        if not token:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT username, expires_at FROM auth_sessions WHERE token = ?", (token,)
            ).fetchone()
        if not row:
            return None
        if row["expires_at"] <= _now():
            self.revoke_token(token)
            return None
        return row["username"]

    def revoke_token(self, token: str | None) -> None:
        if not token:
            return
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM auth_sessions WHERE token = ?", (token,))


__all__ = ["AuthStore", "DEFAULT_TOKEN_TTL"]
