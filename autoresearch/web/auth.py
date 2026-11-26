"""Login/registration/admin-approval for the AutoResearch web viewer.

OPT-IN: only active when ``AUTORESEARCH_WEB_AUTH=1``. When off, the viewer keeps its
previous behaviour (open localhost, or the ``AUTORESEARCH_WEB_TOKEN`` shared-token gate),
so existing dev/test flows are unchanged.

When on:
  - every request must carry a valid login-session cookie, EXCEPT the login/register
    page and the login/register/logout API (the public allow-list). Unauthenticated
    page loads redirect to ``/login``; unauthenticated API calls get 401.
  - a bootstrap admin account is created on startup from ``AUTORESEARCH_ADMIN_USER`` /
    ``AUTORESEARCH_ADMIN_PASSWORD`` (a random password is generated + logged once if the
    latter is unset).
  - new users self-register into ``pending_users`` and an admin approves them before
    they can log in.

The model (SQLite users/pending_users/auth_sessions, bcrypt, admin approval, cookie
tokens) is ported from vibe_programming.
"""
from __future__ import annotations

import logging
import os
import re
import secrets
import sqlite3
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from autoresearch.web.auth_store import DEFAULT_TOKEN_TTL, MAX_PASSWORD_BYTES, AuthStore

logger = logging.getLogger(__name__)

AUTH_ENABLED = os.environ.get("AUTORESEARCH_WEB_AUTH") == "1"
COOKIE_NAME = "ar_session"
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.@-]{3,64}$")
MIN_PASSWORD_LEN = 6

# Requests to these exact paths never require authentication (login/register flow +
# the login page's own assets). Everything else is gated when AUTH_ENABLED.
_PUBLIC_PATHS = frozenset({
    "/login",
    "/api/auth/login",
    "/api/auth/register",
    "/api/auth/logout",
    "/favicon.ico",
})

_STORE: AuthStore | None = None


def _default_db_path() -> str:
    workdir = os.environ.get("AUTORESEARCH_WEB_WORKDIR", "/tmp/autoresearch-web")
    return os.environ.get("AUTORESEARCH_AUTH_DB", str(Path(workdir) / "auth.db"))


def get_store() -> AuthStore:
    """Process-wide AuthStore singleton (lazy)."""
    global _STORE
    if _STORE is None:
        _STORE = AuthStore(_default_db_path())
    return _STORE


def bootstrap_admin(store: AuthStore | None = None) -> None:
    """Ensure the admin account exists (idempotent). Password from
    ``AUTORESEARCH_ADMIN_PASSWORD`` or a generated one that is logged once."""
    store = store or get_store()
    username = os.environ.get("AUTORESEARCH_ADMIN_USER", "admin").strip() or "admin"
    if store.user_exists(username):
        return
    password = os.environ.get("AUTORESEARCH_ADMIN_PASSWORD")
    generated = password is None
    if generated:
        password = secrets.token_urlsafe(12)
    store.create_user(username, password, is_admin=True)
    if generated:
        logger.warning(
            "AUTORESEARCH auth: created admin %r with GENERATED password %r "
            "(set AUTORESEARCH_ADMIN_PASSWORD to choose your own)", username, password
        )
    else:
        logger.info("AUTORESEARCH auth: created admin %r", username)


# ------------------------------------------------------------------ request models
class LoginReq(BaseModel):
    username: str
    password: str


class RegisterReq(BaseModel):
    username: str
    password: str


class CurrentUser(BaseModel):
    username: str
    is_admin: bool


class PendingUser(BaseModel):
    username: str
    created_at: str


# ------------------------------------------------------------------ dependencies
def current_user(request: Request) -> CurrentUser:
    store = get_store()
    username = store.resolve_token(request.cookies.get(COOKIE_NAME))
    if not username:
        raise HTTPException(status_code=401, detail="not authenticated")
    return CurrentUser(username=username, is_admin=store.is_admin(username))


def require_admin(user: CurrentUser = Depends(current_user)) -> CurrentUser:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="admin required")
    return user


# ------------------------------------------------------------------ validation
def _validate_username(username: str) -> None:
    if not USERNAME_RE.fullmatch(username):
        raise HTTPException(
            status_code=400,
            detail="username must be 3-64 chars: letters, numbers, _, ., -, @",
        )


def _validate_password(password: str) -> None:
    if len(password) < MIN_PASSWORD_LEN:
        raise HTTPException(status_code=400, detail="password must be at least 6 characters")
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise HTTPException(status_code=400, detail="password must be at most 72 UTF-8 bytes")


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        max_age=int(DEFAULT_TOKEN_TTL),
        httponly=True,
        secure=os.environ.get("AUTORESEARCH_COOKIE_SECURE") == "1",
        samesite="lax",
        path="/",
    )


# ------------------------------------------------------------------ router
router = APIRouter(prefix="/api/auth")


@router.post("/register")
async def register(req: RegisterReq) -> dict:
    username = req.username.strip()
    _validate_username(username)
    _validate_password(req.password)
    store = get_store()
    if store.user_exists(username) or store.pending_user_exists(username):
        raise HTTPException(status_code=409, detail="username already exists or pending")
    try:
        store.create_pending_user(username, req.password)
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="username already exists or pending")
    return {"ok": True, "pending": True}


@router.post("/login", response_model=CurrentUser)
async def login(req: LoginReq, response: Response) -> CurrentUser:
    store = get_store()
    if not store.verify_password(req.username, req.password):
        raise HTTPException(status_code=401, detail="invalid username or password")
    _set_session_cookie(response, store.create_token(req.username))
    return CurrentUser(username=req.username, is_admin=store.is_admin(req.username))


@router.post("/logout")
async def logout(request: Request, response: Response) -> dict:
    get_store().revoke_token(request.cookies.get(COOKIE_NAME))
    response.delete_cookie(key=COOKIE_NAME, path="/")
    return {"ok": True}


@router.get("/me", response_model=CurrentUser)
async def me(user: CurrentUser = Depends(current_user)) -> CurrentUser:
    return user


@router.get("/admin/pending-users", response_model=list[PendingUser])
async def list_pending_users(_: CurrentUser = Depends(require_admin)) -> list[dict]:
    return get_store().list_pending_users()


@router.post("/admin/pending-users/{username}/approve")
async def approve_pending_user(username: str, _: CurrentUser = Depends(require_admin)) -> dict:
    try:
        approved = get_store().approve_pending_user(username)
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="username already exists")
    if not approved:
        raise HTTPException(status_code=404, detail="pending user not found")
    return {"ok": True}


@router.post("/admin/pending-users/{username}/reject")
async def reject_pending_user(username: str, _: CurrentUser = Depends(require_admin)) -> dict:
    if not get_store().reject_pending_user(username):
        raise HTTPException(status_code=404, detail="pending user not found")
    return {"ok": True}


# ------------------------------------------------------------------ gate
def is_authenticated(request: Request) -> bool:
    return get_store().resolve_token(request.cookies.get(COOKIE_NAME)) is not None


def is_public_path(path: str) -> bool:
    """Auth-exempt paths: the public allow-list plus the login page's static assets."""
    return path in _PUBLIC_PATHS or path.startswith("/static/login")


__all__ = [
    "AUTH_ENABLED", "COOKIE_NAME", "router", "bootstrap_admin", "get_store",
    "current_user", "require_admin", "is_authenticated", "is_public_path", "CurrentUser",
]
