"""#5 login/auth: SQLite store + FastAPI endpoints (register -> admin approval -> login).

Auth is OPT-IN (AUTORESEARCH_WEB_AUTH=1). Tests toggle ``auth.AUTH_ENABLED`` and inject a
temp AuthStore; they do NOT enter TestClient as a context manager, so startup events
(ENGINE.start) never fire -- same pattern as the other web endpoint tests.
"""
from __future__ import annotations

import sqlite3

import pytest
from starlette.testclient import TestClient

from autoresearch.web import auth, server
from autoresearch.web.auth_store import AuthStore


@pytest.fixture
def store(tmp_path):
    s = AuthStore(str(tmp_path / "auth.db"))
    yield s
    s.close()


@pytest.fixture
def client(monkeypatch, store):
    monkeypatch.setattr(auth, "AUTH_ENABLED", True)
    monkeypatch.setattr(auth, "_STORE", store)
    return TestClient(server.app)


# --------------------------------------------------------------------- store unit
def test_register_approval_and_login_flow(store):
    store.create_pending_user("alice", "hunter2")
    assert store.pending_user_exists("alice") and not store.user_exists("alice")
    # a pending user cannot authenticate (not in the users table)
    assert store.verify_password("alice", "hunter2") is False

    assert store.approve_pending_user("alice") is True
    assert store.user_exists("alice") and not store.pending_user_exists("alice")
    assert store.verify_password("alice", "hunter2") is True
    assert store.verify_password("alice", "wrong") is False
    assert store.is_admin("alice") is False


def test_reject_pending_user(store):
    store.create_pending_user("bob", "secret1")
    assert store.reject_pending_user("bob") is True
    assert not store.pending_user_exists("bob")
    assert store.reject_pending_user("bob") is False  # already gone


def test_token_lifecycle_and_expiry(store):
    store.create_user("carol", "passpass")
    tok = store.create_token("carol")
    assert store.resolve_token(tok) == "carol"
    store.revoke_token(tok)
    assert store.resolve_token(tok) is None
    # an already-expired token resolves to None (and is cleaned up)
    expired = store.create_token("carol", ttl=-1)
    assert store.resolve_token(expired) is None


def test_duplicate_user_raises(store):
    store.create_user("dave", "passpass")
    with pytest.raises(sqlite3.IntegrityError):
        store.create_user("dave", "other1")


def test_bootstrap_admin_idempotent(monkeypatch, store):
    monkeypatch.setenv("AUTORESEARCH_ADMIN_USER", "root")
    monkeypatch.setenv("AUTORESEARCH_ADMIN_PASSWORD", "rootpw12")
    auth.bootstrap_admin(store)
    assert store.user_exists("root") and store.is_admin("root")
    auth.bootstrap_admin(store)  # second call is a no-op, not an error
    assert store.verify_password("root", "rootpw12")


# --------------------------------------------------------------------- endpoints
def test_unauthenticated_api_is_401(client):
    assert client.get("/api/state").status_code == 401


def test_unauthenticated_page_redirects_to_login(client):
    r = client.get("/", headers={"accept": "text/html"}, follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"] == "/login"


def test_login_page_is_public(client):
    r = client.get("/login")
    assert r.status_code == 200 and "Sign in" in r.text


def test_register_requires_admin_approval_before_login(client, store):
    # self-register -> pending, cannot log in yet
    assert client.post("/api/auth/register",
                       json={"username": "erin", "password": "hunter2"}).status_code == 200
    assert client.post("/api/auth/login",
                       json={"username": "erin", "password": "hunter2"}).status_code == 401

    # an admin approves via the API
    store.create_user("admin", "adminpw1", is_admin=True)
    assert client.post("/api/auth/login",
                       json={"username": "admin", "password": "adminpw1"}).status_code == 200
    ap = client.post("/api/auth/admin/pending-users/erin/approve")
    assert ap.status_code == 200
    client.post("/api/auth/logout")

    # now erin can log in and is authenticated
    r = client.post("/api/auth/login", json={"username": "erin", "password": "hunter2"})
    assert r.status_code == 200 and r.json() == {"username": "erin", "is_admin": False}
    assert client.get("/api/state").status_code == 200          # gate now lets her through
    assert client.get("/api/auth/me").json()["username"] == "erin"


def test_non_admin_cannot_access_admin_endpoints(client, store):
    store.create_user("frank", "frankpw1")
    client.post("/api/auth/login", json={"username": "frank", "password": "frankpw1"})
    assert client.get("/api/auth/admin/pending-users").status_code == 403


@pytest.mark.parametrize("password", ["a" * 72, "测" * 24])
def test_password_at_bcrypt_byte_limit_can_register_and_login(client, store, password):
    assert client.post("/api/auth/register",
                       json={"username": "boundary", "password": password}).status_code == 200
    assert store.approve_pending_user("boundary")
    assert client.post("/api/auth/login",
                       json={"username": "boundary", "password": password}).status_code == 200


@pytest.mark.parametrize("password", ["a" * 73, "测" * 25])
def test_registration_rejects_password_over_bcrypt_byte_limit(client, store, password):
    r = client.post("/api/auth/register", json={"username": "oversize", "password": password})
    assert r.status_code == 400
    assert not store.pending_user_exists("oversize")


@pytest.mark.parametrize("password", ["a" * 73, "测" * 25])
@pytest.mark.parametrize("username", ["existing", "missing"])
def test_login_rejects_password_over_bcrypt_byte_limit(client, store, password, username):
    store.create_user("existing", "a" * 72)
    r = client.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == 401
    assert auth.COOKIE_NAME not in r.cookies


def test_admin_can_list_and_reject_pending(client, store):
    store.create_user("admin", "adminpw1", is_admin=True)
    client.post("/api/auth/register", json={"username": "gina", "password": "hunter2"})
    client.post("/api/auth/login", json={"username": "admin", "password": "adminpw1"})
    listing = client.get("/api/auth/admin/pending-users").json()
    assert [u["username"] for u in listing] == ["gina"]
    assert client.post("/api/auth/admin/pending-users/gina/reject").status_code == 200
    assert client.get("/api/auth/admin/pending-users").json() == []


def test_auth_disabled_leaves_endpoints_open(monkeypatch, store):
    # the default (AUTH_ENABLED False, no shared token) must stay open -- no regression.
    monkeypatch.setattr(auth, "AUTH_ENABLED", False)
    monkeypatch.setattr(server, "_WEB_TOKEN", None)
    monkeypatch.setattr(auth, "_STORE", store)
    c = TestClient(server.app)
    assert c.get("/api/state").status_code == 200
