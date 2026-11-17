"""API surface tests (fully offline).

Imports of ``fastapi`` / ``serve_api`` happen inside a fixture so the module
skips cleanly when the ``serve`` extra is absent (CI runs the dependency-light
suite), without any mid-file imports that linters dislike.

The embedding model and the LLM are stubbed, so the suite never downloads
weights or talks to Ollama.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def api():
    pytest.importorskip("fastapi")
    import serve_api

    return serve_api


class _Collection:
    def count(self) -> int:
        return 3


class _StubStore:
    """Minimal stand-in for a Chroma handle (only ``count`` is probed)."""

    _collection = _Collection()


@pytest.fixture
def client(api, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(api, "get_vectorstore", lambda *a, **k: _StubStore())
    monkeypatch.setattr(api, "get_llm", lambda *a, **k: object())
    with TestClient(api.app) as test_client:
        yield test_client


@pytest.fixture
def degraded_client(api, monkeypatch):
    """Startup where the embedding model cannot be loaded."""

    from fastapi.testclient import TestClient

    def boom(*_args, **_kwargs):
        raise OSError("could not reach huggingface.co")

    monkeypatch.setattr(api, "get_vectorstore", boom)
    monkeypatch.setattr(api, "get_llm", lambda *a, **k: object())
    with TestClient(api.app) as test_client:
        yield test_client


def test_routes_are_registered(api):
    paths = {route.path for route in api.app.routes}
    assert {"/healthz", "/readyz", "/ask"} <= paths


def test_healthz_always_answers(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_readyz_reports_ready_when_index_is_present(client):
    payload = client.get("/readyz").json()
    assert payload["ready"] is True
    assert payload["error"] is None


def test_ask_rejects_empty_question(client):
    """Validated before any model or index call happens."""
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_startup_degrades_instead_of_crashing(degraded_client):
    """A missing model must not take the process down (503 with a reason)."""
    assert degraded_client.get("/healthz").status_code == 200

    payload = degraded_client.get("/readyz").json()
    assert payload["ready"] is False
    assert "huggingface" in (payload["error"] or "")

    response = degraded_client.post("/ask", json={"question": "who goes first?"})
    assert response.status_code == 503
    assert "not ready" in response.json()["detail"]
