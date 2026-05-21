from __future__ import annotations

from fastapi.testclient import TestClient

import src.main as main


def test_health_endpoint() -> None:
    with TestClient(main.app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ready_endpoint_reports_neo4j_failure(monkeypatch) -> None:
    class _FakeClient:
        def verify_connectivity(self) -> None:
            raise RuntimeError("neo4j down")

        def close(self) -> None:
            return None

    monkeypatch.setattr(main, "neo4j_client", _FakeClient())

    with TestClient(main.app) as client:
        resp = client.get("/ready")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["neo4j"]["ok"] is False


def test_metrics_endpoint_shape() -> None:
    with TestClient(main.app) as client:
        resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "hf_jobs_total" in resp.text


def test_guard_requires_api_key_when_configured(monkeypatch) -> None:
    monkeypatch.setattr(main.settings, "app_api_key", "secret")

    with TestClient(main.app) as client:
        # /query is protected by guard dependency
        resp = client.post("/query", json={"question": "hi there", "top_k": 1})

    assert resp.status_code == 401


def test_guard_allows_with_correct_api_key(monkeypatch) -> None:
    monkeypatch.setattr(main.settings, "app_api_key", "secret")

    def _fake_query_graph(_question: str, _top_k: int):
        return type("R", (), {"answer": "ok", "citations": []})()

    monkeypatch.setattr(main, "query_graph", _fake_query_graph)

    with TestClient(main.app) as client:
        resp = client.post(
            "/query",
            json={"question": "hello world", "top_k": 1},
            headers={"X-API-Key": "secret"},
        )

    assert resp.status_code == 200
    assert resp.json()["answer"] == "ok"


def test_rate_limit_rejects_excess(monkeypatch) -> None:
    monkeypatch.setattr(main.settings, "app_api_key", None)

    class _Limiter:
        def allow(self, _key: str) -> tuple[bool, int]:
            return False, 0

    monkeypatch.setattr(main, "rate_limiter", _Limiter())

    with TestClient(main.app) as client:
        resp = client.post("/query", json={"question": "hello world", "top_k": 1})

    assert resp.status_code == 429


def test_with_request_context_sets_and_resets_request_id(monkeypatch) -> None:
    captured = {"value": None}

    def _fake_set_request_id(value: str):
        captured["value"] = value
        return "token"

    def _fake_reset_request_id(token):
        captured["reset"] = token

    monkeypatch.setattr(main, "set_request_id", _fake_set_request_id)
    monkeypatch.setattr(main, "reset_request_id", _fake_reset_request_id)

    class _Client:
        host = "127.0.0.1"

    class _Req:
        headers = {"X-Request-ID": "rid-001"}
        client = _Client()

    rid, token = main._with_request_context(_Req())

    assert rid == "rid-001"
    assert token == "token"
    assert captured["value"] == "rid-001"


def test_main_uses_json_log_setting_for_configure_logging(monkeypatch) -> None:
    calls = {}

    def _fake_configure_logging(level_name: str, json_logs: bool = False):
        calls["level_name"] = level_name
        calls["json_logs"] = json_logs

    monkeypatch.setattr(main, "configure_logging", _fake_configure_logging)
    monkeypatch.setattr(main.settings, "log_level", "DEBUG")
    monkeypatch.setattr(main.settings, "json_logs", True)

    main.configure_logging(main.settings.log_level, json_logs=main.settings.json_logs)

    assert calls["level_name"] == "DEBUG"
    assert calls["json_logs"] is True
