"""Tests for neo4j_client module — driver config, session, schema setup."""

from __future__ import annotations

from unittest.mock import MagicMock

import src.neo4j_client as neo4j_mod


class TestNeo4jClientInit:
    def test_driver_created_with_timeout_settings(self, monkeypatch) -> None:
        captured = {}

        def _fake_driver(uri, auth, **kwargs):
            captured["uri"] = uri
            captured["auth"] = auth
            captured.update(kwargs)
            return MagicMock()

        monkeypatch.setattr(neo4j_mod, "GraphDatabase", type("GD", (), {"driver": staticmethod(_fake_driver)}))
        monkeypatch.setattr(neo4j_mod.settings, "neo4j_uri", "bolt://test:7687")
        monkeypatch.setattr(neo4j_mod.settings, "neo4j_username", "user")
        monkeypatch.setattr(neo4j_mod.settings, "neo4j_password", "pass")

        neo4j_mod.Neo4jClient()

        assert captured["uri"] == "bolt://test:7687"
        assert captured["auth"] == ("user", "pass")
        assert captured["connection_timeout"] == 30
        assert captured["max_connection_pool_size"] == 50
        assert captured["connection_acquisition_timeout"] == 60


class TestNeo4jClientSession:
    def test_session_context_manager_yields_and_closes(self) -> None:
        mock_session = MagicMock()
        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session

        client = neo4j_mod.Neo4jClient.__new__(neo4j_mod.Neo4jClient)
        client.driver = mock_driver

        with client.session() as session:
            assert session is mock_session

        mock_session.close.assert_called_once()

    def test_session_closes_on_exception(self) -> None:
        mock_session = MagicMock()
        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session

        client = neo4j_mod.Neo4jClient.__new__(neo4j_mod.Neo4jClient)
        client.driver = mock_driver

        try:
            with client.session():
                raise ValueError("boom")
        except ValueError:
            pass

        mock_session.close.assert_called_once()


class TestNeo4jClientSetupSchema:
    def test_creates_constraints_and_indexes(self) -> None:
        mock_session = MagicMock()
        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session

        client = neo4j_mod.Neo4jClient.__new__(neo4j_mod.Neo4jClient)
        client.driver = mock_driver

        client.setup_schema()

        queries = [call.args[0] for call in mock_session.run.call_args_list]
        assert any("page_id" in q for q in queries)
        assert any("chunk_id" in q for q in queries)
        assert any("entity_id" in q for q in queries)
        assert any("FULLTEXT" in q or "fulltext" in q.lower() for q in queries)
        assert len(queries) >= 8
        mock_session.close.assert_called_once()
