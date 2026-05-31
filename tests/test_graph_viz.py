"""Tests for graph visualization API endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    from src.main import app
    return TestClient(app)


class TestEntityNeighborhood:
    @patch("src.dashboard.graph_viz.neo4j_client")
    def test_returns_nodes_and_edges(self, mock_neo4j, client):
        mock_session = MagicMock()
        mock_session.run.return_value = iter([
            {
                "center_name": "Hồ Chí Minh",
                "center_type": "Person",
                "neighbors": [
                    {"name": "Việt Nam", "type": "Location", "rel": "co_mentioned"},
                    {"name": "Đảng Cộng sản", "type": "Organization", "rel": "co_mentioned"},
                ],
            }
        ])
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.get("/dashboard/api/graph/entity-neighborhood", params={"name": "Hồ Chí Minh"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["nodes"]) == 3
        assert data["nodes"][0]["center"] is True
        assert len(data["edges"]) == 2

    @patch("src.dashboard.graph_viz.neo4j_client")
    def test_entity_not_found(self, mock_neo4j, client):
        mock_session = MagicMock()
        mock_session.run.return_value = iter([{"center_name": None, "center_type": None, "neighbors": []}])
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.get("/dashboard/api/graph/entity-neighborhood", params={"name": "NonExistent"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["nodes"] == []
        assert "not found" in data["error"]

    @patch("src.dashboard.graph_viz.neo4j_client")
    def test_empty_result(self, mock_neo4j, client):
        mock_session = MagicMock()
        mock_session.run.return_value = iter([])
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.get("/dashboard/api/graph/entity-neighborhood", params={"name": "X"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["nodes"] == []

    @patch("src.dashboard.graph_viz.neo4j_client")
    def test_handles_db_error(self, mock_neo4j, client):
        mock_neo4j.session.return_value.__enter__ = MagicMock(
            side_effect=RuntimeError("connection lost")
        )
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.get("/dashboard/api/graph/entity-neighborhood", params={"name": "test"})
        assert resp.status_code == 500
        assert "error" in resp.json()

    @patch("src.dashboard.graph_viz.neo4j_client")
    def test_skips_neighbors_without_name(self, mock_neo4j, client):
        mock_session = MagicMock()
        mock_session.run.return_value = iter([
            {
                "center_name": "A",
                "center_type": "Person",
                "neighbors": [
                    {"name": "", "type": "Location", "rel": "co_mentioned"},
                    {"name": None, "type": "Org", "rel": "co_mentioned"},
                    {"name": "B", "type": "Person", "rel": "co_mentioned"},
                ],
            }
        ])
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.get("/dashboard/api/graph/entity-neighborhood", params={"name": "A"})
        data = resp.json()
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1


class TestPageGraph:
    @patch("src.dashboard.graph_viz.neo4j_client")
    def test_returns_page_links(self, mock_neo4j, client):
        mock_session = MagicMock()
        mock_session.run.return_value = iter([
            {
                "center_title": "Hà Nội",
                "center_url": "https://vi.wikipedia.org/wiki/Hà_Nội",
                "links": [
                    {"title": "Việt Nam", "url": "https://vi.wikipedia.org/wiki/Việt_Nam"},
                    {"title": "Thăng Long", "url": "https://vi.wikipedia.org/wiki/Thăng_Long"},
                ],
            }
        ])
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.get("/dashboard/api/graph/page-graph", params={"title": "Hà Nội"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["nodes"]) == 3
        assert data["nodes"][0]["center"] is True
        assert len(data["edges"]) == 2
        assert data["edges"][0]["type"] == "LINKS_TO"

    @patch("src.dashboard.graph_viz.neo4j_client")
    def test_page_not_found(self, mock_neo4j, client):
        mock_session = MagicMock()
        mock_session.run.return_value = iter([{"center_title": None, "center_url": None, "links": []}])
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.get("/dashboard/api/graph/page-graph", params={"title": "NoPage"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["nodes"] == []
        assert "not found" in data["error"]

    @patch("src.dashboard.graph_viz.neo4j_client")
    def test_handles_db_error(self, mock_neo4j, client):
        mock_neo4j.session.return_value.__enter__ = MagicMock(
            side_effect=RuntimeError("timeout")
        )
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.get("/dashboard/api/graph/page-graph", params={"title": "test"})
        assert resp.status_code == 500
        assert "error" in resp.json()

    @patch("src.dashboard.graph_viz.neo4j_client")
    def test_skips_links_without_title(self, mock_neo4j, client):
        mock_session = MagicMock()
        mock_session.run.return_value = iter([
            {
                "center_title": "Main",
                "center_url": "http://x",
                "links": [
                    {"title": "", "url": "http://a"},
                    {"title": None, "url": "http://b"},
                    {"title": "Valid", "url": "http://c"},
                ],
            }
        ])
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.get("/dashboard/api/graph/page-graph", params={"title": "Main"})
        data = resp.json()
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1


class TestEntityTypesDistribution:
    @patch("src.dashboard.graph_viz.neo4j_client")
    def test_returns_distribution(self, mock_neo4j, client):
        mock_session = MagicMock()
        mock_record = {
            "count": 100, "count2": 50, "count3": 80, "count4": 30,
            "type": "Person", "type2": "Organization", "type3": "Location", "type4": "Work",
        }
        mock_session.run.return_value.single.return_value = mock_record
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.get("/dashboard/api/graph/entity-types-distribution")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["distribution"]) == 4
        assert data["distribution"][0] == {"type": "Person", "count": 100}
        assert data["distribution"][3] == {"type": "Work", "count": 30}

    @patch("src.dashboard.graph_viz.neo4j_client")
    def test_empty_result(self, mock_neo4j, client):
        mock_session = MagicMock()
        mock_session.run.return_value.single.return_value = None
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.get("/dashboard/api/graph/entity-types-distribution")
        assert resp.status_code == 200
        data = resp.json()
        assert data["distribution"] == []

    @patch("src.dashboard.graph_viz.neo4j_client")
    def test_handles_db_error(self, mock_neo4j, client):
        mock_neo4j.session.return_value.__enter__ = MagicMock(
            side_effect=RuntimeError("db down")
        )
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.get("/dashboard/api/graph/entity-types-distribution")
        assert resp.status_code == 500
        assert "error" in resp.json()


class TestShortestPath:
    @patch("src.dashboard.graph_viz.neo4j_client")
    def test_returns_path(self, mock_neo4j, client):
        mock_session = MagicMock()
        mock_session.run.return_value = iter([
            {
                "path_nodes": [
                    {"label": "Entity", "name": "A", "type": "Person"},
                    {"label": "Chunk", "id": "c1"},
                    {"label": "Entity", "name": "B", "type": "Location"},
                ],
                "rel_types": ["MENTIONS", "MENTIONS"],
                "path_length": 2,
            }
        ])
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.get("/dashboard/api/graph/shortest-path", params={"entity_a": "A", "entity_b": "B"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["nodes"]) == 3
        assert len(data["edges"]) == 2
        assert data["path_length"] == 2
        assert data["nodes"][0]["id"] == "A"
        assert data["nodes"][1]["id"] == "c1"

    @patch("src.dashboard.graph_viz.neo4j_client")
    def test_no_path_found(self, mock_neo4j, client):
        mock_session = MagicMock()
        mock_session.run.return_value = iter([])
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.get("/dashboard/api/graph/shortest-path", params={"entity_a": "X", "entity_b": "Y"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["nodes"] == []
        assert data["path_length"] is None
        assert "No path found" in data["message"]

    @patch("src.dashboard.graph_viz.neo4j_client")
    def test_handles_db_error(self, mock_neo4j, client):
        mock_neo4j.session.return_value.__enter__ = MagicMock(
            side_effect=RuntimeError("syntax error")
        )
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.get("/dashboard/api/graph/shortest-path", params={"entity_a": "A", "entity_b": "B"})
        assert resp.status_code == 500
        assert "error" in resp.json()

    def test_validates_max_hops_range(self, client):
        resp = client.get("/dashboard/api/graph/shortest-path", params={"entity_a": "A", "entity_b": "B", "max_hops": 10})
        assert resp.status_code == 422

    def test_validates_required_params(self, client):
        resp = client.get("/dashboard/api/graph/shortest-path")
        assert resp.status_code == 422


class TestQueryParamValidation:
    def test_entity_neighborhood_requires_name(self, client):
        resp = client.get("/dashboard/api/graph/entity-neighborhood")
        assert resp.status_code == 422

    def test_entity_neighborhood_hops_range(self, client):
        resp = client.get("/dashboard/api/graph/entity-neighborhood", params={"name": "x", "hops": 5})
        assert resp.status_code == 422

    def test_page_graph_requires_title(self, client):
        resp = client.get("/dashboard/api/graph/page-graph")
        assert resp.status_code == 422

    def test_page_graph_depth_range(self, client):
        resp = client.get("/dashboard/api/graph/page-graph", params={"title": "x", "depth": 5})
        assert resp.status_code == 422


class TestCommunityOverview:
    @patch("src.dashboard.graph_viz.neo4j_client")
    def test_returns_community_members(self, mock_neo4j, client):
        mock_session = MagicMock()
        mock_session.run.return_value = iter([
            {"name": "Hồ Chí Minh", "type": "Person", "connected_to": ["Việt Nam", "Đảng Cộng sản"]},
            {"name": "Việt Nam", "type": "Location", "connected_to": ["Hồ Chí Minh"]},
            {"name": "Đảng Cộng sản", "type": "Organization", "connected_to": ["Hồ Chí Minh"]},
        ])
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.get("/dashboard/api/graph/community-overview", params={"community_id": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["community_id"] == 5
        assert data["member_count"] == 3
        assert len(data["nodes"]) == 3
        assert len(data["edges"]) == 2  # deduplicated undirected edges

    @patch("src.dashboard.graph_viz.neo4j_client")
    def test_empty_community(self, mock_neo4j, client):
        mock_session = MagicMock()
        mock_session.run.return_value = iter([])
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.get("/dashboard/api/graph/community-overview", params={"community_id": 999})
        assert resp.status_code == 200
        data = resp.json()
        assert data["nodes"] == []
        assert "No entities found" in data["message"]

    @patch("src.dashboard.graph_viz.neo4j_client")
    def test_handles_db_error(self, mock_neo4j, client):
        mock_neo4j.session.return_value.__enter__ = MagicMock(
            side_effect=RuntimeError("connection refused")
        )
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.get("/dashboard/api/graph/community-overview", params={"community_id": 1})
        assert resp.status_code == 500
        assert "error" in resp.json()

    @patch("src.dashboard.graph_viz.neo4j_client")
    def test_skips_nodes_without_name(self, mock_neo4j, client):
        mock_session = MagicMock()
        mock_session.run.return_value = iter([
            {"name": "A", "type": "Person", "connected_to": ["B"]},
            {"name": None, "type": "Location", "connected_to": []},
            {"name": "", "type": "Org", "connected_to": []},
            {"name": "B", "type": "Person", "connected_to": ["A"]},
        ])
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.get("/dashboard/api/graph/community-overview", params={"community_id": 2})
        data = resp.json()
        assert data["member_count"] == 2
        assert len(data["edges"]) == 1

    def test_requires_community_id(self, client):
        resp = client.get("/dashboard/api/graph/community-overview")
        assert resp.status_code == 422


class TestSearchEntities:
    @patch("src.dashboard.graph_viz.neo4j_client")
    def test_returns_matching_entities(self, mock_neo4j, client):
        mock_session = MagicMock()
        mock_session.run.return_value = iter([
            {"name": "Hồ Chí Minh", "type": "Person", "community_id": 5},
            {"name": "Hồ Xuân Hương", "type": "Person", "community_id": 12},
        ])
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.get("/dashboard/api/graph/search-entities", params={"q": "Hồ"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert data["results"][0]["name"] == "Hồ Chí Minh"

    @patch("src.dashboard.graph_viz.neo4j_client")
    def test_empty_results(self, mock_neo4j, client):
        mock_session = MagicMock()
        mock_session.run.return_value = iter([])
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.get("/dashboard/api/graph/search-entities", params={"q": "xyz"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["results"] == []

    @patch("src.dashboard.graph_viz.neo4j_client")
    def test_handles_db_error(self, mock_neo4j, client):
        mock_neo4j.session.return_value.__enter__ = MagicMock(
            side_effect=RuntimeError("timeout")
        )
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.get("/dashboard/api/graph/search-entities", params={"q": "test"})
        assert resp.status_code == 500
        assert "error" in resp.json()

    def test_requires_query_param(self, client):
        resp = client.get("/dashboard/api/graph/search-entities")
        assert resp.status_code == 422


class TestSearchPages:
    @patch("src.dashboard.graph_viz.neo4j_client")
    def test_returns_matching_pages(self, mock_neo4j, client):
        mock_session = MagicMock()
        mock_session.run.return_value = iter([
            {"title": "Hà Nội", "url": "https://vi.wikipedia.org/wiki/Hà_Nội"},
            {"title": "Hà Tĩnh", "url": "https://vi.wikipedia.org/wiki/Hà_Tĩnh"},
        ])
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.get("/dashboard/api/graph/search-pages", params={"q": "Hà"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert data["results"][0]["title"] == "Hà Nội"

    @patch("src.dashboard.graph_viz.neo4j_client")
    def test_empty_results(self, mock_neo4j, client):
        mock_session = MagicMock()
        mock_session.run.return_value = iter([])
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.get("/dashboard/api/graph/search-pages", params={"q": "zzz"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0

    @patch("src.dashboard.graph_viz.neo4j_client")
    def test_handles_db_error(self, mock_neo4j, client):
        mock_neo4j.session.return_value.__enter__ = MagicMock(
            side_effect=RuntimeError("db down")
        )
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.get("/dashboard/api/graph/search-pages", params={"q": "test"})
        assert resp.status_code == 500
        assert "error" in resp.json()

    def test_requires_query_param(self, client):
        resp = client.get("/dashboard/api/graph/search-pages")
        assert resp.status_code == 422
