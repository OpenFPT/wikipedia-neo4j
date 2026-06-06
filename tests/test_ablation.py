"""Tests for ablation studies."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from scripts.run_ablation import _print_comparison_table
from src.evaluation import (
    ABLATION_MODES,
    EvalMetrics,
    _retrieve_for_ablation,
    evaluate_ablation,
)


class TestAblationModes:
    """Test ablation mode definitions."""

    def test_all_modes_defined(self):
        assert len(ABLATION_MODES) == 5

    def test_mode_names(self):
        assert "full_hybrid" in ABLATION_MODES
        assert "graph_only" in ABLATION_MODES
        assert "text_only" in ABLATION_MODES
        assert "no_reranking" in ABLATION_MODES
        assert "no_multi_hop" in ABLATION_MODES


class TestRetrieveForAblation:
    """Test _retrieve_for_ablation dispatches correctly."""

    @patch("src.evaluation._run_fallback_query")
    def test_full_hybrid_uses_fallback(self, mock_fallback):
        mock_fallback.return_value = [{"chunk_id": "c1", "chunk_text": "text"}]
        result = _retrieve_for_ablation("test question", "full_hybrid", top_k=10)
        mock_fallback.assert_called_once_with("test question", 10)
        assert result == [{"chunk_id": "c1", "chunk_text": "text"}]

    @patch("src.evaluation._run_fallback_query")
    def test_no_reranking_uses_fallback(self, mock_fallback):
        mock_fallback.return_value = [{"chunk_id": "c1", "chunk_text": "text"}]
        _retrieve_for_ablation("q", "no_reranking", top_k=5)
        mock_fallback.assert_called_once_with("q", 5)

    @patch("src.evaluation._run_fallback_query")
    def test_no_multi_hop_uses_fallback(self, mock_fallback):
        mock_fallback.return_value = []
        _retrieve_for_ablation("q", "no_multi_hop", top_k=5)
        mock_fallback.assert_called_once_with("q", 5)

    @patch("src.infrastructure.neo4j_client.neo4j_client")
    def test_text_only_uses_text_cypher(self, mock_client):
        mock_session = MagicMock()
        mock_client.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_client.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.run.return_value = []

        result = _retrieve_for_ablation("q", "text_only", top_k=10)
        assert result == []
        call_args = mock_session.run.call_args
        assert "chunk_text_ft" in call_args[0][0]
        assert "entity_alias_ft" not in call_args[0][0]

    @patch("src.infrastructure.neo4j_client.neo4j_client")
    def test_graph_only_uses_entity_cypher(self, mock_client):
        mock_session = MagicMock()
        mock_client.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_client.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.run.return_value = []

        result = _retrieve_for_ablation("q", "graph_only", top_k=10)
        assert result == []
        call_args = mock_session.run.call_args
        assert "entity_alias_ft" in call_args[0][0]


class TestEvaluateAblation:
    """Test evaluate_ablation function."""

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown ablation mode"):
            evaluate_ablation(mode="invalid_mode", limit=1)

    @patch("src.evaluation.load_test_set")
    @patch("src.evaluation._retrieve_for_ablation")
    def test_empty_dataset(self, mock_retrieve, mock_load):
        mock_load.return_value = []
        metrics = evaluate_ablation(mode="full_hybrid", limit=10)
        assert metrics.total == 0
        assert metrics.context_hit_rate == 0.0

    @patch("src.evaluation.rerank")
    @patch("src.evaluation._retrieve_for_ablation")
    @patch("src.evaluation.load_test_set")
    def test_no_reranking_skips_rerank(self, mock_load, mock_retrieve, mock_rerank):
        mock_load.return_value = [
            {
                "question": "test?",
                "metadata": {"evidence_chunk_ids": ["c1"]},
                "id": "q1",
            }
        ]
        mock_retrieve.return_value = [{"chunk_id": "c1", "chunk_text": "text", "page_id": "p1"}]

        metrics = evaluate_ablation(mode="no_reranking", limit=1)
        mock_rerank.assert_not_called()
        assert metrics.context_hit_rate == 1.0

    @patch("src.evaluation.rerank")
    @patch("src.evaluation._retrieve_for_ablation")
    @patch("src.evaluation.load_test_set")
    def test_full_hybrid_applies_rerank(self, mock_load, mock_retrieve, mock_rerank):
        mock_load.return_value = [
            {
                "question": "test?",
                "metadata": {"evidence_chunk_ids": ["c1"]},
                "id": "q1",
            }
        ]
        mock_retrieve.return_value = [{"chunk_id": "c1", "chunk_text": "text", "page_id": "p1"}]
        mock_rerank.return_value = [{"chunk_id": "c1", "chunk_text": "text", "page_id": "p1"}]

        with patch("src.retrieval.hybrid._expand_via_links", return_value=[]):
            metrics = evaluate_ablation(mode="full_hybrid", limit=1)
        mock_rerank.assert_called()
        assert metrics.context_hit_rate == 1.0

    @patch("src.evaluation._retrieve_for_ablation")
    @patch("src.evaluation.load_test_set")
    def test_samples_without_gold_ids_skipped(self, mock_load, mock_retrieve):
        mock_load.return_value = [
            {"question": "no gold", "metadata": {}, "id": "q1"},
        ]
        metrics = evaluate_ablation(mode="text_only", limit=1)
        mock_retrieve.assert_not_called()
        assert metrics.context_hit_rate == 0.0


class TestPrintComparisonTable:
    """Test the CLI comparison table formatter."""

    def test_empty_results(self):
        report = _print_comparison_table({})
        assert "Ablation Study Results" in report

    def test_single_result(self):
        results = {
            "full_hybrid": EvalMetrics(
                total=10, context_hit_rate=0.8, mrr=0.7, avg_latency_ms=120.0
            ),
        }
        report = _print_comparison_table(results)
        assert "full_hybrid" in report
        assert "0.800" in report

    def test_multiple_results_shows_deltas(self):
        results = {
            "full_hybrid": EvalMetrics(
                total=10, context_hit_rate=0.8, mrr=0.7, avg_latency_ms=120.0
            ),
            "text_only": EvalMetrics(
                total=10, context_hit_rate=0.5, mrr=0.4, avg_latency_ms=60.0
            ),
        }
        report = _print_comparison_table(results)
        assert "Component Contributions" in report
        assert "text_only" in report
        # HR delta: (0.5 - 0.8) * 100 = -30.0
        assert "-30.0%" in report
