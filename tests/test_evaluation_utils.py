"""Tests for the evaluation module utility functions."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.evaluation import (
    ABLATION_MODES,
    EvalMetrics,
    RAGASMetrics,
    ViQuADMetrics,
    _compute_hit_rate,
    _compute_mrr,
    _normalize_answer,
    compute_em,
    compute_token_f1,
    load_test_set,
    print_ragas_report,
    print_report,
    save_ragas_results,
    save_results,
)


class TestComputeHitRate:
    def test_hit_found(self):
        assert _compute_hit_rate(["a", "b", "c"], ["b"]) == 1.0

    def test_hit_not_found(self):
        assert _compute_hit_rate(["a", "b", "c"], ["d"]) == 0.0

    def test_empty_gold(self):
        assert _compute_hit_rate(["a", "b"], []) == 0.0

    def test_empty_retrieved(self):
        assert _compute_hit_rate([], ["a"]) == 0.0

    def test_multiple_gold_one_match(self):
        assert _compute_hit_rate(["a", "b"], ["x", "b"]) == 1.0


class TestComputeMRR:
    def test_first_position(self):
        assert _compute_mrr(["a", "b", "c"], ["a"]) == 1.0

    def test_second_position(self):
        assert _compute_mrr(["a", "b", "c"], ["b"]) == 0.5

    def test_third_position(self):
        assert _compute_mrr(["a", "b", "c"], ["c"]) == pytest.approx(1 / 3)

    def test_not_found(self):
        assert _compute_mrr(["a", "b", "c"], ["d"]) == 0.0

    def test_empty_gold(self):
        assert _compute_mrr(["a", "b"], []) == 0.0

    def test_multiple_gold_first_match(self):
        assert _compute_mrr(["a", "b", "c"], ["c", "a"]) == 1.0


class TestNormalizeAnswer:
    def test_lowercase(self):
        assert _normalize_answer("Hello World") == "hello world"

    def test_strip_punctuation(self):
        assert _normalize_answer("hello, world!") == "hello world"

    def test_collapse_whitespace(self):
        assert _normalize_answer("hello   world") == "hello world"

    def test_combined(self):
        assert _normalize_answer("  Hồ Chí Minh, Việt Nam!  ") == "hồ chí minh việt nam"


class TestComputeEM:
    def test_exact_match(self):
        assert compute_em("Hà Nội", ["Hà Nội"]) == 1.0

    def test_case_insensitive(self):
        assert compute_em("hà nội", ["Hà Nội"]) == 1.0

    def test_no_match(self):
        assert compute_em("Hà Nội", ["Sài Gòn"]) == 0.0

    def test_empty_gold(self):
        assert compute_em("anything", []) == 0.0

    def test_multiple_gold(self):
        assert compute_em("answer b", ["answer a", "answer b"]) == 1.0


class TestComputeTokenF1:
    def test_perfect_match(self):
        assert compute_token_f1("hello world", ["hello world"]) == 1.0

    def test_partial_match(self):
        f1 = compute_token_f1("hello world foo", ["hello world bar"])
        assert 0.0 < f1 < 1.0

    def test_no_match(self):
        assert compute_token_f1("abc", ["xyz"]) == 0.0

    def test_empty_gold(self):
        assert compute_token_f1("hello", []) == 0.0

    def test_empty_prediction(self):
        assert compute_token_f1("", ["hello"]) == 0.0

    def test_best_of_multiple_gold(self):
        f1 = compute_token_f1("hello world", ["xyz", "hello world"])
        assert f1 == 1.0


class TestLoadTestSet:
    def test_load_valid(self, tmp_path):
        data = [
            {"question": "q1", "metadata": {"evidence_chunk_ids": ["c1"]}},
            {"question": "q2", "metadata": {"evidence_chunk_ids": ["c2"]}},
        ]
        p = tmp_path / "test.jsonl"
        p.write_text("\n".join(json.dumps(d) for d in data))

        result = load_test_set(p)
        assert len(result) == 2
        assert result[0]["question"] == "q1"

    def test_load_with_limit(self, tmp_path):
        data = [{"question": f"q{i}"} for i in range(10)]
        p = tmp_path / "test.jsonl"
        p.write_text("\n".join(json.dumps(d) for d in data))

        result = load_test_set(p, limit=3)
        assert len(result) == 3

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_test_set(Path("/nonexistent/path.jsonl"))

    def test_load_skips_blank_lines(self, tmp_path):
        p = tmp_path / "test.jsonl"
        p.write_text('{"question": "q1"}\n\n{"question": "q2"}\n\n')

        result = load_test_set(p)
        assert len(result) == 2


class TestPrintReport:
    def test_format(self):
        metrics = EvalMetrics(
            total=10,
            context_hit_rate=0.8,
            mrr=0.65,
            avg_latency_ms=120.5,
            rerank_context_hit_rate=0.9,
            rerank_mrr=0.75,
        )
        report = print_report(metrics)
        assert "Total samples: 10" in report
        assert "0.800" in report
        assert "0.650" in report
        assert "0.900" in report


class TestSaveResults:
    def test_save_creates_file(self, tmp_path):
        metrics = EvalMetrics(
            total=5,
            context_hit_rate=0.7,
            mrr=0.5,
            avg_latency_ms=100.0,
            rerank_context_hit_rate=0.8,
            rerank_mrr=0.6,
            details=[{"id": 1, "question": "q1"}],
        )
        out = tmp_path / "reports" / "eval.json"
        save_results(metrics, str(out))

        assert out.exists()
        data = json.loads(out.read_text())
        assert data["total"] == 5
        assert data["context_hit_rate"] == 0.7
        assert len(data["details"]) == 1


class TestRAGASMetrics:
    def test_print_ragas_report(self):
        metrics = RAGASMetrics(
            total=10,
            context_precision=0.85,
            context_recall=0.78,
            faithfulness=0.92,
            answer_relevancy=0.88,
        )
        report = print_ragas_report(metrics)
        assert "Total samples: 10" in report
        assert "0.850" in report
        assert "0.780" in report

    def test_save_ragas_results(self, tmp_path):
        metrics = RAGASMetrics(
            total=5,
            context_precision=0.9,
            context_recall=0.8,
            faithfulness=0.85,
            answer_relevancy=0.75,
            details=[{"question": "q1"}],
        )
        out = tmp_path / "reports" / "ragas.json"
        save_ragas_results(metrics, str(out))

        assert out.exists()
        data = json.loads(out.read_text())
        assert data["total"] == 5
        assert data["faithfulness"] == 0.85

    def test_compute_ragas_input_validation(self):
        from src.evaluation import compute_ragas_metrics

        with pytest.raises(ValueError, match="same length"):
            compute_ragas_metrics(
                questions=["q1", "q2"],
                contexts=[["c1"]],
                answers=["a1", "a2"],
            )

    def test_compute_ragas_ground_truth_validation(self):
        from src.evaluation import compute_ragas_metrics

        with pytest.raises(ValueError, match="ground_truths must match"):
            compute_ragas_metrics(
                questions=["q1"],
                contexts=[["c1"]],
                answers=["a1"],
                ground_truths=["g1", "g2"],
            )


class TestAblation:
    def test_invalid_mode(self):
        from src.evaluation import evaluate_ablation

        with pytest.raises(ValueError, match="Unknown ablation mode"):
            evaluate_ablation("invalid_mode", limit=1)

    def test_ablation_modes_constant(self):
        assert "full_hybrid" in ABLATION_MODES
        assert "graph_only" in ABLATION_MODES
        assert "text_only" in ABLATION_MODES
        assert "no_reranking" in ABLATION_MODES
        assert "no_multi_hop" in ABLATION_MODES


class TestViQuADMetrics:
    def test_dataclass_defaults(self):
        m = ViQuADMetrics()
        assert m.total == 0
        assert m.answerable_count == 0
        assert m.impossible_count == 0
        assert m.exact_match == 0.0


class TestEvaluateFunction:
    def test_evaluate_with_mock(self, tmp_path):
        from src.evaluation import evaluate

        data = [
            {
                "question": "Ai là tổng thống?",
                "metadata": {"evidence_chunk_ids": ["chunk_1"]},
            },
            {
                "question": "Thủ đô ở đâu?",
                "metadata": {"evidence_chunk_ids": ["chunk_2"]},
            },
        ]
        p = tmp_path / "test.jsonl"
        p.write_text("\n".join(json.dumps(d) for d in data))

        mock_rows = [
            {"chunk_id": "chunk_1", "chunk_text": "text1", "score": 0.9},
            {"chunk_id": "chunk_3", "chunk_text": "text3", "score": 0.5},
        ]

        with patch("src.evaluation._retrieve_chunks", return_value=mock_rows):
            metrics = evaluate(limit=2, dataset_path=p)

        assert metrics.total == 2
        assert metrics.context_hit_rate > 0
        assert len(metrics.details) == 2


class TestEvaluateViquad:
    def test_evaluate_viquad_with_mock(self):
        from src.evaluation import evaluate_viquad

        mock_samples = [
            {
                "id": "q1",
                "question": "Thủ đô Việt Nam là gì?",
                "gold_answers": ["Hà Nội"],
                "is_impossible": False,
                "context": "Hà Nội là thủ đô của Việt Nam nằm ở miền Bắc",
            },
            {
                "id": "q2",
                "question": "Ai phát minh ra máy bay?",
                "gold_answers": [],
                "is_impossible": True,
                "context": "Không có thông tin",
            },
        ]

        mock_rows = [
            {"chunk_id": "c1", "chunk_text": "Hà Nội là thủ đô của Việt Nam nằm ở miền Bắc", "score": 0.9},
            {"chunk_id": "c2", "chunk_text": "other text", "score": 0.5},
        ]

        with patch("src.viquad_adapter.load_viquad", return_value=mock_samples), \
             patch("src.evaluation._retrieve_chunks", return_value=mock_rows):
            metrics = evaluate_viquad(limit=2)

        assert metrics.total == 2
        assert metrics.answerable_count == 1
        assert metrics.impossible_count == 1
        assert metrics.context_hit_rate > 0
        assert len(metrics.details) == 2

    def test_evaluate_viquad_empty_retrieval(self):
        from src.evaluation import evaluate_viquad

        mock_samples = [
            {
                "id": "q1",
                "question": "Test?",
                "gold_answers": ["answer"],
                "is_impossible": False,
                "context": "some context here with enough tokens",
            },
        ]

        with patch("src.viquad_adapter.load_viquad", return_value=mock_samples), \
             patch("src.evaluation._retrieve_chunks", return_value=[]):
            metrics = evaluate_viquad(limit=1)

        assert metrics.total == 1
        assert metrics.context_hit_rate == 0.0


class TestPrintViquadReport:
    def test_format(self):
        from src.evaluation import ViQuADMetrics, print_viquad_report

        metrics = ViQuADMetrics(
            total=20,
            answerable_count=15,
            impossible_count=5,
            context_hit_rate=0.73,
            mrr=0.65,
            exact_match=0.40,
            token_f1=0.55,
            abstain_accuracy=0.80,
            avg_latency_ms=200.0,
        )
        report = print_viquad_report(metrics)
        assert "Total samples: 20" in report
        assert "answerable: 15" in report
        assert "impossible: 5" in report
        assert "0.730" in report
        assert "0.400" in report


class TestRetrieveChunks:
    def test_fallback_on_generated_failure(self):
        from src.evaluation import _retrieve_chunks

        mock_fallback_rows = [{"chunk_id": "c1", "chunk_text": "text", "score": 0.5}]

        with patch("src.evaluation._run_generated_query", side_effect=Exception("fail")), \
             patch("src.evaluation._run_fallback_query", return_value=mock_fallback_rows):
            result = _retrieve_chunks("test question")

        assert result == mock_fallback_rows

    def test_uses_generated_query_first(self):
        from src.evaluation import _retrieve_chunks

        mock_rows = [{"chunk_id": "c1", "chunk_text": "text", "score": 0.9}]

        with patch("src.evaluation._run_generated_query", return_value=mock_rows) as mock_gen:
            result = _retrieve_chunks("test question", top_k=10)

        assert result == mock_rows
        mock_gen.assert_called_once_with("test question", 10)
