"""Tests for RAGAS evaluation metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation import (
    RAGASMetrics,
    compute_ragas_metrics,
    print_ragas_report,
    save_ragas_results,
)


def test_ragas_metrics_dataclass():
    """Test RAGASMetrics dataclass initialization."""
    metrics = RAGASMetrics(total=10)
    assert metrics.total == 10
    assert metrics.context_precision == 0.0
    assert metrics.context_recall == 0.0
    assert metrics.faithfulness == 0.0
    assert metrics.answer_relevancy == 0.0
    assert metrics.details == []


def test_ragas_metrics_with_values():
    """Test RAGASMetrics with populated values."""
    metrics = RAGASMetrics(
        total=5,
        context_precision=0.85,
        context_recall=0.90,
        faithfulness=0.88,
        answer_relevancy=0.92,
        details=[{"question": "test", "context_precision": 0.85}],
    )
    assert metrics.total == 5
    assert metrics.context_precision == 0.85
    assert metrics.context_recall == 0.90
    assert metrics.faithfulness == 0.88
    assert metrics.answer_relevancy == 0.92
    assert len(metrics.details) == 1


def test_print_ragas_report():
    """Test RAGAS report formatting."""
    metrics = RAGASMetrics(
        total=10,
        context_precision=0.85,
        context_recall=0.90,
        faithfulness=0.88,
        answer_relevancy=0.92,
    )
    report = print_ragas_report(metrics)
    assert "RAGAS Evaluation Report" in report
    assert "Context Precision: 0.850" in report
    assert "Context Recall:    0.900" in report
    assert "Faithfulness:      0.880" in report
    assert "Answer Relevancy:  0.920" in report


def test_print_ragas_report_zero_values():
    """Test RAGAS report with zero values."""
    metrics = RAGASMetrics(total=0)
    report = print_ragas_report(metrics)
    assert "RAGAS Evaluation Report" in report
    assert "Total samples: 0" in report


def test_compute_ragas_metrics_length_mismatch():
    """Test that compute_ragas_metrics raises on length mismatch."""
    questions = ["What is X?", "What is Y?"]
    contexts = [["context1"], ["context2"]]
    answers = ["answer1"]  # Mismatch: only 1 answer for 2 questions

    with pytest.raises(ValueError, match="must have same length"):
        compute_ragas_metrics(questions, contexts, answers)


def test_compute_ragas_metrics_ground_truth_mismatch():
    """Test that compute_ragas_metrics raises on ground_truth length mismatch."""
    questions = ["What is X?", "What is Y?"]
    contexts = [["context1"], ["context2"]]
    answers = ["answer1", "answer2"]
    ground_truths = ["truth1"]  # Mismatch: only 1 ground truth for 2 questions

    with pytest.raises(ValueError, match="ground_truths must match length"):
        compute_ragas_metrics(questions, contexts, answers, ground_truths)


def test_compute_ragas_metrics_empty():
    """Test compute_ragas_metrics with empty inputs."""
    metrics = compute_ragas_metrics([], [], [])
    assert metrics.total == 0
    assert metrics.details == []


def test_compute_ragas_metrics_basic():
    """Test basic RAGAS metrics computation (if RAGAS is available)."""
    try:
        from ragas import evaluate  # noqa: F401

        questions = ["What is Python?"]
        contexts = [["Python is a programming language"]]
        answers = ["Python is a programming language"]

        metrics = compute_ragas_metrics(questions, contexts, answers)
        assert metrics.total == 1
        # Metrics should be computed (values between 0 and 1)
        assert 0 <= metrics.context_precision <= 1
        assert 0 <= metrics.context_recall <= 1
        assert 0 <= metrics.faithfulness <= 1
        assert 0 <= metrics.answer_relevancy <= 1

    except ImportError:
        pytest.skip("RAGAS not installed")


def test_save_ragas_results(tmp_path):
    """Test saving RAGAS results to JSON."""
    metrics = RAGASMetrics(
        total=2,
        context_precision=0.85,
        context_recall=0.90,
        faithfulness=0.88,
        answer_relevancy=0.92,
        details=[
            {"question": "Q1", "context_precision": 0.85},
            {"question": "Q2", "context_precision": 0.85},
        ],
    )

    output_path = str(tmp_path / "ragas_results.json")
    save_ragas_results(metrics, output_path)

    assert Path(output_path).exists()

    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["total"] == 2
    assert data["context_precision"] == 0.85
    assert data["context_recall"] == 0.90
    assert data["faithfulness"] == 0.88
    assert data["answer_relevancy"] == 0.92
    assert len(data["details"]) == 2


def test_compute_ragas_metrics_with_ground_truth():
    """Test compute_ragas_metrics with ground truth answers."""
    questions = ["What is X?"]
    contexts = [["X is a variable"]]
    answers = ["X is a variable"]
    ground_truths = ["X is a variable"]

    metrics = compute_ragas_metrics(questions, contexts, answers, ground_truths)
    assert metrics.total == 1

