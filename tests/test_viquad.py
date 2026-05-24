"""Tests for ViQuAD2.0 adapter and EM/F1 metrics."""

from __future__ import annotations

from unittest.mock import patch

from src.evaluation import _normalize_answer, compute_em, compute_token_f1


def test_normalize_answer_strips_punct() -> None:
    assert _normalize_answer("  Hà Nội!  ") == "hà nội"


def test_normalize_answer_collapses_whitespace() -> None:
    assert _normalize_answer("thủ   đô   Hà Nội") == "thủ đô hà nội"


def test_normalize_answer_empty() -> None:
    assert _normalize_answer("") == ""


def test_compute_em_exact() -> None:
    assert compute_em("Hà Nội", ["Hà Nội"]) == 1.0


def test_compute_em_normalized() -> None:
    assert compute_em("hà nội.", ["Hà Nội"]) == 1.0


def test_compute_em_no_match() -> None:
    assert compute_em("Sài Gòn", ["Hà Nội"]) == 0.0


def test_compute_em_multiple_gold() -> None:
    assert compute_em("Sài Gòn", ["Hà Nội", "Sài Gòn"]) == 1.0


def test_compute_em_empty_gold() -> None:
    assert compute_em("anything", []) == 0.0


def test_compute_f1_exact() -> None:
    assert compute_token_f1("Hà Nội", ["Hà Nội"]) == 1.0


def test_compute_f1_partial() -> None:
    f1 = compute_token_f1("thủ đô Hà Nội", ["Hà Nội"])
    assert 0.6 < f1 < 0.9


def test_compute_f1_no_overlap() -> None:
    assert compute_token_f1("abc def", ["xyz uvw"]) == 0.0


def test_compute_f1_empty_prediction() -> None:
    assert compute_token_f1("", ["Hà Nội"]) == 0.0


def test_compute_f1_empty_gold() -> None:
    assert compute_token_f1("Hà Nội", []) == 0.0


def test_load_viquad_schema() -> None:
    mock_ds = [
        {
            "id": "test001",
            "uit_id": "u001",
            "title": "Hà Nội",
            "context": "Hà Nội là thủ đô của Việt Nam.",
            "question": "Thủ đô của Việt Nam là gì?",
            "answers": {"text": ["Hà Nội"], "answer_start": [0]},
            "is_impossible": False,
            "plausible_answers": {"text": [], "answer_start": []},
        }
    ]

    with patch("src.viquad_adapter.load_dataset", return_value=mock_ds), \
         patch("pathlib.Path.exists", return_value=False):
        from src.viquad_adapter import load_viquad

        samples = load_viquad("validation", limit=1)

    assert len(samples) == 1
    s = samples[0]
    assert s["id"] == "test001"
    assert s["question"] == "Thủ đô của Việt Nam là gì?"
    assert s["gold_answers"] == ["Hà Nội"]
    assert s["context"] == "Hà Nội là thủ đô của Việt Nam."
    assert s["title"] == "Hà Nội"
    assert s["is_impossible"] is False


def test_load_viquad_impossible() -> None:
    mock_ds = [
        {
            "id": "test002",
            "uit_id": "u002",
            "title": "Test",
            "context": "Some context.",
            "question": "Impossible question?",
            "answers": {"text": [], "answer_start": []},
            "is_impossible": True,
            "plausible_answers": {"text": ["plausible"], "answer_start": [0]},
        }
    ]

    with patch("src.viquad_adapter.load_dataset", return_value=mock_ds), \
         patch("pathlib.Path.exists", return_value=False):
        from src.viquad_adapter import load_viquad

        samples = load_viquad("validation", limit=1)

    assert samples[0]["gold_answers"] == []
    assert samples[0]["is_impossible"] is True
