"""Tests for eval_mcp_claude metric functions."""

from scripts.eval_mcp_claude import (
    compute_em,
    compute_efficiency,
    compute_f1,
    compute_faithfulness,
    compute_best_metrics,
    detect_refusal,
    load_dataset,
    normalize_answer,
)
from pathlib import Path


class TestNormalizeAnswer:
    def test_basic(self):
        assert normalize_answer("  Hà Nội  ") == "hà nội"

    def test_punctuation(self):
        assert normalize_answer("Hồ Chí Minh.") == "hồ chí minh"

    def test_multiple_spaces(self):
        assert normalize_answer("a   b   c") == "a b c"


class TestComputeF1:
    def test_exact_match(self):
        assert compute_f1("Lâm Bá Kiệt", "Lâm Bá Kiệt") == 1.0

    def test_partial_overlap(self):
        f1 = compute_f1("Thủ tướng Phạm Văn Đồng", "Phạm Văn Đồng")
        assert 0.5 < f1 < 1.0

    def test_no_overlap(self):
        assert compute_f1("Hà Nội", "Sài Gòn") == 0.0

    def test_empty_prediction(self):
        assert compute_f1("", "something") == 0.0

    def test_empty_gold(self):
        assert compute_f1("something", "") == 0.0


class TestComputeEM:
    def test_exact(self):
        assert compute_em("Hà Nội", "Hà Nội") == 1.0

    def test_case_insensitive(self):
        assert compute_em("hà nội", "Hà Nội") == 1.0

    def test_not_match(self):
        assert compute_em("Hà Nội", "Sài Gòn") == 0.0


class TestComputeBestMetrics:
    def test_multiple_gold(self):
        result = compute_best_metrics("Hà Nội", ["Sài Gòn", "Hà Nội"])
        assert result["f1"] == 1.0
        assert result["em"] == 1.0

    def test_no_gold(self):
        result = compute_best_metrics("answer", [])
        assert result["f1"] == 0.0


class TestComputeFaithfulness:
    def test_fully_grounded(self):
        answer = "Hà Nội là thủ đô"
        context = ["Hà Nội là thủ đô của Việt Nam"]
        assert compute_faithfulness(answer, context) == 1.0

    def test_partially_grounded(self):
        answer = "Hà Nội là thủ đô của Pháp"
        context = ["Hà Nội là thủ đô của Việt Nam"]
        faith = compute_faithfulness(answer, context)
        assert 0.5 < faith < 1.0

    def test_not_grounded(self):
        answer = "Tokyo là thành phố lớn nhất"
        context = ["Hà Nội là thủ đô của Việt Nam"]
        faith = compute_faithfulness(answer, context)
        assert faith < 0.5

    def test_empty_answer(self):
        assert compute_faithfulness("", ["some context"]) == 0.0

    def test_empty_context(self):
        assert compute_faithfulness("some answer", []) == 0.0


class TestComputeEfficiency:
    def test_one_call(self):
        assert compute_efficiency(1) == 1.0

    def test_two_calls(self):
        assert compute_efficiency(2) == 0.5

    def test_zero_calls(self):
        assert compute_efficiency(0) == 0.0


class TestDetectRefusal:
    def test_refusal_phrases(self):
        assert detect_refusal("Không tìm thấy thông tin về chủ đề này") is True
        assert detect_refusal("Không có thông tin trong cơ sở dữ liệu") is True

    def test_normal_answer(self):
        assert detect_refusal("Hà Nội là thủ đô của Việt Nam") is False

    def test_empty(self):
        assert detect_refusal("") is False


class TestLoadDataset:
    def test_load_unsolvable(self):
        path = Path("data/viquad2/unsolvable.jsonl")
        if not path.exists():
            return
        samples = load_dataset(path, limit=5)
        assert len(samples) == 5
        assert samples[0]["is_unsolvable"] is True

    def test_load_validation(self):
        path = Path("data/viquad2/validation.jsonl")
        if not path.exists():
            return
        samples = load_dataset(path, limit=3)
        assert len(samples) == 3
        assert "question" in samples[0]
        assert "gold_answers" in samples[0]
        assert samples[0]["is_unsolvable"] is False
