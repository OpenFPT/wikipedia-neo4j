"""Tests for ViQuAD adapter module."""

from __future__ import annotations

import json

import src.viquad_adapter as viquad


class TestConvertSample:
    def test_basic_conversion(self) -> None:
        sample = {
            "id": "q1",
            "question": "What is Neo4j?",
            "context": "Neo4j is a graph database.",
            "title": "Neo4j",
            "answers": {"text": ["a graph database"]},
            "is_impossible": False,
        }
        result = viquad._convert_sample(sample)
        assert result["id"] == "q1"
        assert result["question"] == "What is Neo4j?"
        assert result["gold_answers"] == ["a graph database"]
        assert result["is_impossible"] is False

    def test_missing_answers(self) -> None:
        sample = {
            "id": "q2",
            "question": "Q?",
            "context": "C",
            "title": "T",
        }
        result = viquad._convert_sample(sample)
        assert result["gold_answers"] == []
        assert result["is_impossible"] is False

    def test_non_dict_answers(self) -> None:
        sample = {
            "id": "q3",
            "question": "Q?",
            "context": "C",
            "title": "T",
            "answers": "not a dict",
        }
        result = viquad._convert_sample(sample)
        assert result["gold_answers"] == []


class TestLoadViquad:
    def test_loads_from_cache(self, tmp_path, monkeypatch) -> None:
        cache_dir = tmp_path / "viquad2"
        cache_dir.mkdir()
        cache_file = cache_dir / "validation.jsonl"

        records = [
            {"id": "1", "question": "Q1", "gold_answers": [], "context": "C", "title": "T", "is_impossible": False},
            {"id": "2", "question": "Q2", "gold_answers": [], "context": "C", "title": "T", "is_impossible": False},
        ]
        cache_file.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records))

        monkeypatch.setattr(viquad, "VIQUAD_CACHE_DIR", cache_dir)

        result = viquad.load_viquad("validation")
        assert len(result) == 2
        assert result[0]["id"] == "1"

    def test_loads_from_cache_with_limit(self, tmp_path, monkeypatch) -> None:
        cache_dir = tmp_path / "viquad2"
        cache_dir.mkdir()
        cache_file = cache_dir / "validation.jsonl"

        records = [
            {"id": str(i), "question": f"Q{i}", "gold_answers": [], "context": "C", "title": "T", "is_impossible": False}
            for i in range(10)
        ]
        cache_file.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records))

        monkeypatch.setattr(viquad, "VIQUAD_CACHE_DIR", cache_dir)

        result = viquad.load_viquad("validation", limit=3)
        assert len(result) == 3

    def test_skips_blank_lines_in_cache(self, tmp_path, monkeypatch) -> None:
        cache_dir = tmp_path / "viquad2"
        cache_dir.mkdir()
        cache_file = cache_dir / "validation.jsonl"

        content = (
            json.dumps({"id": "1", "question": "Q", "gold_answers": [], "context": "C", "title": "T", "is_impossible": False})
            + "\n\n"
            + json.dumps({"id": "2", "question": "Q", "gold_answers": [], "context": "C", "title": "T", "is_impossible": False})
        )
        cache_file.write_text(content)

        monkeypatch.setattr(viquad, "VIQUAD_CACHE_DIR", cache_dir)

        result = viquad.load_viquad("validation")
        assert len(result) == 2

    def test_loads_from_huggingface_when_no_cache(self, tmp_path, monkeypatch) -> None:
        cache_dir = tmp_path / "viquad2"
        monkeypatch.setattr(viquad, "VIQUAD_CACHE_DIR", cache_dir)

        fake_rows = [
            {"id": "hf1", "question": "Q?", "context": "C", "title": "T", "answers": {"text": ["A"]}, "is_impossible": False},
            {"id": "hf2", "question": "Q2?", "context": "C2", "title": "T2", "answers": {"text": []}, "is_impossible": True},
        ]
        monkeypatch.setattr(viquad, "load_dataset", lambda *a, **kw: fake_rows)

        result = viquad.load_viquad("validation")
        assert len(result) == 2
        assert result[0]["id"] == "hf1"
        assert result[1]["is_impossible"] is True


class TestExportEvalJsonl:
    def test_export_creates_file(self, tmp_path, monkeypatch) -> None:
        cache_dir = tmp_path / "viquad2"
        monkeypatch.setattr(viquad, "VIQUAD_CACHE_DIR", cache_dir)

        fake_rows = [
            {"id": "e1", "question": "Q?", "context": "C", "title": "T", "answers": {"text": ["A"]}, "is_impossible": False},
            {"id": "e2", "question": "Q2?", "context": "C2", "title": "T2", "answers": {"text": []}, "is_impossible": True},
        ]
        monkeypatch.setattr(viquad, "load_dataset", lambda *a, **kw: fake_rows)

        out = tmp_path / "output" / "test.jsonl"
        result_path = viquad.export_eval_jsonl(split="validation", output=out)

        assert result_path == out
        assert out.exists()
        lines = [line for line in out.read_text().strip().split("\n") if line.strip()]
        assert len(lines) == 2
        parsed = json.loads(lines[0])
        assert parsed["id"] == "e1"
        assert parsed["gold_answers"] == ["A"]

    def test_export_default_path(self, tmp_path, monkeypatch) -> None:
        cache_dir = tmp_path / "viquad2"
        monkeypatch.setattr(viquad, "VIQUAD_CACHE_DIR", cache_dir)

        fake_rows = [
            {"id": "e1", "question": "Q?", "context": "C", "title": "T", "answers": {"text": ["A"]}, "is_impossible": False},
        ]
        monkeypatch.setattr(viquad, "load_dataset", lambda *a, **kw: fake_rows)

        result_path = viquad.export_eval_jsonl(split="validation")
        assert result_path == cache_dir / "validation.jsonl"
        assert result_path.exists()
