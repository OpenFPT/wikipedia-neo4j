"""Tests for src/relation_extract module."""

from __future__ import annotations

from src.relation_extract import RELATION_TYPES, Triple, _parse_triples, extract_relations_batch


class TestParseTriples:
    """Tests for _parse_triples."""

    def test_valid_json_array(self):
        raw = '[{"subject": "Hà Nội", "relation": "LOCATED_IN", "object": "Việt Nam"}]'
        result = _parse_triples(raw)
        assert len(result) == 1
        assert result[0].subject == "Hà Nội"
        assert result[0].relation == "LOCATED_IN"
        assert result[0].object == "Việt Nam"
        assert result[0].confidence == 1.0

    def test_empty_array(self):
        result = _parse_triples("[]")
        assert result == []

    def test_code_fenced_output(self):
        raw = '```json\n[{"subject": "A", "relation": "PART_OF", "object": "B"}]\n```'
        result = _parse_triples(raw)
        assert len(result) == 1
        assert result[0].relation == "PART_OF"

    def test_invalid_json(self):
        result = _parse_triples("not json at all")
        assert result == []

    def test_invalid_relation_type_filtered(self):
        raw = '[{"subject": "A", "relation": "UNKNOWN_REL", "object": "B"}]'
        result = _parse_triples(raw)
        assert result == []

    def test_missing_subject_filtered(self):
        raw = '[{"subject": "", "relation": "BORN_IN", "object": "B"}]'
        result = _parse_triples(raw)
        assert result == []

    def test_missing_object_filtered(self):
        raw = '[{"subject": "A", "relation": "BORN_IN", "object": ""}]'
        result = _parse_triples(raw)
        assert result == []

    def test_multiple_triples(self):
        raw = """[
            {"subject": "Nguyễn Du", "relation": "BORN_IN", "object": "Hà Tĩnh"},
            {"subject": "Truyện Kiều", "relation": "CREATED_BY", "object": "Nguyễn Du"},
            {"subject": "FPT", "relation": "LOCATED_IN", "object": "Hà Nội"}
        ]"""
        result = _parse_triples(raw)
        assert len(result) == 3

    def test_non_dict_items_filtered(self):
        raw = '[{"subject": "A", "relation": "PART_OF", "object": "B"}, "invalid", 123]'
        result = _parse_triples(raw)
        assert len(result) == 1

    def test_confidence_preserved(self):
        raw = '[{"subject": "A", "relation": "MEMBER_OF", "object": "B", "confidence": 0.8}]'
        result = _parse_triples(raw)
        assert result[0].confidence == 0.8

    def test_relation_case_insensitive(self):
        raw = '[{"subject": "A", "relation": "founded_by", "object": "B"}]'
        result = _parse_triples(raw)
        assert len(result) == 1
        assert result[0].relation == "FOUNDED_BY"

    def test_json_embedded_in_text(self):
        raw = 'Here are the results:\n[{"subject": "X", "relation": "PART_OF", "object": "Y"}]\nDone.'
        result = _parse_triples(raw)
        assert len(result) == 1


class TestRelationTypes:
    """Tests for RELATION_TYPES constant."""

    def test_has_six_types(self):
        assert len(RELATION_TYPES) == 6

    def test_expected_types(self):
        expected = {"FOUNDED_BY", "LOCATED_IN", "BORN_IN", "MEMBER_OF", "PART_OF", "CREATED_BY"}
        assert set(RELATION_TYPES) == expected


class TestTripleDataclass:
    """Tests for Triple dataclass."""

    def test_creation(self):
        t = Triple(subject="A", relation="PART_OF", object="B")
        assert t.subject == "A"
        assert t.relation == "PART_OF"
        assert t.object == "B"
        assert t.confidence == 1.0

    def test_custom_confidence(self):
        t = Triple(subject="A", relation="PART_OF", object="B", confidence=0.5)
        assert t.confidence == 0.5


class TestExtractRelationsBatch:
    """Tests for extract_relations_batch with mocked LLM."""

    def test_batch_with_mock(self, monkeypatch):
        """Test batch extraction with mocked extract_relations."""
        import src.relation_extract as mod

        call_count = 0

        def mock_extract(text, use_local=True):
            nonlocal call_count
            call_count += 1
            return [Triple(subject="A", relation="PART_OF", object="B")]

        monkeypatch.setattr(mod, "extract_relations", mock_extract)

        results = extract_relations_batch(["text1", "text2", "text3"])
        assert len(results) == 3
        assert call_count == 3
        assert all(len(r) == 1 for r in results)

    def test_batch_handles_failures(self, monkeypatch):
        """Test that batch extraction handles individual failures gracefully."""
        import src.relation_extract as mod

        def mock_extract(text, use_local=True):
            if "fail" in text:
                raise RuntimeError("LLM error")
            return [Triple(subject="A", relation="PART_OF", object="B")]

        monkeypatch.setattr(mod, "extract_relations", mock_extract)

        results = extract_relations_batch(["good", "fail", "good"])
        assert len(results) == 3
        assert len(results[0]) == 1
        assert results[1] == []
        assert len(results[2]) == 1
