"""Regression tests for NER data quality fixes."""

from src.ner import (
    classify_entity_type,
    classify_disambiguation_hint,
    normalize_entity,
    strip_disambiguation,
    _extract_entities_wikilink,
)


class TestTruncatedNames:
    def test_normalize_preserves_balanced_parens(self):
        assert normalize_entity("Eureka (word)") == "Eureka (word)"

    def test_strip_disambiguation_balanced(self):
        clean, hint = strip_disambiguation("Eureka (word)")
        assert clean == "Eureka"
        assert hint == "word"

    def test_strip_disambiguation_truncated(self):
        clean, hint = strip_disambiguation("Eureka (word")
        assert clean == "Eureka"
        assert hint == "word"

    def test_strip_disambiguation_no_parens(self):
        clean, hint = strip_disambiguation("California")
        assert clean == "California"
        assert hint == ""

    def test_strip_disambiguation_comma_location(self):
        clean, hint = strip_disambiguation("Sacramento, California")
        assert clean == "Sacramento, California"
        assert hint == ""

    def test_wikilink_backend_strips_disambig(self):
        raw = "[[Eureka (word)|Eureka]] is the motto of [[Sierra Nevada (Hoa Kỳ)|Sierra Nevada]]."
        entities = _extract_entities_wikilink(raw)
        names = [name for name, _ in entities]
        assert "Eureka" in names
        assert "Sierra Nevada" in names
        assert not any("(" in n for n, _ in entities)


class TestLocationClassification:
    def test_comma_state_pattern(self):
        assert classify_entity_type("Sacramento, California") == "Location"

    def test_long_beach_california(self):
        assert classify_entity_type("Long Beach, California") == "Location"

    def test_real_person_not_affected(self):
        assert classify_entity_type("Nguyễn Văn A") == "Person"

    def test_western_person_no_comma(self):
        assert classify_entity_type("John Smith") == "Person"

    def test_i_love_you_california(self):
        result = classify_entity_type("I Love You, California")
        assert result == "Location"

    def test_geo_prefix_los(self):
        assert classify_entity_type("Los Angeles") == "Location"

    def test_geo_prefix_san(self):
        assert classify_entity_type("San Francisco") == "Location"

    def test_geo_prefix_new(self):
        assert classify_entity_type("New Zealand") == "Location"

    def test_geo_prefix_saint(self):
        assert classify_entity_type("Saint Barthélemy") == "Location"


class TestDisambiguationHints:
    def test_location_hint_vietnamese(self):
        assert classify_disambiguation_hint("thành phố") == "Location"
        assert classify_disambiguation_hint("tỉnh") == "Location"
        assert classify_disambiguation_hint("sông") == "Location"

    def test_org_hint(self):
        assert classify_disambiguation_hint("tổ chức") == "Organization"
        assert classify_disambiguation_hint("công ty") == "Organization"

    def test_person_hint(self):
        assert classify_disambiguation_hint("chính trị gia") == "Person"
        assert classify_disambiguation_hint("ca sĩ") == "Person"

    def test_work_hint(self):
        assert classify_disambiguation_hint("phim") == "Work"
        assert classify_disambiguation_hint("tiểu thuyết") == "Work"

    def test_unknown_hint(self):
        assert classify_disambiguation_hint("word") == "Unknown"
        assert classify_disambiguation_hint("") == "Unknown"

    def test_wikilink_uses_hint_for_typing(self):
        raw = "[[Hà Nội (thành phố)]] và [[Đảng Cộng sản (đảng)]] và [[Titanic (phim)]]"
        entities = _extract_entities_wikilink(raw)
        type_map = {name: etype for name, etype in entities}
        assert type_map.get("Hà Nội") == "Location"
        assert type_map.get("Đảng Cộng sản") == "Organization"
        assert type_map.get("Titanic") == "Work"
