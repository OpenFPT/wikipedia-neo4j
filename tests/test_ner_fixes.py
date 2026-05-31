"""Regression tests for NER data quality fixes."""

from src.ner import (
    classify_entity_type,
    classify_disambiguation_hint,
    normalize_entity,
    strip_disambiguation,
    _extract_entities_wikilink,
)
from src.text_utils import strip_wiki_markup


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


class TestStripWikiMarkup:
    def test_removes_templates(self):
        assert strip_wiki_markup("Hello {{cite web|url=x}} world") == "Hello world"

    def test_removes_ref_tags(self):
        text = "Fact<ref>source</ref> here<ref name='a'/>"
        assert strip_wiki_markup(text) == "Fact here"

    def test_removes_multiline_ref(self):
        text = "Start<ref>\nmultiline\ncontent\n</ref>End"
        assert strip_wiki_markup(text) == "StartEnd"

    def test_removes_html_tags(self):
        assert strip_wiki_markup("<b>bold</b> and <i>italic</i>") == "bold and italic"

    def test_strips_headings(self):
        text = "== Lịch sử ==\nContent here"
        result = strip_wiki_markup(text)
        assert "==" not in result
        assert "Lịch sử" in result
        assert "Content here" in result

    def test_strips_bold_italic(self):
        assert strip_wiki_markup("'''Hà Nội''' là ''thủ đô''") == "Hà Nội là thủ đô"

    def test_removes_category_links(self):
        text = "Text [[Thể loại:Thành phố]] end"
        assert strip_wiki_markup(text) == "Text end"

    def test_removes_file_links(self):
        text = "Before [[Tập tin:Map.png|thumb|Caption]] after"
        assert strip_wiki_markup(text) == "Before after"

    def test_piped_wikilink_keeps_display(self):
        assert strip_wiki_markup("[[Việt Nam|Vietnam]]") == "Vietnam"

    def test_plain_wikilink_keeps_target(self):
        assert strip_wiki_markup("[[Hà Nội]]") == "Hà Nội"

    def test_collapses_whitespace(self):
        text = "a   b  \n  c"
        assert strip_wiki_markup(text) == "a b c"

    def test_combined_markup(self):
        text = "== Tiêu đề ==\n'''Hà Nội'''{{ref|x}} là [[thủ đô]] của [[Việt Nam|VN]]<ref>src</ref>."
        result = strip_wiki_markup(text)
        assert "==" not in result
        assert "{{" not in result
        assert "<ref" not in result
        assert "Hà Nội" in result
        assert "thủ đô" in result
        assert "VN" in result
