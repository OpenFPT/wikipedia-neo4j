"""Tests for entity resolution module."""

from src.extraction.entity_resolution import (
    EntityResolver,
    ResolvedEntity,
    normalize_key,
    normalize_key_no_diacritics,
    remove_diacritics,
    KNOWN_ALIASES,
)


class TestRemoveDiacritics:
    def test_basic(self):
        assert remove_diacritics("Hồ Chí Minh") == "Ho Chi Minh"

    def test_no_diacritics(self):
        assert remove_diacritics("Hello World") == "Hello World"

    def test_empty(self):
        assert remove_diacritics("") == ""


class TestNormalizeKey:
    def test_strips_and_lowers(self):
        assert normalize_key("  Hà Nội  ") == "hà nội"

    def test_collapses_whitespace(self):
        assert normalize_key("Hồ  Chí   Minh") == "hồ chí minh"


class TestNormalizeKeyNoDiacritics:
    def test_combined(self):
        assert normalize_key_no_diacritics("Hồ Chí Minh") == "ho chi minh"


class TestResolvedEntity:
    def test_id_property(self):
        entity = ResolvedEntity(canonical_name="Hà Nội", entity_type="Location")
        assert entity.id == normalize_key("Hà Nội")


class TestEntityResolver:
    def test_known_alias_resolves(self):
        resolver = EntityResolver()
        result = resolver.resolve("Bác Hồ", "Person")
        assert result.canonical_name == "Hồ Chí Minh"
        assert "Bác Hồ" in result.aliases

    def test_known_alias_saigon(self):
        resolver = EntityResolver()
        result = resolver.resolve("Sài Gòn", "Location")
        assert result.canonical_name == "Thành phố Hồ Chí Minh"

    def test_new_entity_created(self):
        resolver = EntityResolver()
        result = resolver.resolve("Nguyễn Văn A", "Person")
        assert result.canonical_name == "Nguyễn Văn A"
        assert result.entity_type == "Person"

    def test_same_entity_resolved_twice(self):
        resolver = EntityResolver()
        r1 = resolver.resolve("Nguyễn Văn B", "Person")
        r2 = resolver.resolve("Nguyễn Văn B", "Person")
        assert r1 is r2

    def test_diacritics_match(self):
        resolver = EntityResolver()
        resolver.resolve("Hà Nội", "Location")
        result = resolver.resolve("Ha Noi", "Location")
        assert result.canonical_name == "Hà Nội"
        assert "Ha Noi" in result.aliases

    def test_type_upgrade_from_unknown(self):
        resolver = EntityResolver()
        r1 = resolver.resolve("Test Entity", "Unknown")
        assert r1.entity_type == "Unknown"
        r2 = resolver.resolve("Test Entity", "Person")
        assert r2.entity_type == "Person"

    def test_resolve_batch(self):
        resolver = EntityResolver()
        results = resolver.resolve_batch([
            ("Hà Nội", "Location"),
            ("Bác Hồ", "Person"),
        ])
        assert len(results) == 2
        assert results[0].canonical_name == "Hà Nội"
        assert results[1].canonical_name == "Hồ Chí Minh"

    def test_entities_property(self):
        resolver = EntityResolver()
        resolver.resolve("Entity A", "Person")
        resolver.resolve("Entity B", "Location")
        entities = resolver.entities
        assert len(entities) >= 2

    def test_stats(self):
        resolver = EntityResolver()
        resolver.resolve("New One", "Person")
        s = resolver.stats()
        assert "total_entities" in s
        assert "total_aliases" in s
        assert s["total_entities"] >= len(KNOWN_ALIASES) + 1

    def test_custom_alias_map(self):
        custom = {"Foo": ["Bar", "Baz"]}
        resolver = EntityResolver(alias_map=custom)
        result = resolver.resolve("Bar", "Unknown")
        assert result.canonical_name == "Foo"
        assert "Bar" in result.aliases
