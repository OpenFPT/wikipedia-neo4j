"""Entity resolution for Vietnamese text: merge diacritic variants and aliases."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


@dataclass
class ResolvedEntity:
    """A canonical entity with all known surface forms."""

    canonical_name: str
    entity_type: str
    aliases: set[str] = field(default_factory=set)

    @property
    def id(self) -> str:
        return normalize_key(self.canonical_name)


KNOWN_ALIASES: dict[str, list[str]] = {
    "Hồ Chí Minh": ["Bác Hồ", "Nguyễn Tất Thành", "Nguyễn Ái Quốc", "Nguyễn Sinh Cung"],
    "Hà Nội": ["Thăng Long", "Đông Đô", "Kẻ Chợ"],
    "Thành phố Hồ Chí Minh": ["Sài Gòn", "TP.HCM", "TPHCM", "TP HCM"],
    "Việt Nam": ["Vietnam", "Viet Nam", "VN"],
    "Đảng Cộng sản Việt Nam": ["ĐCSVN", "Đảng Cộng sản"],
}


def remove_diacritics(text: str) -> str:
    """Remove Vietnamese diacritics for fuzzy matching."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize_key(name: str) -> str:
    """Produce a stable key for entity deduplication."""
    lowered = name.strip().lower()
    collapsed = re.sub(r"\s+", " ", lowered)
    return collapsed


def normalize_key_no_diacritics(name: str) -> str:
    """Key without diacritics — for cross-variant matching."""
    return remove_diacritics(normalize_key(name))


class EntityResolver:
    """Resolve entity mentions to canonical forms."""

    def __init__(self, alias_map: dict[str, list[str]] | None = None) -> None:
        self._canonical: dict[str, ResolvedEntity] = {}
        self._key_to_canonical: dict[str, str] = {}
        self._no_diac_to_canonical: dict[str, str] = {}

        aliases = alias_map or KNOWN_ALIASES
        for canonical, alias_list in aliases.items():
            self._register_canonical(canonical, "Unknown", alias_list)

    def _register_canonical(
        self, name: str, entity_type: str, aliases: list[str] | None = None
    ) -> None:
        key = normalize_key(name)
        no_diac_key = normalize_key_no_diacritics(name)

        if key not in self._canonical:
            self._canonical[key] = ResolvedEntity(
                canonical_name=name, entity_type=entity_type
            )

        self._key_to_canonical[key] = key
        self._no_diac_to_canonical[no_diac_key] = key

        for alias in aliases or []:
            alias_key = normalize_key(alias)
            alias_no_diac = normalize_key_no_diacritics(alias)
            self._canonical[key].aliases.add(alias)
            self._key_to_canonical[alias_key] = key
            self._no_diac_to_canonical[alias_no_diac] = key

    def resolve(self, name: str, entity_type: str = "Unknown") -> ResolvedEntity:
        """Resolve a mention to its canonical entity, creating one if new."""
        key = normalize_key(name)

        # Exact match
        if key in self._key_to_canonical:
            canonical_key = self._key_to_canonical[key]
            entity = self._canonical[canonical_key]
            if entity_type != "Unknown":
                entity.entity_type = entity_type
            if key != canonical_key:
                entity.aliases.add(name)
            return entity

        # Diacritics-stripped match
        no_diac_key = normalize_key_no_diacritics(name)
        if no_diac_key in self._no_diac_to_canonical:
            canonical_key = self._no_diac_to_canonical[no_diac_key]
            entity = self._canonical[canonical_key]
            if entity_type != "Unknown":
                entity.entity_type = entity_type
            entity.aliases.add(name)
            self._key_to_canonical[key] = canonical_key
            return entity

        # New entity
        self._register_canonical(name, entity_type)
        return self._canonical[key]

    def resolve_batch(
        self, mentions: list[tuple[str, str]]
    ) -> list[ResolvedEntity]:
        """Resolve a batch of (name, type) pairs."""
        return [self.resolve(name, etype) for name, etype in mentions]

    @property
    def entities(self) -> list[ResolvedEntity]:
        """All known canonical entities."""
        return list(self._canonical.values())

    def stats(self) -> dict[str, int]:
        return {
            "total_entities": len(self._canonical),
            "total_aliases": sum(len(e.aliases) for e in self._canonical.values()),
        }
