"""Enrich Neo4j graph with typed relations from Wikidata SPARQL.

For each Page node in Neo4j, resolves its Wikidata QID from the Wikipedia URL,
then queries Wikidata for structured triples (birth year, occupation, located in,
author of, etc.) and writes typed semantic relations into the graph.

Usage:
    uv run python scripts/enrich_wikidata.py --limit 100 --batch-size 50
"""

from __future__ import annotations

import argparse
import time
import urllib.parse
from typing import Any

import requests

from src.logging_utils import get_logger
from src.neo4j_client import neo4j_client

logger = get_logger(__name__)

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
USER_AGENT = "ViWikiMHR-Enricher/1.0 (academic research)"

WIKIDATA_PROPERTY_MAP: dict[str, dict[str, str]] = {
    "P19": {"relation": "SINH_TẠI", "target_type": "Location"},
    "P20": {"relation": "MẤT_TẠI", "target_type": "Location"},
    "P27": {"relation": "QUỐC_TỊCH", "target_type": "Location"},
    "P50": {"relation": "TÁC_GIẢ", "target_type": "Person"},
    "P57": {"relation": "ĐẠO_DIỄN", "target_type": "Person"},
    "P86": {"relation": "NHẠC_SĨ", "target_type": "Person"},
    "P112": {"relation": "SÁNG_LẬP", "target_type": "Person"},
    "P127": {"relation": "SỞ_HỮU", "target_type": "Organization"},
    "P131": {"relation": "THUỘC_ĐỊA_PHƯƠNG", "target_type": "Location"},
    "P159": {"relation": "TRỤ_SỞ", "target_type": "Location"},
    "P170": {"relation": "SÁNG_TÁC", "target_type": "Person"},
    "P175": {"relation": "TRÌNH_BÀY", "target_type": "Person"},
    "P264": {"relation": "HÃNG_ĐĨA", "target_type": "Organization"},
    "P276": {"relation": "TỌA_LẠC", "target_type": "Location"},
    "P361": {"relation": "THUỘC_VỀ", "target_type": "Unknown"},
    "P463": {"relation": "THÀNH_VIÊN", "target_type": "Organization"},
    "P495": {"relation": "XUẤT_XỨ", "target_type": "Location"},
    "P740": {"relation": "THÀNH_LẬP_TẠI", "target_type": "Location"},
    "P800": {"relation": "TÁC_PHẨM_TIÊU_BIỂU", "target_type": "Work"},
    "P937": {"relation": "LÀM_VIỆC_TẠI", "target_type": "Location"},
    "P1376": {"relation": "THỦ_PHỦ_CỦA", "target_type": "Location"},
}

LITERAL_PROPERTIES: dict[str, str] = {
    "P569": "năm_sinh",
    "P570": "năm_mất",
    "P571": "năm_thành_lập",
    "P1082": "dân_số",
    "P2046": "diện_tích_km2",
}


def _sparql_query(query: str) -> list[dict[str, Any]]:
    """Execute a SPARQL query against Wikidata and return bindings."""
    headers = {"Accept": "application/sparql-results+json", "User-Agent": USER_AGENT}
    params = {"query": query}
    resp = requests.get(WIKIDATA_SPARQL_URL, params=params, headers=headers, timeout=30)
    if resp.status_code == 429:
        retry_after = int(resp.headers.get("Retry-After", "60"))
        logger.warning("Wikidata rate limit hit, sleeping", extra={"retry_after": retry_after})
        time.sleep(retry_after)
        resp = requests.get(WIKIDATA_SPARQL_URL, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json().get("results", {}).get("bindings", [])


def _get_qids_batch(titles: list[str]) -> dict[str, str]:
    """Resolve Wikipedia vi titles to Wikidata QIDs in batch via SPARQL."""
    if not titles:
        return {}
    values = " ".join(f'"{t}"@vi' for t in titles)
    query = f"""
    SELECT ?item ?title WHERE {{
      VALUES ?title {{ {values} }}
      ?article schema:about ?item ;
               schema:isPartOf <https://vi.wikipedia.org/> ;
               schema:name ?title .
    }}
    """
    results = _sparql_query(query)
    mapping = {}
    for r in results:
        qid = r["item"]["value"].rsplit("/", 1)[-1]
        title = r["title"]["value"]
        mapping[title] = qid
    return mapping


def _get_relations_batch(qids: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Fetch object-property relations for a batch of QIDs."""
    values = " ".join(f"wd:{qid}" for qid in qids)
    prop_ids = list(WIKIDATA_PROPERTY_MAP.keys())
    prop_values = " ".join(f"wdt:{p}" for p in prop_ids)

    query = f"""
    SELECT ?item ?prop ?value ?valueLabel WHERE {{
      VALUES ?item {{ {values} }}
      VALUES ?prop {{ {prop_values} }}
      ?item ?prop ?value .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "vi,en". }}
    }}
    """
    results = _sparql_query(query)

    grouped: dict[str, list[dict[str, Any]]] = {qid: [] for qid in qids}
    for r in results:
        qid = r["item"]["value"].rsplit("/", 1)[-1]
        prop = r["prop"]["value"].rsplit("/", 1)[-1]
        value_label = r.get("valueLabel", {}).get("value", "")
        if qid in grouped:
            grouped[qid].append({"prop": prop, "value": r["value"]["value"], "label": value_label})
    return grouped


def _get_literals_batch(qids: list[str]) -> dict[str, dict[str, str]]:
    """Fetch literal-value properties (dates, numbers) for a batch of QIDs."""
    values = " ".join(f"wd:{qid}" for qid in qids)
    prop_ids = list(LITERAL_PROPERTIES.keys())
    prop_values = " ".join(f"wdt:{p}" for p in prop_ids)

    query = f"""
    SELECT ?item ?prop ?value WHERE {{
      VALUES ?item {{ {values} }}
      VALUES ?prop {{ {prop_values} }}
      ?item ?prop ?value .
    }}
    """
    results = _sparql_query(query)

    grouped: dict[str, dict[str, str]] = {qid: {} for qid in qids}
    for r in results:
        qid = r["item"]["value"].rsplit("/", 1)[-1]
        prop = r["prop"]["value"].rsplit("/", 1)[-1]
        value = r["value"]["value"]
        if qid in grouped:
            grouped[qid][prop] = value
    return grouped


def _write_relations_to_neo4j(
    page_title: str, relations: list[dict[str, Any]], literals: dict[str, str]
) -> int:
    """Write Wikidata relations into Neo4j as typed edges and properties."""
    written = 0

    with neo4j_client.session() as session:
        for lit_prop, lit_value in literals.items():
            prop_name = LITERAL_PROPERTIES.get(lit_prop)
            if not prop_name:
                continue
            if "T" in lit_value:
                lit_value = lit_value.split("T")[0][:4]
            session.run(
                f"MATCH (p:Page {{title: $title}}) SET p.`{prop_name}` = $value",
                title=page_title,
                value=lit_value,
            )
            written += 1

        for rel in relations:
            prop_id = rel["prop"]
            mapping = WIKIDATA_PROPERTY_MAP.get(prop_id)
            if not mapping:
                continue
            rel_type = mapping["relation"]
            target_type = mapping["target_type"]
            target_label = rel["label"] or rel["value"].rsplit("/", 1)[-1]

            if not target_label or target_label.startswith("Q"):
                continue

            session.run(
                f"""
                MATCH (source:Page {{title: $source_title}})
                MERGE (target:Entity {{name: $target_name}})
                ON CREATE SET target.type = $target_type
                FOREACH (_ IN CASE WHEN $target_type = 'Person' THEN [1] ELSE [] END | SET target:Person)
                FOREACH (_ IN CASE WHEN $target_type = 'Organization' THEN [1] ELSE [] END | SET target:Organization)
                FOREACH (_ IN CASE WHEN $target_type = 'Location' THEN [1] ELSE [] END | SET target:Location)
                FOREACH (_ IN CASE WHEN $target_type = 'Work' THEN [1] ELSE [] END | SET target:Work)
                MERGE (source)-[:`{rel_type}`]->(target)
                """,
                source_title=page_title,
                target_name=target_label,
                target_type=target_type,
            )
            written += 1

    return written


def enrich_pages(limit: int | None = None, batch_size: int = 50) -> dict[str, int]:
    """Main enrichment loop: fetch pages from Neo4j, resolve QIDs, write relations."""
    limit_clause = f"LIMIT {limit}" if limit else ""

    with neo4j_client.session() as session:
        result = session.run(
            f"MATCH (p:Page) WHERE p.title IS NOT NULL RETURN p.title AS title, p.url AS url {limit_clause}"
        )
        pages = [(r["title"], r["url"]) for r in result]

    logger.info("Starting Wikidata enrichment", extra={"pages": len(pages)})

    stats = {"pages_processed": 0, "qids_resolved": 0, "relations_written": 0, "errors": 0}

    for i in range(0, len(pages), batch_size):
        batch = pages[i : i + batch_size]
        titles = [t for t, _ in batch]

        try:
            qid_map = _get_qids_batch(titles)
        except Exception as e:
            logger.warning("QID batch failed", extra={"error": str(e), "batch_start": i})
            stats["errors"] += 1
            time.sleep(2)
            continue

        stats["qids_resolved"] += len(qid_map)

        if not qid_map:
            stats["pages_processed"] += len(batch)
            continue

        qids = list(qid_map.values())
        title_by_qid = {v: k for k, v in qid_map.items()}

        try:
            relations_map = _get_relations_batch(qids)
            literals_map = _get_literals_batch(qids)
        except Exception as e:
            logger.warning("Relations batch failed", extra={"error": str(e), "batch_start": i})
            stats["errors"] += 1
            time.sleep(2)
            continue

        for qid in qids:
            title = title_by_qid.get(qid, "")
            rels = relations_map.get(qid, [])
            lits = literals_map.get(qid, {})
            written = _write_relations_to_neo4j(title, rels, lits)
            stats["relations_written"] += written

        stats["pages_processed"] += len(batch)

        if (i // batch_size) % 5 == 0:
            logger.info("Enrichment progress", extra={
                "processed": stats["pages_processed"],
                "total": len(pages),
                "relations": stats["relations_written"],
            })

        time.sleep(1)

    logger.info("Wikidata enrichment complete", extra=stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich Neo4j graph with Wikidata relations")
    parser.add_argument("--limit", type=int, default=None, help="Max pages to process")
    parser.add_argument("--batch-size", type=int, default=50, help="SPARQL batch size")
    args = parser.parse_args()

    neo4j_client.setup_schema()
    stats = enrich_pages(limit=args.limit, batch_size=args.batch_size)
    print(f"Done: {stats}")


if __name__ == "__main__":
    main()
