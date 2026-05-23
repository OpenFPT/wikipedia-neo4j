"""Setup extended Neo4j schema with typed nodes, relationships, and constraints."""

from __future__ import annotations

from src.config import settings
from src.logging_utils import get_logger

from neo4j import GraphDatabase

logger = get_logger(__name__)

CONSTRAINTS = [
    "CREATE CONSTRAINT article_id IF NOT EXISTS FOR (a:Article) REQUIRE a.id IS UNIQUE",
    "CREATE CONSTRAINT paragraph_id IF NOT EXISTS FOR (p:Paragraph) REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (n:Person) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT organization_id IF NOT EXISTS FOR (o:Organization) REQUIRE o.id IS UNIQUE",
    "CREATE CONSTRAINT location_id IF NOT EXISTS FOR (l:Location) REQUIRE l.id IS UNIQUE",
    "CREATE CONSTRAINT work_id IF NOT EXISTS FOR (w:Work) REQUIRE w.id IS UNIQUE",
    "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX article_title_idx IF NOT EXISTS FOR (a:Article) ON (a.title)",
    "CREATE INDEX person_name_idx IF NOT EXISTS FOR (n:Person) ON (n.name)",
    "CREATE INDEX organization_name_idx IF NOT EXISTS FOR (o:Organization) ON (o.name)",
    "CREATE INDEX location_name_idx IF NOT EXISTS FOR (l:Location) ON (l.name)",
    "CREATE INDEX work_name_idx IF NOT EXISTS FOR (w:Work) ON (w.name)",
    "CREATE INDEX event_name_idx IF NOT EXISTS FOR (e:Event) ON (e.name)",
    "CREATE INDEX paragraph_article_idx IF NOT EXISTS FOR (p:Paragraph) ON (p.article_id)",
    "CREATE FULLTEXT INDEX article_title_ft IF NOT EXISTS FOR (a:Article) ON EACH [a.title]",
    "CREATE FULLTEXT INDEX paragraph_text_ft IF NOT EXISTS FOR (p:Paragraph) ON EACH [p.text]",
    "CREATE FULLTEXT INDEX entity_name_ft IF NOT EXISTS FOR (n:Person|Organization|Location|Work|Event) ON EACH [n.name, n.aliases_text]",
]


def setup_schema() -> None:
    """Create all constraints and indexes in Neo4j."""
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )

    with driver.session() as session:
        for stmt in CONSTRAINTS:
            session.run(stmt)
            logger.debug("Created constraint", extra={"statement": stmt})

        for stmt in INDEXES:
            session.run(stmt)
            logger.debug("Created index", extra={"statement": stmt})

    driver.close()
    logger.info("Neo4j schema setup complete", extra={
        "constraints": len(CONSTRAINTS),
        "indexes": len(INDEXES),
    })


def main() -> None:
    print("Setting up Neo4j schema...")
    setup_schema()
    print("Done.")


if __name__ == "__main__":
    main()
