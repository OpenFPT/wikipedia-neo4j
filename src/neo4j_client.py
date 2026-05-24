"""Neo4j driver wrapper and schema setup utilities."""

from __future__ import annotations

from contextlib import contextmanager

from neo4j import GraphDatabase

from src.config import settings
from src.logging_utils import get_logger


logger = get_logger(__name__)


class Neo4jClient:
    """Thin wrapper around the Neo4j Python driver."""

    def __init__(self) -> None:
        """Initialize Neo4j driver from application settings."""
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
            connection_timeout=30,
            max_connection_pool_size=50,
            connection_acquisition_timeout=60,
        )
        logger.debug("Neo4j driver initialized", extra={"uri": settings.neo4j_uri})

    def close(self) -> None:
        """Close shared Neo4j driver."""
        self.driver.close()
        logger.info("Neo4j driver closed")

    def verify_connectivity(self) -> None:
        """Verify connectivity to Neo4j server."""
        self.driver.verify_connectivity()
        logger.debug("Neo4j connectivity verified")

    @contextmanager
    def session(self):
        """Yield a Neo4j session with guaranteed cleanup."""
        session = self.driver.session()
        try:
            yield session
        finally:
            session.close()

    def setup_schema(self) -> None:
        """Create required constraints and indexes if missing."""
        with self.session() as session:
            session.run(
                "CREATE CONSTRAINT page_id IF NOT EXISTS FOR (p:Page) REQUIRE p.id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE"
            )
            session.run(
                "CREATE FULLTEXT INDEX page_title_ft IF NOT EXISTS FOR (p:Page) ON EACH [p.title, p.summary]"
            )
            session.run(
                "CREATE FULLTEXT INDEX chunk_text_ft IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]"
            )
            session.run("CREATE INDEX page_title_idx IF NOT EXISTS FOR (p:Page) ON (p.title)")
            session.run("CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)")
            session.run("CREATE INDEX person_name_idx IF NOT EXISTS FOR (p:Person) ON (p.name)")
            session.run("CREATE INDEX organization_name_idx IF NOT EXISTS FOR (o:Organization) ON (o.name)")
            session.run("CREATE INDEX location_name_idx IF NOT EXISTS FOR (l:Location) ON (l.name)")
            session.run("CREATE INDEX work_name_idx IF NOT EXISTS FOR (w:Work) ON (w.name)")
            session.run(
                "CREATE FULLTEXT INDEX entity_alias_ft IF NOT EXISTS "
                "FOR (e:Entity) ON EACH [e.name, e.aliases]"
            )
        logger.debug("Neo4j schema ensured")


neo4j_client = Neo4jClient()
