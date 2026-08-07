from __future__ import annotations

import sys

from neo4j import GraphDatabase

from src.config import settings, validate_runtime_settings


def main() -> int:
    validate_runtime_settings()

    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )
    try:
        driver.verify_connectivity()
        with driver.session() as session:
            value = session.run("RETURN 1 AS ok").single()["ok"]
        print(f"Neo4j OK (RETURN 1 -> {value})")
        return 0
    finally:
        driver.close()


if __name__ == "__main__":
    raise SystemExit(main())

