from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import src.infrastructure.neo4j_client as n4


class _FakeDriver:
    def __init__(self) -> None:
        self.closed = False
        self.verified = False

    def close(self) -> None:
        self.closed = True

    def verify_connectivity(self) -> None:
        self.verified = True

    @contextmanager
    def session(self) -> Iterator["_FakeSession"]:
        yield _FakeSession()


class _FakeSession:
    def __init__(self) -> None:
        self.runs: list[str] = []

    def run(self, query: str):
        self.runs.append(query)


def test_close_calls_driver_close() -> None:
    client = n4.Neo4jClient.__new__(n4.Neo4jClient)
    client.driver = _FakeDriver()

    client.close()

    assert client.driver.closed is True


def test_verify_connectivity_calls_driver() -> None:
    client = n4.Neo4jClient.__new__(n4.Neo4jClient)
    client.driver = _FakeDriver()

    client.verify_connectivity()

    assert client.driver.verified is True


def test_get_server_version() -> None:
    class _VersionSession:
        def run(self, query, **params):
            class _Result:
                def single(self):
                    return {"version": "5.15.0"}
            return _Result()

        def close(self):
            pass

    class _VersionDriver:
        def session(self):
            return _VersionSession()

        def verify_connectivity(self):
            pass

    client = n4.Neo4jClient.__new__(n4.Neo4jClient)
    client.driver = _VersionDriver()

    version = client.get_server_version()
    assert version == "5.15.0"


def test_get_server_version_unknown() -> None:
    class _NoneSession:
        def run(self, query, **params):
            class _Result:
                def single(self):
                    return None
            return _Result()

        def close(self):
            pass

    class _NoneDriver:
        def session(self):
            return _NoneSession()

        def verify_connectivity(self):
            pass

    client = n4.Neo4jClient.__new__(n4.Neo4jClient)
    client.driver = _NoneDriver()

    version = client.get_server_version()
    assert version == "unknown"


def test_run_batch() -> None:
    runs = []

    class _BatchSession:
        def run(self, cypher, **params):
            runs.append(params.get("rows", []))

        def close(self):
            pass

    class _BatchDriver:
        def session(self):
            return _BatchSession()

        def verify_connectivity(self):
            pass

    client = n4.Neo4jClient.__new__(n4.Neo4jClient)
    client.driver = _BatchDriver()

    rows = [{"id": i} for i in range(5)]
    total = client.run_batch("UNWIND $rows AS row CREATE (n {id: row.id})", rows, batch_size=2)

    assert total == 5
    assert len(runs) == 3


def test_session_retry_on_first_failure() -> None:
    call_count = [0]

    class _RetrySession:
        def run(self, query, **params):
            pass

        def close(self):
            pass

    class _RetryDriver:
        def __init__(self):
            self.verified = False

        def session(self):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("connection lost")
            return _RetrySession()

        def verify_connectivity(self):
            self.verified = True

    client = n4.Neo4jClient.__new__(n4.Neo4jClient)
    client.driver = _RetryDriver()

    with client.session() as session:
        assert session is not None
    assert client.driver.verified is True
