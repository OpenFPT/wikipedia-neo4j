"""In-memory ring buffer for recent query log entries."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import asdict, dataclass, field


@dataclass
class QueryLogEntry:
    """Single query log record."""

    timestamp: str
    question: str
    retrieval_tier: str
    latency_ms: int
    result_count: int
    signal_scores: dict[str, int] = field(default_factory=dict)


class QueryLog:
    """Thread-safe ring buffer holding recent query entries."""

    def __init__(self, maxlen: int = 50) -> None:
        self._buffer: deque[QueryLogEntry] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, entry: QueryLogEntry) -> None:
        """Add a new entry to the log."""
        with self._lock:
            self._buffer.appendleft(entry)

    def recent(self, n: int = 20) -> list[dict]:
        """Return the most recent n entries as dicts (newest first)."""
        with self._lock:
            items = list(self._buffer)[:n]
        return [asdict(e) for e in items]

    def latest(self) -> QueryLogEntry | None:
        """Return the most recent entry, or None."""
        with self._lock:
            return self._buffer[0] if self._buffer else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)


query_log = QueryLog()
