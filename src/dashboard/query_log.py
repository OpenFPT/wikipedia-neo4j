"""Persistent query log with in-memory ring buffer and JSONL file backing."""

from __future__ import annotations

import json
import os
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class QueryLogEntry:
    """Single query log record."""

    timestamp: str
    question: str
    retrieval_tier: str
    latency_ms: int
    result_count: int
    signal_scores: dict[str, int] = field(default_factory=dict)


_DEFAULT_LOG_PATH = Path(os.environ.get("QUERY_LOG_PATH", "logs/queries.jsonl"))


class QueryLog:
    """Thread-safe ring buffer with JSONL persistence."""

    def __init__(
        self,
        maxlen: int = 50,
        log_path: Path | None = None,
        *,
        load_existing: bool = False,
    ) -> None:
        self._buffer: deque[QueryLogEntry] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._log_path = log_path or _DEFAULT_LOG_PATH
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        if load_existing:
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Load recent entries from JSONL on startup."""
        if not self._log_path.exists():
            return
        try:
            lines = self._log_path.read_text(encoding="utf-8").strip().splitlines()
            entries = []
            tail = lines[-self._buffer.maxlen:] if self._buffer.maxlen else lines
            for line in tail:
                try:
                    data = json.loads(line)
                    entries.append(QueryLogEntry(**data))
                except (json.JSONDecodeError, TypeError):
                    continue
            for entry in entries:
                self._buffer.append(entry)
        except OSError as e:
            logger.warning("query_log: failed to load history from %s: %s", self._log_path, e)

    def _persist(self, entry: QueryLogEntry) -> None:
        """Append a single entry to the JSONL file."""
        try:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning("query_log: failed to persist entry to %s: %s", self._log_path, e)

    def append(self, entry: QueryLogEntry) -> None:
        """Add a new entry to the log and persist to disk."""
        with self._lock:
            self._buffer.appendleft(entry)
        self._persist(entry)

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


query_log = QueryLog(load_existing=True)
