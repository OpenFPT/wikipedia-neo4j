"""Backward-compatibility shim: imports redirect to src.ingestion.pipeline."""

from src.ingestion.pipeline import (  # noqa: F401
    IngestResult,
    ingest_topic,
    ingest_from_hf,
)
