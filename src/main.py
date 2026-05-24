"""FastAPI entrypoint with API guardrails and ingestion job orchestration."""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from contextvars import Token
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field

from src.config import (
    resolve_cypher_model,
    resolve_orchestrator_model,
    settings,
    validate_runtime_settings,
)
from src.ingest import IngestResult, ingest_from_hf, ingest_topic
from src.job_store import JobStore
from src.errors import ErrorCode, ErrorResponse, normalize_error_detail
from src.export import export_csv, export_jsonl
from src.logging_utils import (
    configure_logging,
    get_logger,
    get_request_id,
    reset_request_id,
    set_request_id,
)
from src.neo4j_client import neo4j_client
from src.retrieve import QueryResult, query_graph


configure_logging(settings.log_level, json_logs=settings.json_logs)
logger = get_logger(__name__)


class _RateLimiter:
    """Simple in-memory fixed-window rate limiter per client key."""

    def __init__(self, max_requests: int, period_seconds: int = 60) -> None:
        self.max_requests: int = max_requests
        self.period_seconds: int = period_seconds
        self._lock: threading.Lock = threading.Lock()
        self._hits: dict[str, deque[float]] = {}

    def allow(self, key: str) -> bool:
        """Return whether a request is allowed for given client key."""
        now = time.time()
        with self._lock:
            bucket = self._hits.setdefault(key, deque())
            cutoff = now - self.period_seconds
            while bucket and bucket[0] < cutoff:
                _ = bucket.popleft()
            if len(bucket) >= self.max_requests:
                return False
            bucket.append(now)
            return True


rate_limiter = _RateLimiter(max_requests=settings.rate_limit_per_minute, period_seconds=60)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize runtime configuration and release resources on shutdown."""
    validate_runtime_settings()
    logger.info("Service starting")
    try:
        yield
    finally:
        neo4j_client.close()
        logger.info("Service shutdown complete")


app = FastAPI(title="Wikipedia Neo4j GraphRAG Demo", version="0.1.0", lifespan=lifespan)


@app.exception_handler(HTTPException)
def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    """Return standardized error payload for HTTP exceptions."""
    request_id = get_request_id()
    error_code, message, hint = normalize_error_detail(exc.detail, exc.status_code)
    payload = ErrorResponse(
        error_code=error_code.value,
        message=message,
        request_id=request_id,
        hint=hint,
    )
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump(exclude_none=True))


@app.exception_handler(RequestValidationError)
def validation_exception_handler(
    _request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Return standardized error payload for validation failures."""
    request_id = get_request_id()
    message = "Invalid request payload"
    payload = ErrorResponse(
        error_code=ErrorCode.invalid_request.value,
        message=message,
        request_id=request_id,
        hint="Check required fields and value ranges",
    )
    logger.warning("Validation failed", extra={"error": str(exc)})
    return JSONResponse(status_code=422, content=payload.model_dump(exclude_none=True))


@app.exception_handler(Exception)
def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Return standardized error payload for unhandled exceptions."""
    request_id = get_request_id()
    logger.exception("Unhandled error", extra={"error": str(exc)})
    payload = ErrorResponse(
        error_code=ErrorCode.cypher_generation_failed.value,
        message="Request failed due to server error",
        request_id=request_id,
        hint="Check server logs with request_id",
    )
    return JSONResponse(status_code=500, content=payload.model_dump(exclude_none=True))


class IngestRequest(BaseModel):
    """Request payload for Wikipedia topic ingestion."""

    topics: list[str] = Field(min_length=1, description="Wikipedia page topics")


class QueryRequest(BaseModel):
    """Request payload for query endpoint."""

    question: str = Field(min_length=3)
    top_k: int = Field(default=4, ge=1, le=20)


class HFDatasetIngestRequest(BaseModel):
    """Request payload for direct HF dataset ingestion endpoint."""

    config_name: str = Field(default="20231101.en", description="HF config, e.g. 20231101.en")
    split: str = Field(default="train")
    sample_size: int = Field(default=5, ge=1, le=200)
    streaming: bool = Field(default=True, description="Use HF streaming mode for large configs")


class HFIngestJobRequest(HFDatasetIngestRequest):
    """Request payload for background HF ingestion job."""


class _JobState(BaseModel):
    """Persisted state model for one background HF ingestion job."""

    job_id: str
    status: str
    config_name: str
    split: str
    sample_size: int
    streaming: bool
    started_at: str
    finished_at: str | None = None
    processed: int = 0
    total: int | None = None
    last_title: str | None = None
    error: str | None = None
    ingested: list[dict[str, object]] = Field(default_factory=list)


_jobs_lock = threading.Lock()
_jobs: dict[str, _JobState] = {}
_job_stops: dict[str, threading.Event] = {}
_job_store: JobStore = JobStore(".hf_ingest_jobs.json")


def _serialize_ingest_result(result: IngestResult) -> dict[str, object]:
    """Convert internal ingest result object to API response dictionary."""
    return {
        "topic": result.topic,
        "page_id": result.page_id,
        "title": result.title,
        "url": result.url,
        "chunk_count": result.chunk_count,
        "entity_count": result.entity_count,
    }


def _persist_job(job: _JobState) -> None:
    """Persist one job state to storage."""
    _job_store.upsert(job.job_id, job.model_dump())


def _restore_jobs() -> None:
    """Restore persisted jobs and normalize stale in-progress states."""
    persisted: dict[str, dict[str, object]] = _job_store.load_all()
    for job_id, payload in persisted.items():
        try:
            job = _JobState.model_validate(payload)
        except (TypeError, ValueError):
            logger.warning("Skipping invalid persisted job payload", extra={"job_id": job_id})
            continue
        if job.status in {"running", "cancelling"}:
            job.status = "interrupted"
            if not job.error:
                job.error = "Server restarted while job was in progress"
            if not job.finished_at:
                job.finished_at = datetime.now(timezone.utc).isoformat()
        _jobs[job_id] = job
        _persist_job(job)


_restore_jobs()


def _request_id(request: Request) -> str:
    """Resolve request id from header or generate one."""
    header_value = request.headers.get("X-Request-ID")
    return header_value if header_value else str(uuid.uuid4())


def _client_key(request: Request) -> str:
    """Compute client key used by rate limiter."""
    return request.client.host if request.client else "unknown"


def _authorize(x_api_key: Annotated[str | None, Header()] = None) -> None:
    """Validate optional API key when configured."""
    if settings.app_api_key and x_api_key != settings.app_api_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error_code": ErrorCode.invalid_request.value,
                "message": "Unauthorized",
                "hint": "Provide a valid X-API-Key header",
            },
        )


def _enforce_rate_limit(request: Request) -> None:
    """Enforce per-client request rate limit."""
    key = _client_key(request)
    if not rate_limiter.allow(key):
        raise HTTPException(
            status_code=429,
            detail={
                "error_code": ErrorCode.invalid_request.value,
                "message": "Rate limit exceeded",
                "hint": "Reduce request rate or increase RATE_LIMIT_PER_MINUTE",
            },
        )


def _guard(request: Request, x_api_key: Annotated[str | None, Header()] = None) -> None:
    """Run auth and rate-limit checks for protected endpoints."""
    _authorize(x_api_key)
    _enforce_rate_limit(request)


def _with_request_context(request: Request) -> tuple[str, Token[str]]:
    """Attach request id into logging context and return id plus token."""
    request_id = _request_id(request)
    token = set_request_id(request_id)
    return request_id, token


@app.get("/health")
def health() -> dict[str, str]:
    """Return basic liveness status."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, object]:
    """Return readiness status including Neo4j dependency check."""
    try:
        neo4j_client.verify_connectivity()
        neo4j_ok = True
        neo4j_error = None
    except Exception as exc:  # noqa: BLE001
        neo4j_ok = False
        neo4j_error = str(exc)

    return {
        "status": "ok" if neo4j_ok else "degraded",
        "neo4j": {"ok": neo4j_ok, "error": neo4j_error},
        "gemini": {"key_file": settings.gemini_key_file},
    }


@app.get("/metrics")
def metrics() -> str:
    """Return Prometheus-style counters for ingestion jobs."""
    with _jobs_lock:
        counts: dict[str, int] = {}
        for job in _jobs.values():
            counts[job.status] = counts.get(job.status, 0) + 1

    lines = [
        "# HELP hf_jobs_total Number of HF ingestion jobs by status",
        "# TYPE hf_jobs_total gauge",
    ]
    for status, count in sorted(counts.items()):
        lines.append(f'hf_jobs_total{{status="{status}"}} {count}')
    if len(lines) == 2:
        lines.append('hf_jobs_total{status="none"} 0')

    return "\n".join(lines) + "\n"


@app.post("/ingest", dependencies=[Depends(_guard)])
def ingest(req: IngestRequest, request: Request) -> dict[str, object]:
    """Ingest one or more Wikipedia topics."""
    _request_id_value, token = _with_request_context(request)
    try:
        results: list[dict[str, object]] = []
        for topic in req.topics:
            try:
                result = ingest_topic(topic)
            except (ValueError, RuntimeError) as exc:
                logger.warning("Topic ingest failed", extra={"topic": topic})
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error_code": ErrorCode.ingest_failed.value,
                        "message": f"Failed ingest for '{topic}': {exc}",
                        "hint": "Verify topic exists and Gemini keys are valid",
                    },
                ) from exc
            results.append(_serialize_ingest_result(result))

        logger.info("Topic ingest completed", extra={"count": len(results)})
        return {"ingested": results}
    finally:
        reset_request_id(token)


@app.post("/query", dependencies=[Depends(_guard)])
def query(req: QueryRequest, request: Request) -> dict[str, object]:
    """Query graph and return deterministic answer plus citations."""
    _request_id_value, token = _with_request_context(request)
    started = time.perf_counter()
    try:
        result: QueryResult = query_graph(req.question, req.top_k)
    except RuntimeError as exc:
        logger.exception("Query failed")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.cypher_generation_failed.value,
                "message": f"Query failed: {exc}",
                "hint": "Review logs for fallback usage and Cypher validation",
            },
        ) from exc
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info("Query completed", extra={"duration_ms": elapsed_ms})
        reset_request_id(token)

    return {
        "answer": result.answer,
        "citations": result.citations,
        "strategy": result.strategy,
        "strategy_used": result.strategy_used,
        "fallback_reason": result.fallback_reason,
    }


@app.post("/query/explain", dependencies=[Depends(_guard)])
def query_explain(req: QueryRequest, request: Request) -> dict[str, object]:
    """Explain query strategy and model configuration without executing retrieval."""
    _request_id_value, token = _with_request_context(request)
    try:
        return {
            "question": req.question,
            "top_k": req.top_k,
            "strategy": {
                "primary": "generated_readonly_cypher",
                "fallback": "hybrid_fulltext",
            },
            "providers": {
                "orchestrator": settings.orchestrator_provider,
                "orchestrator_model": resolve_orchestrator_model(),
                "cypher": settings.cypher_provider,
                "cypher_model": resolve_cypher_model(),
            },
        }
    finally:
        reset_request_id(token)


@app.post("/ingest/hf", dependencies=[Depends(_guard)])
def ingest_hf(req: HFDatasetIngestRequest, request: Request) -> dict[str, object]:
    """Ingest a bounded sample directly from HF dataset (synchronous)."""
    _request_id_value, token = _with_request_context(request)
    try:
        results = ingest_from_hf(
            config_name=req.config_name,
            split=req.split,
            sample_size=req.sample_size,
            streaming=req.streaming,
        )
    except RuntimeError as exc:
        logger.warning("HF ingest failed")
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": ErrorCode.ingest_failed.value,
                "message": f"Failed HF ingestion: {exc}",
                "hint": "Check dataset config and Gemini key file",
            },
        ) from exc
    finally:
        reset_request_id(token)

    logger.info("HF ingest completed", extra={"count": len(results)})
    return {"ingested": [_serialize_ingest_result(r) for r in results]}


def _run_hf_ingest_job(job_id: str, req: HFIngestJobRequest) -> None:
    """Run one HF ingestion job in background worker thread."""
    stop_event = _job_stops[job_id]

    def _on_progress(processed: int, total: int | None, title: str) -> None:
        with _jobs_lock:
            job = _jobs[job_id]
            job.processed = processed
            job.total = total
            job.last_title = title
            _persist_job(job)

    try:
        results = ingest_from_hf(
            config_name=req.config_name,
            split=req.split,
            sample_size=req.sample_size,
            streaming=req.streaming,
            on_progress=_on_progress,
            should_stop=lambda: stop_event.is_set(),
        )
        with _jobs_lock:
            job = _jobs[job_id]
            job.ingested = [_serialize_ingest_result(r) for r in results]
            job.status = "cancelled" if stop_event.is_set() else "completed"
            logger.info("HF job finished", extra={"job_id": job_id, "status": job.status})
    except RuntimeError as exc:
        with _jobs_lock:
            job = _jobs[job_id]
            job.status = "failed"
            job.error = str(exc)
            logger.warning("HF job failed", extra={"job_id": job_id, "error": str(exc)})
    finally:
        with _jobs_lock:
            job = _jobs[job_id]
            if not job.finished_at:
                job.finished_at = datetime.now(timezone.utc).isoformat()
            _persist_job(job)
            _ = _job_stops.pop(job_id, None)


@app.post("/ingest/hf/jobs", dependencies=[Depends(_guard)])
def start_hf_ingest_job(req: HFIngestJobRequest) -> dict[str, object]:
    """Start asynchronous HF ingestion job and return job id."""
    job_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()

    with _jobs_lock:
        _jobs[job_id] = _JobState(
            job_id=job_id,
            status="running",
            config_name=req.config_name,
            split=req.split,
            sample_size=req.sample_size,
            streaming=req.streaming,
            started_at=started_at,
            total=req.sample_size if req.streaming else None,
        )
        _persist_job(_jobs[job_id])
        _job_stops[job_id] = threading.Event()

    thread = threading.Thread(target=_run_hf_ingest_job, args=(job_id, req), daemon=True)
    thread.start()

    logger.info("HF job started", extra={"job_id": job_id})
    return {"job_id": job_id, "status": "running", "started_at": started_at}


@app.get("/ingest/hf/jobs/{job_id}", dependencies=[Depends(_guard)])
def get_hf_ingest_job(job_id: str) -> dict[str, object]:
    """Get one HF ingestion job state by id."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": ErrorCode.invalid_request.value,
                    "message": f"Job not found: {job_id}",
                    "hint": "Verify job id or list jobs for valid ids",
                },
            )
        return job.model_dump()


@app.get("/ingest/hf/jobs", dependencies=[Depends(_guard)])
def list_hf_ingest_jobs(
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, object]:
    """List HF ingestion jobs with optional status filter and pagination."""
    with _jobs_lock:
        jobs = list(_jobs.values())

    if status:
        jobs = [j for j in jobs if j.status == status]

    jobs.sort(key=lambda j: j.started_at, reverse=True)
    selected = jobs[offset : offset + max(1, min(limit, 200))]

    return {
        "total": len(jobs),
        "limit": limit,
        "offset": offset,
        "items": [j.model_dump() for j in selected],
    }


@app.post("/ingest/hf/jobs/{job_id}/stop", dependencies=[Depends(_guard)])
def stop_hf_ingest_job(job_id: str) -> dict[str, object]:
    """Request cancellation for a running HF ingestion job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        stop_event = _job_stops.get(job_id)
        if not job or not stop_event:
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": ErrorCode.invalid_request.value,
                    "message": f"Job not found: {job_id}",
                    "hint": "Verify job id or list jobs for valid ids",
                },
            )
        stop_event.set()
        if job.status == "running":
            job.status = "cancelling"
        _persist_job(job)
        logger.info("HF job stop requested", extra={"job_id": job_id, "status": job.status})
        return {"job_id": job_id, "status": job.status}


@app.get("/export", dependencies=[Depends(_guard)], response_model=None)
def export_graph(request: Request, format: str = "jsonl") -> StreamingResponse | JSONResponse:
    """Export Page/Chunk/Entity nodes and relationships."""
    _request_id_value, token = _with_request_context(request)

    try:
        if format == "csv":
            return StreamingResponse(export_csv(), media_type="text/csv")
        if format == "jsonl":
            return StreamingResponse(export_jsonl(), media_type="application/x-ndjson")
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": ErrorCode.invalid_request.value,
                "message": f"Unsupported export format: {format}",
                "hint": "Use format=jsonl or format=csv",
            },
        )
    finally:
        reset_request_id(token)
