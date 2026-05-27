"""FastAPI entrypoint with API guardrails and ingestion job orchestration."""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from contextvars import Token

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from src.config import settings, validate_runtime_settings
from src.ingest import IngestResult, ingest_from_hf, ingest_topic
from src.job_store import JobStore
from src.logging_utils import (
    configure_logging,
    get_logger,
    reset_request_id,
    set_request_id,
)
from src.neo4j_client import neo4j_client
from src.retrieve import hybrid_retrieve, query_graph


configure_logging(settings.log_level, json_logs=settings.json_logs)
logger = get_logger(__name__)


class _RateLimiter:
    """Simple in-memory fixed-window rate limiter per client key."""

    def __init__(self, max_requests: int, period_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.period_seconds = period_seconds
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = {}

    def allow(self, key: str) -> tuple[bool, int]:
        """Return (allowed, remaining) for given client key."""
        now = time.time()
        with self._lock:
            bucket = self._hits.setdefault(key, deque())
            cutoff = now - self.period_seconds
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self.max_requests:
                return False, 0
            bucket.append(now)
            if len(self._hits) > 100 and int(now) % 10 == 0:
                stale = [k for k, v in self._hits.items() if not v]
                for k in stale:
                    del self._hits[k]
            return True, self.max_requests - len(bucket)


rate_limiter = _RateLimiter(max_requests=settings.rate_limit_per_minute, period_seconds=60)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize runtime configuration and release resources on shutdown."""
    validate_runtime_settings()
    try:
        neo4j_client.setup_schema()
    except Exception:
        logger.warning("Schema setup failed on startup — will retry on first request")
    logger.info("Service starting")
    try:
        yield
    finally:
        neo4j_client.close()
        logger.info("Service shutdown complete")


app = FastAPI(title="Wikipedia Neo4j GraphRAG Demo", version="0.1.0", lifespan=lifespan)


# --- MCP Server Mount ---
from src.mcp_server import mcp as _mcp_instance

_mcp_app = _mcp_instance.http_app(path="/", transport="streamable-http")
app.mount("/mcp", _mcp_app)


# --- MCP Auth Middleware ---
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse as _StarletteJSONResponse


class _MCPAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/mcp") and settings.app_api_key:
            auth = request.headers.get("authorization", "")
            if not auth.startswith("Bearer ") or auth[7:].strip() != settings.app_api_key:
                return _StarletteJSONResponse(
                    {"error": "Unauthorized"}, status_code=401
                )
        return await call_next(request)


app.add_middleware(_MCPAuthMiddleware)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception", extra={"path": request.url.path})
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


class IngestRequest(BaseModel):
    """Request payload for Wikipedia topic ingestion."""

    topics: list[str] = Field(min_length=1, max_length=20, description="Wikipedia page topics")


class QueryRequest(BaseModel):
    """Request payload for query endpoint."""

    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=20)


class HFDatasetIngestRequest(BaseModel):
    """Request payload for direct HF dataset ingestion endpoint."""

    dataset_id: str = Field(default="Keithsel/viwiki-20260523", description="HuggingFace dataset ID")
    config_name: str = Field(default="cleaned", description="HF config, e.g. cleaned or raw")
    split: str = Field(default="train")
    sample_size: int = Field(default=5, ge=1, le=200)
    streaming: bool = Field(default=True, description="Use HF streaming mode for large configs")
    local_path: str | None = Field(default=None, description="Path to local Arrow dataset dir")


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
    ingested: list[dict] = Field(default_factory=list)


_jobs_lock = threading.Lock()
_jobs: dict[str, _JobState] = {}
_job_stops: dict[str, threading.Event] = {}
_job_store = JobStore(".hf_ingest_jobs.json")


def _serialize_ingest_result(result: IngestResult) -> dict:
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
    persisted = _job_store.load_all()
    now = datetime.now(timezone.utc)
    pruned = 0
    for job_id, payload in persisted.items():
        try:
            job = _JobState(**payload)
        except (TypeError, ValueError):
            logger.warning("Skipping invalid persisted job payload", extra={"job_id": job_id})
            continue
        if job.status in {"running", "cancelling"}:
            job.status = "interrupted"
            if not job.error:
                job.error = "Server restarted while job was in progress"
            if not job.finished_at:
                job.finished_at = now.isoformat()
        if job.finished_at and job.status in {"completed", "failed", "interrupted", "cancelled"}:
            try:
                finished = datetime.fromisoformat(job.finished_at)
                if (now - finished).total_seconds() > 86400:
                    pruned += 1
                    continue
            except (ValueError, TypeError):
                pass
        _jobs[job_id] = job
        _persist_job(job)
    if pruned:
        logger.info("Pruned old completed jobs on startup", extra={"pruned": pruned})


_restore_jobs()


def _request_id(request: Request) -> str:
    """Resolve request id from header or generate one."""
    return request.headers.get("X-Request-ID", str(uuid.uuid4()))


def _client_key(request: Request) -> str:
    """Compute client key used by rate limiter."""
    return request.client.host if request.client else "unknown"


def _authorize(x_api_key: str | None = Header(default=None)) -> None:
    """Validate optional API key when configured."""
    if settings.app_api_key and x_api_key != settings.app_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _enforce_rate_limit(request: Request) -> None:
    """Enforce per-client request rate limit."""
    key = _client_key(request)
    allowed, remaining = rate_limiter.allow(key)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": "60", "X-RateLimit-Remaining": "0"},
        )
    request.state.rate_limit_remaining = remaining


def _guard(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    """Run auth and rate-limit checks for protected endpoints."""
    _authorize(x_api_key)
    _enforce_rate_limit(request)


def _with_request_context(request: Request) -> tuple[str, Token[str]]:
    """Attach request id into logging context and return id plus token."""
    request_id = _request_id(request)
    token = set_request_id(request_id)
    return request_id, token


@app.get("/health")
def health() -> dict:
    """Return basic liveness status."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
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
def ingest(req: IngestRequest, request: Request) -> dict:
    """Ingest one or more Wikipedia topics."""
    _request_id_value, token = _with_request_context(request)
    try:
        results = []
        for topic in req.topics:
            try:
                result = ingest_topic(topic)
            except (ValueError, RuntimeError) as exc:
                logger.warning("Topic ingest failed", extra={"topic": topic})
                raise HTTPException(status_code=400, detail=f"Failed ingest for '{topic}': {exc}") from exc
            results.append(_serialize_ingest_result(result))

        logger.info("Topic ingest completed", extra={"count": len(results)})
        return {"ingested": results}
    finally:
        reset_request_id(token)


@app.post("/query", dependencies=[Depends(_guard)])
def query(req: QueryRequest, request: Request) -> dict:
    """Query graph and return deterministic answer plus citations."""
    _request_id_value, token = _with_request_context(request)
    started = time.perf_counter()
    try:
        result = query_graph(req.question, req.top_k)
    except RuntimeError as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info("Query completed", extra={"duration_ms": elapsed_ms})
        reset_request_id(token)

    return {
        "answer": result.answer,
        "citations": result.citations,
    }


@app.post("/query/hybrid", dependencies=[Depends(_guard)])
def query_hybrid(req: QueryRequest, request: Request) -> dict:
    """Hybrid retrieval combining BM25, vector, and graph channels via wRRF."""
    _request_id_value, token = _with_request_context(request)
    started = time.perf_counter()
    try:
        results = hybrid_retrieve(req.question, req.top_k)
    except RuntimeError as exc:
        logger.exception("Hybrid query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info("Hybrid query completed", extra={"duration_ms": elapsed_ms})
        reset_request_id(token)

    return {"results": results}


@app.post("/ingest/hf", dependencies=[Depends(_guard)])
def ingest_hf(req: HFDatasetIngestRequest, request: Request) -> dict:
    """Ingest a bounded sample directly from HF dataset (synchronous)."""
    _request_id_value, token = _with_request_context(request)
    try:
        results = ingest_from_hf(
            config_name=req.config_name,
            split=req.split,
            sample_size=req.sample_size,
            streaming=req.streaming,
            local_path=req.local_path,
            dataset_id=req.dataset_id,
        )
    except RuntimeError as exc:
        logger.warning("HF ingest failed")
        raise HTTPException(status_code=400, detail=f"Failed HF ingestion: {exc}") from exc
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
            local_path=req.local_path,
            dataset_id=req.dataset_id,
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
            _job_stops.pop(job_id, None)


@app.post("/ingest/hf/jobs", dependencies=[Depends(_guard)])
def start_hf_ingest_job(req: HFIngestJobRequest) -> dict:
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
def get_hf_ingest_job(job_id: str) -> dict:
    """Get one HF ingestion job state by id."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        return job.model_dump()


@app.get("/ingest/hf/jobs", dependencies=[Depends(_guard)])
def list_hf_ingest_jobs(status: str | None = None, limit: int = 50, offset: int = 0) -> dict:
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
def stop_hf_ingest_job(job_id: str) -> dict:
    """Request cancellation for a running HF ingestion job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        stop_event = _job_stops.get(job_id)
        if not job or not stop_event:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        stop_event.set()
        if job.status == "running":
            job.status = "cancelling"
        _persist_job(job)
        logger.info("HF job stop requested", extra={"job_id": job_id, "status": job.status})
        return {"job_id": job_id, "status": job.status}
