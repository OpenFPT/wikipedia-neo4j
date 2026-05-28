"""FastAPI router for the GraphPulse dashboard."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from jinja2 import Environment, FileSystemLoader

from src.dashboard.data import (
    fetch_eval_metrics,
    fetch_graph_stats,
    fetch_recent_queries,
    fetch_signal_breakdown,
    fetch_wrrf_weights,
)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

_template_dir = Path(__file__).parent / "templates"
_jinja_env = Environment(loader=FileSystemLoader(str(_template_dir)), autoescape=True)


@router.get("", response_class=HTMLResponse)
def dashboard_page(request: Request) -> HTMLResponse:
    """Render the main dashboard HTML page."""
    template = _jinja_env.get_template("dashboard.html")
    stats = fetch_graph_stats()
    queries = fetch_recent_queries()
    signals = fetch_signal_breakdown()
    eval_metrics = fetch_eval_metrics()
    weights = fetch_wrrf_weights()

    html = template.render(
        stats=stats,
        queries=queries,
        signals=signals,
        eval_metrics=eval_metrics,
        weights=weights,
    )
    return HTMLResponse(content=html)


@router.get("/api/stats", response_class=JSONResponse)
def api_stats() -> JSONResponse:
    """Return graph statistics as JSON."""
    return JSONResponse(content=fetch_graph_stats())


@router.get("/api/queries", response_class=JSONResponse)
def api_queries() -> JSONResponse:
    """Return recent queries as JSON."""
    return JSONResponse(content={"queries": fetch_recent_queries()})


@router.get("/api/signals", response_class=JSONResponse)
def api_signals() -> JSONResponse:
    """Return signal breakdown for the latest query."""
    data = fetch_signal_breakdown()
    return JSONResponse(content=data or {"scores": None})


@router.get("/api/eval", response_class=JSONResponse)
def api_eval() -> JSONResponse:
    """Return evaluation metrics as JSON."""
    data = fetch_eval_metrics()
    return JSONResponse(content=data or {"available": False})
