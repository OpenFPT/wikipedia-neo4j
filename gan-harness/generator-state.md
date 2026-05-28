# Generator State — Iteration 001

## What Was Built
- `src/dashboard/__init__.py` — Package init
- `src/dashboard/query_log.py` — QueryLogEntry dataclass + thread-safe ring buffer singleton (deque maxlen=50)
- `src/dashboard/data.py` — Data fetching: Neo4j graph stats, recent queries, signal breakdown, eval metrics, WRRF weights
- `src/dashboard/routes.py` — FastAPI router with `/dashboard` (HTML) and 4 JSON API endpoints
- `src/dashboard/templates/dashboard.html` — Full Jinja2 template with inline CSS and JS

## Features Implemented
- Graph Statistics Panel: 4 stat cards (Pages, Chunks, Entities, Relationships) + entity type breakdown
- Recent Queries Log: table with time, question (truncated 80 chars), tier, latency, result count
- WRRF Signal Breakdown: horizontal bar chart showing per-signal contribution from last query
- Evaluation Metrics: color-coded display (green/yellow/red thresholds) from data/eval_results.json
- WRRF Weight Display: visual bars showing configured retrieval weights
- Auto-refresh: 30s countdown with fetch-based DOM updates (no page reload)
- Responsive layout: 3 breakpoints (desktop 1440px, tablet 1024px, mobile 640px)
- Post-query hooks in /query and /query/hybrid that log to ring buffer
- Error resilience: Neo4j failures show "unavailable" state, never crash

## Design Choices
- Dark theme: bg #0f1419, cards #1a2332, accent teal #4ecdc4, coral #ff6b6b
- Inter + JetBrains Mono from Google Fonts CDN
- No gradients, no shadows, max 4px border-radius, 1px borders
- Large monospace numbers (2rem), small uppercase labels
- No external dependencies added

## Known Issues
- Signal scores in hybrid query hook are estimated from result dict keys (depends on hybrid_retrieve return format)
- Coverage threshold will fail (existing issue, not caused by dashboard)

## Dev Server
- URL: http://localhost:8000
- Status: not started (use `uv run uvicorn src.main:app --reload --port 8000`)
- Command: uv run uvicorn src.main:app --reload --port 8000
