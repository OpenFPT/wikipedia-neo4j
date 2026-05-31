    # Product Specification: GraphPulse

> Generated from brief: "Build a dashboard page — A /dashboard route showing graph stats, recent queries, eval metrics, retrieval signal breakdown. Visual, iteratable, testable."

## Vision

GraphPulse is an operational dashboard for the Vietnamese Wikipedia GraphRAG system. It gives operators a single-screen view of knowledge graph health, query activity, retrieval signal performance, and evaluation quality — all rendered as server-side HTML with lightweight client-side interactivity. It should feel like a purpose-built control panel, not a generic admin template.

## Design Direction

- **Color palette**: Background `#0f1419` (near-black), card surfaces `#1a2332` (dark navy), primary accent `#4ecdc4` (teal), secondary accent `#ff6b6b` (coral), text `#e8eaed` (light gray), muted text `#8899a6`, success `#00c853`, warning `#ffd600`, border `#2d3f50`
- **Typography**: `Inter` via Google Fonts CDN for UI text, `JetBrains Mono` for numbers/metrics/code. Hierarchy: metric values 2.5rem bold, card titles 0.875rem uppercase tracking-wide, body 0.9375rem
- **Layout philosophy**: Dense dashboard grid — 12-column CSS Grid on desktop, collapsing to single column on mobile. No wasted whitespace. Cards have subtle 1px borders, no drop shadows. Information density over decoration.
- **Visual identity**: Inspired by Grafana's dark mode and Linear's precision. Metric cards use large monospace numbers with small colored trend indicators. Charts use SVG with the teal/coral palette. No gradients, no rounded-everything, no stock illustrations.
- **Inspiration**: Grafana dashboards, Vercel analytics, Linear's UI density, Railway's dark theme
- **Anti-AI-slop directives**: No gradient backgrounds, no glassmorphism, no excessive border-radius (max 6px), no decorative SVG blobs, no "hero" sections, no card shadows, no emoji in the UI, no generic placeholder illustrations

## Features (prioritized)

### Must-Have (Sprint 1)

1. **Graph Statistics Panel**: Display live counts from Neo4j — Pages, Chunks, Entities (by type: Person/Org/Location/Work), Relationships (HAS_CHUNK, MENTIONS, LINKS_TO). Each stat is a card with the count in large monospace font and a label below. Data fetched via Cypher `MATCH (n:Label) RETURN count(n)`.
   - Acceptance: Panel renders 4+ stat cards with real Neo4j data. Shows 0 gracefully if DB is empty.

2. **Recent Queries Log**: A table showing the last 20 queries processed by the `/query` and `/query/hybrid` endpoints. Columns: timestamp, question (truncated to 80 chars), retrieval tier, latency (ms), result count. Stored in an in-memory ring buffer (deque) in the app module.
   - Acceptance: Table populates after queries are made. Empty state shows "No queries recorded yet" message. Rows are newest-first.

3. **WRRF Signal Breakdown**: A horizontal stacked bar or grouped bar chart (inline SVG) showing the contribution of each retrieval signal (BM25, Vector, Graph, Community) for the most recent query. Also shows the configured weights from settings as a reference row.
   - Acceptance: Chart renders with correct proportions. Labeled axes. Falls back to "No retrieval data" when no queries have been made.

4. **Evaluation Metrics Panel**: Display the latest evaluation run results — context hit rate, MRR, rerank hit rate, rerank MRR, average latency. Read from `data/eval_results.json` if it exists (written by eval scripts). Show "No evaluation data" if file is missing.
   - Acceptance: Metrics display with 2-decimal precision. Color-coded (green > 0.7, yellow 0.4-0.7, red < 0.4).

5. **Auto-refresh**: Page auto-refreshes stats every 30 seconds via a small vanilla JS snippet using `fetch()` to a JSON API endpoint, updating DOM without full page reload.
   - Acceptance: Stats update without page flicker. Refresh interval visible as a subtle countdown indicator.

6. **Responsive Layout**: Dashboard is usable on 1920px desktop, 1024px tablet, and 375px mobile. Grid collapses gracefully.
   - Acceptance: No horizontal scroll at any breakpoint. Cards stack vertically on mobile.

### Should-Have (Sprint 2)

7. **Query Latency Sparkline**: A small inline SVG sparkline (last 20 queries) showing latency trend over time. Rendered server-side as an SVG path.
   - Acceptance: Sparkline renders with correct data points. Handles < 3 data points gracefully.

8. **System Health Indicators**: Show Neo4j connection status (connected/degraded), embedding backend status, model mode (api/local), NER backend. Small colored dots (green/yellow/red) next to each.
   - Acceptance: Reflects actual runtime state. Neo4j dot turns red if connection fails.

9. **Entity Type Distribution**: A donut chart (SVG) showing the breakdown of entity types (Person, Organization, Location, Work, untyped Entity).
   - Acceptance: Chart segments are proportional. Legend shows counts. Handles zero entities.

10. **Retrieval Weight Configuration Display**: Show current WRRF weights as a visual bar with labeled segments. Read-only display of `settings.wrrf_weight_*` values.
    - Acceptance: Weights sum displayed. Each segment labeled with weight name and value.

11. **Ingestion Job Status**: Show active/recent background HF ingestion jobs with progress bars. Reuses existing `_jobs` data from main.py.
    - Acceptance: Running jobs show animated progress. Completed/failed jobs show final status.

12. **Keyboard Navigation**: Tab-navigable cards, `R` key to force refresh, `?` to show keyboard shortcuts overlay.
    - Acceptance: All interactive elements reachable via keyboard. Focus indicators visible.

### Nice-to-Have (Sprint 3)

13. **Dark/Light Toggle**: CSS custom properties allow switching between dark (default) and light theme. Preference stored in localStorage.
    - Acceptance: Toggle works without page reload. Persists across sessions.

14. **Export Metrics as JSON**: A small "Export" button that downloads current dashboard state as a JSON file.
    - Acceptance: Downloaded file contains all visible metrics with timestamp.

15. **Query Detail Drawer**: Clicking a query row expands an inline detail view showing full question, all retrieved chunks, and signal scores.
    - Acceptance: Drawer opens/closes smoothly. Shows chunk text truncated to 200 chars.

16. **Historical Eval Comparison**: If multiple eval result files exist, show a trend line of hit rate and MRR over time.
    - Acceptance: Chart renders with 2+ data points. Single point shows just the dot.

## Technical Stack

- **Frontend**: Server-rendered HTML via FastAPI `HTMLResponse` + Jinja2 templates. Inline `<style>` block (no external CSS framework). Google Fonts CDN for Inter + JetBrains Mono. Vanilla JS for interactivity (fetch-based refresh, keyboard shortcuts).
- **Backend**: FastAPI route at `/dashboard` returning HTML. JSON API endpoints at `/dashboard/api/stats`, `/dashboard/api/queries`, `/dashboard/api/signals`, `/dashboard/api/eval` for AJAX refresh.
- **Data layer**: Direct Neo4j Cypher queries for graph stats. In-memory `deque(maxlen=50)` for query log (populated via middleware or post-query hook). File-based eval results from `data/eval_results.json`.
- **Key libraries**: No new dependencies required. Uses existing `neo4j`, `fastapi`, `jinja2` (already a FastAPI dependency). Optional: `markupsafe` for template escaping (comes with Jinja2).
- **Testing**: Playwright for E2E tests (page loads, elements present, responsive behavior). pytest for API endpoint unit tests.

## Data Sources

### Graph Statistics
```cypher
MATCH (p:Page) RETURN count(p) AS pages
MATCH (c:Chunk) RETURN count(c) AS chunks
MATCH (e:Entity) RETURN count(e) AS entities
MATCH ()-[r:HAS_CHUNK]->() RETURN count(r) AS has_chunk_rels
MATCH ()-[r:MENTIONS]->() RETURN count(r) AS mention_rels
MATCH ()-[r:LINKS_TO]->() RETURN count(r) AS links_to_rels
MATCH (p:Person) RETURN count(p) AS persons
MATCH (o:Organization) RETURN count(o) AS orgs
MATCH (l:Location) RETURN count(l) AS locations
MATCH (w:Work) RETURN count(w) AS works
```

### Recent Queries
In-memory ring buffer populated by a post-query hook in the `/query` and `/query/hybrid` handlers. Each entry:
```python
@dataclass
class QueryLogEntry:
    timestamp: str        # ISO format
    question: str         # full question text
    retrieval_tier: str   # "hybrid", "cypher", "fallback"
    latency_ms: int       # elapsed time
    result_count: int     # number of results returned
    signal_scores: dict   # {"bm25": 5, "vector": 3, "graph": 2, "community": 1}
```

### Evaluation Metrics
Read from `data/eval_results.json`:
```json
{
  "timestamp": "2026-05-29T10:00:00Z",
  "total": 100,
  "context_hit_rate": 0.726,
  "mrr": 0.583,
  "rerank_context_hit_rate": 0.81,
  "rerank_mrr": 0.67,
  "avg_latency_ms": 342
}
```

## Layout (ASCII Mockup)

```
+------------------------------------------------------------------+
|  GraphPulse                              [Neo4j: *] [Refresh: 28s]|
+------------------------------------------------------------------+
|                                                                    |
|  GRAPH STATISTICS                                                  |
|  +----------+ +----------+ +----------+ +----------+              |
|  | 142,831  | |  891,204 | |  67,432  | |  234,102 |              |
|  |  Pages   | |  Chunks  | | Entities | |   Rels   |              |
|  +----------+ +----------+ +----------+ +----------+              |
|                                                                    |
|  ENTITY TYPES                    RETRIEVAL WEIGHTS (WRRF)          |
|  +-------------------------+     +-----------------------------+   |
|  |  Person: 23,401         |     | BM25   [========    ] 0.40 |   |
|  |  Org:     8,912         |     | Vector [========    ] 0.40 |   |
|  |  Location: 31,204       |     | Graph  [====        ] 0.20 |   |
|  |  Work:     3,915        |     | Comm.  [===         ] 0.15 |   |
|  +-------------------------+     +-----------------------------+   |
|                                                                    |
|  RECENT QUERIES                              Latency Sparkline     |
|  +----------------------------------------------------------+     |
|  | Time     | Question              | Tier   | ms  | Results|     |
|  |----------|---------------------- |--------|-----|--------|     |
|  | 14:32:01 | Ai la tong thong d... | hybrid | 234 |    4   |     |
|  | 14:31:45 | Thu do cua Viet Na... | hybrid | 189 |    4   |     |
|  | 14:30:12 | Ho Chi Minh sinh n... | fallbk | 412 |    3   |     |
|  +----------------------------------------------------------+     |
|                                                                    |
|  SIGNAL BREAKDOWN (Last Query)       EVALUATION METRICS            |
|  +---------------------------+       +-------------------------+   |
|  | BM25:  [=======   ] 5    |       | Hit Rate:    0.73  [G]  |   |
|  | Vector:[=====     ] 3    |       | MRR:         0.58  [Y]  |   |
|  | Graph: [===       ] 2    |       | Rerank HR:   0.81  [G]  |   |
|  | Comm.: [=         ] 1    |       | Rerank MRR:  0.67  [Y]  |   |
|  +---------------------------+       | Avg Latency: 342ms     |   |
|                                      +-------------------------+   |
|                                                                    |
|  INGESTION JOBS                                                    |
|  +----------------------------------------------------------+     |
|  | job-abc123 | running  | [=====>     ] 45/100 | viwiki    |     |
|  | job-def456 | completed| [==========] 100/100 | viwiki    |     |
|  +----------------------------------------------------------+     |
+------------------------------------------------------------------+
```

## File Structure

```
src/
  dashboard/
    __init__.py
    routes.py          # FastAPI router with /dashboard and /dashboard/api/* endpoints
    data.py            # Data fetching functions (Neo4j queries, query log, eval file)
    query_log.py       # QueryLogEntry dataclass + ring buffer singleton
    templates/
      dashboard.html   # Jinja2 template with inline CSS and JS
tests/
  test_dashboard.py    # Unit tests for data functions and API endpoints
  test_dashboard_e2e.py  # Playwright E2E tests
```

## Implementation Notes

1. **Query log integration**: Add a post-query hook in `src/main.py` that appends to the ring buffer after each `/query` and `/query/hybrid` call. The hook captures timing, tier, and signal contribution data.

2. **Signal breakdown data**: The `hybrid_retrieve` function should be modified to return signal contribution metadata (how many results came from each channel before fusion). This can be a lightweight addition to the return value or a separate instrumentation layer.

3. **Template approach**: Single Jinja2 template file with all CSS inlined in a `<style>` tag and all JS in a `<script>` tag at the bottom. No external files to serve (except Google Fonts CDN). This keeps deployment simple and avoids static file serving configuration.

4. **Error resilience**: Every data-fetching function must handle Neo4j connection failures gracefully. The dashboard should render with "unavailable" indicators rather than crashing.

5. **No authentication on dashboard**: The `/dashboard` route is public (read-only operational view). The `/dashboard/api/*` endpoints are also public. This matches the pattern of `/health` and `/metrics`.

## Evaluation Criteria

### Design Quality (weight: 0.3)
- Dark theme executed with restraint — no gradients, no glow effects
- Typography hierarchy is clear: large monospace numbers, small uppercase labels
- Color usage is purposeful: teal for primary data, coral for alerts/warnings, not decorative
- Grid alignment is pixel-precise — no misaligned cards or uneven spacing
- Responsive breakpoints work without layout breakage

### Originality (weight: 0.2)
- Feels like a purpose-built GraphRAG dashboard, not a generic admin template
- SVG charts are custom-drawn, not library-generated
- The signal breakdown visualization communicates WRRF fusion clearly
- Information density is high without feeling cluttered

### Craft (weight: 0.3)
- Auto-refresh works smoothly without flicker
- Empty states are handled gracefully (not blank, not error)
- Numbers are formatted with locale-appropriate separators (e.g., 142,831)
- Loading states exist for async data fetches
- Keyboard shortcuts work as documented
- HTML is semantic and accessible (proper headings, ARIA labels, contrast ratios)

### Functionality (weight: 0.2)
- All 4 data panels render with real data from Neo4j/files
- Auto-refresh updates without page reload
- API endpoints return correct JSON
- Dashboard loads in < 500ms (server-side render)
- Playwright tests pass: page loads, stats visible, table populated after query

## Sprint Plan

### Sprint 1: Core Dashboard
- Goals: Functional dashboard with all 4 main panels rendering real data
- Features: #1 (Graph Stats), #2 (Recent Queries), #3 (Signal Breakdown), #4 (Eval Metrics), #5 (Auto-refresh), #6 (Responsive)
- Definition of done: `/dashboard` renders with live Neo4j data, query log populates after API calls, eval metrics display from file, auto-refresh works, responsive at 3 breakpoints, unit tests pass

### Sprint 2: Polish and Depth
- Features: #7 (Sparkline), #8 (Health Indicators), #9 (Entity Donut), #10 (Weight Display), #11 (Job Status), #12 (Keyboard Nav)
- Definition of done: All visualizations render correctly, health indicators reflect real state, keyboard navigation complete, Playwright E2E tests pass

### Sprint 3: Extras
- Features: #13 (Theme Toggle), #14 (Export), #15 (Query Drawer), #16 (Historical Eval)
- Definition of done: Theme persists, export downloads valid JSON, drawer animates smoothly, historical chart renders with multiple data points
