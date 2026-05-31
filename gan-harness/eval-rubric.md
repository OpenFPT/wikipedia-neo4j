# Evaluation Rubric: GraphPulse Dashboard

> For use by the Evaluator agent. Score each category 0-10, then apply weights.

## Scoring Formula

```
final_score = (design * 0.30) + (originality * 0.20) + (craft * 0.30) + (functionality * 0.20)
```

---

## 1. Design Quality (weight: 0.30)

| Score | Criteria |
|-------|----------|
| 9-10  | Dark theme is cohesive and restrained. Typography hierarchy (Inter + JetBrains Mono) is clear. Color palette (#4ecdc4 teal, #ff6b6b coral, #0f1419 bg) used purposefully. Grid alignment is precise. Responsive at 375px, 1024px, 1920px without breakage. |
| 7-8   | Theme is consistent but minor alignment issues. Colors mostly correct. Responsive works but has minor quirks at one breakpoint. |
| 5-6   | Recognizably dark theme but uses wrong colors or has gradient/shadow abuse. Typography is inconsistent. Responsive partially works. |
| 3-4   | Generic admin template feel. Light theme or wrong palette. Poor spacing. Breaks on mobile. |
| 0-2   | No coherent design. Unstyled HTML or completely off-brief. |

### Specific Checks
- [ ] Background is `#0f1419` or very close (not pure black, not gray)
- [ ] Card surfaces are `#1a2332` range (dark navy, not gray)
- [ ] Primary accent is teal (`#4ecdc4` range)
- [ ] No gradients on backgrounds or cards
- [ ] No box-shadows on cards (1px borders only)
- [ ] Border-radius max 6px (no pill shapes)
- [ ] Metric numbers use monospace font at large size (2rem+)
- [ ] Labels are small, uppercase, letter-spaced
- [ ] No emoji anywhere in the UI
- [ ] No stock illustrations or decorative SVG blobs
- [ ] Grid layout (not flexbox-only) for main structure
- [ ] Stacks to single column on mobile without horizontal scroll

---

## 2. Originality (weight: 0.20)

| Score | Criteria |
|-------|----------|
| 9-10  | Clearly a GraphRAG-specific dashboard. Signal breakdown visualization is novel and communicates WRRF fusion intuitively. Entity type display leverages the domain (Vietnamese Wikipedia). Custom SVG charts, not library-generated. |
| 7-8   | Domain-specific elements present. Charts are custom but straightforward. Some unique touches. |
| 5-6   | Could be any generic dashboard with labels changed. Charts use a library or are very basic. |
| 3-4   | Completely generic. No domain awareness. Could be a template from a CSS framework demo. |
| 0-2   | Copy-pasted template with no customization. |

### Specific Checks
- [ ] WRRF signal breakdown is visualized (not just numbers in a table)
- [ ] Entity types shown with domain context (Person/Org/Location/Work)
- [ ] SVG charts are hand-crafted (inline SVG, not canvas or library)
- [ ] Dashboard title/branding reflects the GraphRAG domain
- [ ] At least one visualization that would not make sense outside this project

---

## 3. Craft (weight: 0.30)

| Score | Criteria |
|-------|----------|
| 9-10  | Auto-refresh works without flicker (DOM patching, not innerHTML replacement). Empty states are designed (message + subtle icon or illustration). Numbers formatted with separators. Loading states exist. Keyboard shortcuts work. HTML is semantic (headings, landmarks, ARIA). Contrast ratios pass WCAG AA. |
| 7-8   | Auto-refresh works. Most empty states handled. Numbers formatted. Minor accessibility gaps. |
| 5-6   | Refresh causes visible flicker or full reload. Some empty states missing. Numbers unformatted. No keyboard support. |
| 3-4   | No auto-refresh. Crashes or shows errors on empty data. Raw numbers. No accessibility consideration. |
| 0-2   | Page does not load or has JavaScript errors. |

### Specific Checks
- [ ] `/dashboard` returns HTTP 200 with `text/html` content type
- [ ] Auto-refresh interval visible (countdown or indicator)
- [ ] Refresh updates DOM without full page reload (uses fetch + DOM manipulation)
- [ ] Numbers use thousand separators (e.g., "142,831" not "142831")
- [ ] Empty state for query log: shows message, not blank space
- [ ] Empty state for eval metrics: shows "No evaluation data" message
- [ ] Empty state for signal breakdown: shows "No retrieval data" message
- [ ] Neo4j connection failure: dashboard still renders with "unavailable" indicators
- [ ] `<html lang="vi">` or appropriate lang attribute
- [ ] Semantic headings (h1 for page title, h2 for sections)
- [ ] ARIA labels on interactive elements
- [ ] Color contrast ratio >= 4.5:1 for text
- [ ] Keyboard shortcut `R` triggers refresh
- [ ] Tab navigation reaches all interactive elements
- [ ] No console errors on page load

---

## 4. Functionality (weight: 0.20)

| Score | Criteria |
|-------|----------|
| 9-10  | All 4 panels render with real data. API endpoints (`/dashboard/api/stats`, `/dashboard/api/queries`, `/dashboard/api/signals`, `/dashboard/api/eval`) return valid JSON. Query log populates after making queries to `/query` or `/query/hybrid`. Dashboard loads in < 500ms. Playwright tests pass. |
| 7-8   | 3-4 panels work. API endpoints exist and return data. Minor issues with one panel. |
| 5-6   | 2-3 panels work. Some API endpoints missing or returning errors. Query log does not populate. |
| 3-4   | Only 1 panel works. Most endpoints broken. Page loads but shows mostly static/fake data. |
| 0-2   | Page does not load or returns 500. No working data integration. |

### Specific Checks
- [ ] `GET /dashboard` returns 200 with HTML
- [ ] `GET /dashboard/api/stats` returns JSON with page/chunk/entity/rel counts
- [ ] `GET /dashboard/api/queries` returns JSON array of recent queries
- [ ] `GET /dashboard/api/signals` returns JSON with signal contribution data
- [ ] `GET /dashboard/api/eval` returns JSON with eval metrics (or 404/empty if no file)
- [ ] Graph stats reflect actual Neo4j data (not hardcoded)
- [ ] Query log entry appears after calling `POST /query`
- [ ] Signal breakdown updates after a hybrid query
- [ ] Dashboard handles Neo4j being unavailable (no 500 error)
- [ ] Page load time < 500ms (measured server-side)
- [ ] No new pip dependencies required (uses existing fastapi, jinja2, neo4j)

---

## Test Scenarios for Playwright

```python
# test_dashboard_e2e.py expectations

def test_dashboard_loads():
    """GET /dashboard returns 200, contains 'GraphPulse' title."""

def test_stats_panel_visible():
    """Stats cards are visible with numeric content."""

def test_query_log_empty_state():
    """Before any queries, shows empty state message."""

def test_query_log_populates():
    """After POST /query, refresh shows the query in the log."""

def test_responsive_mobile():
    """At 375px width, no horizontal scrollbar, cards stack vertically."""

def test_auto_refresh():
    """After 30s (or manual trigger), stats update without navigation."""

def test_api_stats_endpoint():
    """GET /dashboard/api/stats returns JSON with expected keys."""

def test_api_eval_no_file():
    """GET /dashboard/api/eval returns empty/null when no eval file exists."""
```

---

## Disqualifying Issues (automatic score cap at 4/10)

- Dashboard returns 500 error on load
- Uses React, Vue, Angular, or any JS framework
- Requires `npm install` or additional system dependencies
- Uses a CSS framework loaded as a full bundle (Bootstrap, Tailwind CDN with full CSS)
- Has hardcoded/fake data with no Neo4j integration
- Light theme with no dark mode (spec requires dark-first)
- Breaks the existing `/query`, `/health`, or `/metrics` endpoints
