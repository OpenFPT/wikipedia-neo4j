# Evaluation — Iteration 001

## Scores

| Criterion | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Design Quality | 7/10 | 0.30 | 2.10 |
| Originality | 6/10 | 0.20 | 1.20 |
| Craft | 5/10 | 0.30 | 1.50 |
| Functionality | 6/10 | 0.20 | 1.20 |
| **TOTAL** | | | **6.0/10** |

## Verdict: FAIL (threshold: 7.0)

---

## Critical Issues (must fix)

1. **No keyboard shortcut support**: The spec requires `R` key to trigger refresh and `?` for shortcuts overlay. The JS has zero `keydown`/`keypress` listeners. Add a document-level keydown handler: `document.addEventListener('keydown', function(e) { if (e.key === 'r' || e.key === 'R') refresh(); })`.

2. **No ARIA attributes anywhere**: Zero `aria-label`, `role`, or `aria-live` attributes in the entire HTML. The stats grid should have `role="region" aria-label="Graph Statistics"`. The refresh countdown should have `aria-live="polite"`. The query table needs `role="table"` when dynamically injected.

3. **No semantic headings beyond h1**: Only one `<h1>GraphPulse</h1>` exists. Section titles use `<div class="section-title">` instead of `<h2>`. This breaks screen reader navigation and violates the rubric requirement for semantic headings. Replace all `.section-title` divs with `<h2 class="section-title">`.

4. **Stats panel shows single "unavailable" card instead of 4 stat cards with zero values**: When Neo4j is down, the spec says "Shows 0 gracefully if DB is empty." The current implementation collapses to a single card saying "Neo4j unavailable." It should still render all 4 stat cards (Pages, Chunks, Entities, Relationships) with a dash or "—" and a small unavailable indicator, preserving layout stability.

## Major Issues (should fix)

1. **No `data-stat` attributes in server-rendered HTML when Neo4j is unavailable**: The `updateStats()` JS function queries `[data-stat]` elements, but when Neo4j is down, the server renders a single unavailable card without those attributes. The auto-refresh will never be able to populate stats even if Neo4j comes back online. Always render the 4 stat cards with `data-stat` attributes, showing "—" when unavailable.

2. **Responsive breakpoint at 640px instead of 375px**: The spec and rubric require testing at 375px. The CSS media query uses `max-width: 640px` for mobile. While this covers 375px, the rubric specifically mentions 375px testing. The stat-value drops to 1.75rem at mobile which is fine, but the header stacking could be tighter.

3. **No number formatting visible in server-rendered HTML**: The `formatNumber()` JS function exists for client-side refresh, but the initial server-rendered stat values (when Neo4j is available) should also be pre-formatted with thousand separators. Currently untestable since Neo4j is down, but the template should use a Jinja2 filter like `{{ value | format_number }}`.

4. **Signal breakdown uses innerHTML replacement**: The `updateSignals()`, `updateQueries()`, and `updateEval()` functions all use `panel.innerHTML = html`. This causes DOM flicker on refresh. Use DOM diffing or at minimum, only replace if content actually changed (compare innerHTML before setting).

5. **No loading state on initial fetch or refresh**: When auto-refresh fires, there is no visual indicator that data is being fetched. Add a subtle pulse or opacity change to panels during fetch.

## Minor Issues (nice to fix)

1. **`--radius: 4px` but rubric allows up to 6px**: This is fine and within spec. The `border-radius: 50%` on `.status-dot` is acceptable for a circular indicator.

2. **No `<main>` landmark element**: The `.container` div should be a `<main>` element for accessibility.

3. **Color contrast on muted text**: `#8899a6` on `#0f1419` background yields approximately 4.8:1 contrast ratio — passes AA but barely. On `#1a2332` card backgrounds it drops to approximately 3.8:1 which fails WCAG AA for normal text. Lighten `--muted` to `#9aabb8` or similar.

4. **No `</html>` closing tag visible**: The HTML appears to end after `</script></body>` without `</html>`. Add it.

5. **Weight bar percentages are hardcoded relative**: BM25 and Vector both show `width: 35%` for weight 0.40, but the max weight is 0.40, so they should be 100% relative to max. The current display works visually but is not proportionally accurate. Consider normalizing to max weight = 100%.

6. **XSS protection in query table**: The JS does `question.replace(/</g, '&lt;').replace(/>/g, '&gt;')` which is minimal. It does not escape quotes or ampersands. Use a proper escape function or use `textContent` instead of innerHTML for user-provided data.

---

## Rubric Checklist Results

### Design Quality
- [x] Background is `#0f1419`
- [x] Card surfaces are `#1a2332`
- [x] Primary accent is teal `#4ecdc4`
- [x] No gradients on backgrounds or cards
- [x] No box-shadows on cards (1px borders only)
- [x] Border-radius max 6px (uses 4px via `--radius`)
- [x] Metric numbers use monospace font at large size (2rem)
- [x] Labels are small, uppercase, letter-spaced
- [x] No emoji anywhere in the UI
- [x] No stock illustrations or decorative SVG blobs
- [x] Grid layout for main structure (CSS Grid used)
- [ ] Stacks to single column on mobile — partially (640px breakpoint, not verified at 375px visually)

### Originality
- [ ] WRRF signal breakdown is visualized — present but only as horizontal bars (no SVG chart)
- [x] Entity types shown with domain context (Person/Org/Location/Work)
- [ ] SVG charts are hand-crafted — NO SVG charts at all, only CSS bars
- [x] Dashboard title/branding reflects GraphRAG domain
- [x] WRRF weight display is domain-specific

### Craft
- [x] `/dashboard` returns HTTP 200 with `text/html`
- [x] Auto-refresh interval visible (countdown)
- [x] Refresh updates DOM without full page reload (fetch + DOM manipulation)
- [ ] Numbers use thousand separators — only in JS refresh, not server-rendered
- [x] Empty state for query log: "No queries recorded yet"
- [x] Empty state for eval metrics: "No evaluation data"
- [x] Empty state for signal breakdown: "No retrieval data"
- [x] Neo4j connection failure: dashboard renders with "unavailable" indicators
- [x] `<html lang="vi">`
- [ ] Semantic headings (h1 for page title, h2 for sections) — MISSING h2
- [ ] ARIA labels on interactive elements — MISSING entirely
- [ ] Color contrast ratio >= 4.5:1 for all text — FAILS on muted text in cards
- [ ] Keyboard shortcut `R` triggers refresh — NOT IMPLEMENTED
- [ ] Tab navigation reaches all interactive elements — no focusable elements beyond default
- [x] No console errors on page load (no JS errors in source)

### Functionality
- [x] `GET /dashboard` returns 200 with HTML
- [x] `GET /dashboard/api/stats` returns JSON with expected keys
- [x] `GET /dashboard/api/queries` returns JSON array
- [x] `GET /dashboard/api/signals` returns JSON with signal data (null when empty)
- [x] `GET /dashboard/api/eval` returns JSON (available: false when no file)
- [ ] Graph stats reflect actual Neo4j data — Neo4j unavailable, shows 0s correctly
- [ ] Query log entry appears after calling POST /query — untested (no /query endpoint working without Neo4j)
- [ ] Signal breakdown updates after hybrid query — untested
- [x] Dashboard handles Neo4j being unavailable (no 500 error)
- [x] Page load time < 500ms (measured 18ms)
- [x] No new pip dependencies required

---

## What Improved Since Last Iteration
- N/A (first iteration)

## What Regressed Since Last Iteration
- N/A (first iteration)

## Specific Suggestions for Next Iteration

1. **Add semantic HTML**: Replace all `<div class="section-title">` with `<h2 class="section-title">`. Wrap the main content in `<main>`. Add `<section>` elements around each panel group.

2. **Add ARIA attributes**: `aria-live="polite"` on `#stats-grid`, `#queries-panel`, `#signal-panel`, `#eval-panel`. Add `aria-label` to the header status region. Add `role="status"` to the countdown.

3. **Implement keyboard shortcuts**: Add a document keydown listener for `R` (refresh) and `?` (show help). The help overlay can be a simple fixed-position div toggled by the `?` key.

4. **Render all 4 stat cards even when Neo4j is unavailable**: Show the card structure with "—" values and a small "unavailable" badge. This maintains layout consistency and allows auto-refresh to populate values when Neo4j reconnects.

5. **Add at least one inline SVG visualization**: The signal breakdown or entity type distribution should use an actual `<svg>` element (e.g., horizontal stacked bar as SVG, or a simple donut chart for entity types). CSS-only bars are functional but the rubric specifically checks for "hand-crafted SVG charts."

6. **Fix muted text contrast**: Change `--muted` from `#8899a6` to `#9aacb8` to ensure 4.5:1 contrast on card backgrounds.

7. **Avoid innerHTML for user data**: In `updateQueries()`, build DOM nodes with `document.createElement` and `textContent` for the question field, or use a comprehensive escape function.

---

## Screenshots
- No visual screenshots taken (Playwright MCP not available). Evaluation based on HTML source analysis, API response testing, and CSS inspection via curl.

---

**WEIGHTED TOTAL: 6.0 / 10**
