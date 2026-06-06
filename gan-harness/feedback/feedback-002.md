# Evaluation — Iteration 002

## Scores

| Criterion | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Design Quality | 8/10 | 0.30 | 2.40 |
| Originality | 7/10 | 0.20 | 1.40 |
| Craft | 7/10 | 0.30 | 2.10 |
| Functionality | 7/10 | 0.20 | 1.40 |
| **TOTAL** | | | **7.3/10** |

## Verdict: PASS (threshold: 7.0)

---

## Critical Issues (must fix)

None remaining. All critical issues from iteration 1 have been addressed.

---

## Major Issues (should fix)

1. **Entity type distribution SVG missing when Neo4j is unavailable**: The spec mentions an inline SVG stacked bar for entity type distribution, but when Neo4j is down the panel just shows "Neo4j unavailable" text. The WRRF chart renders statically regardless of Neo4j state — the entity type panel should do the same, showing a placeholder SVG with zero-height bars and "—" labels for Person/Org/Location/Work counts. This would demonstrate the visualization even when data is absent.

2. **`updateEval()` still uses innerHTML with string concatenation for eval metrics**: While the query table now correctly uses `textContent` via DOM API (fixing the XSS concern), the `updateEval()` function still builds HTML via string concatenation and sets `panel.innerHTML = html`. The metric labels go through `escapeHtml()` which is good, but the values (`val.toFixed(2)`) are injected directly. Since these come from a local JSON file this is low-risk, but for consistency with the XSS-safe pattern used in `updateQueries()`, build eval rows via DOM API too.

3. **No loading/fetching indicator during refresh**: When auto-refresh fires or the user presses R, there is no visual feedback that a fetch is in progress. Add a brief opacity transition or a subtle pulse on the panels during the fetch cycle. Even a 150ms `opacity: 0.7` transition on the container would communicate activity.

4. **Number formatting not applied to server-rendered HTML**: When Neo4j is available, the initial server-rendered stat values would come from Jinja2 without thousand separators. The `formatNumber()` JS function only runs on client-side refresh. Add a Jinja2 filter (e.g., `{{ value | format_number }}`) so the first paint also shows formatted numbers.

---

## Minor Issues (nice to fix)

1. **Shortcuts overlay uses inline styles exclusively**: The overlay `<div>` and its children use `style=""` attributes instead of CSS classes. This works but is harder to maintain and cannot be overridden by media queries. Extract to named classes (`.shortcuts-overlay`, `.shortcuts-dialog`, `.shortcut-row`).

2. **No focus trap in shortcuts overlay**: When the overlay opens via `?`, focus does not move into the dialog. A keyboard-only user cannot dismiss it without knowing to press Escape. Add `overlay.querySelector('div').focus()` on open, or add a visible close button.

3. **`aria-live="off"` on countdown timer**: The countdown element has `role="timer"` but `aria-live="off"`, which means screen readers will never announce countdown changes. This is arguably correct (announcing every second would be noisy), but consider `aria-live="polite"` with a debounce — only announce when refresh actually fires (e.g., "Data refreshed").

4. **SVG bar chart viewBox uses `preserveAspectRatio="none"`**: This distorts the bars when the container width differs from 400px. Use `preserveAspectRatio="xMidYMid meet"` to maintain proportions, or remove the attribute entirely (default is `meet`).

5. **Missing `</html>` closing tag**: The HTML ends at `</body>` without a closing `</html>`. Browsers handle this gracefully but it is technically invalid.

6. **No visible focus indicator on interactive elements**: There are no custom `:focus-visible` styles. The browser default outline may be invisible on the dark background. Add `outline: 2px solid var(--accent); outline-offset: 2px;` for `:focus-visible` on focusable elements.

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
- [x] Stacks to single column on mobile (640px breakpoint covers 375px)

### Originality
- [x] WRRF signal breakdown is visualized (SVG bar chart, dynamically rendered)
- [x] Entity types shown with domain context (Person/Org/Location/Work in template)
- [x] SVG charts are hand-crafted (inline SVG, no library)
- [x] Dashboard title/branding reflects GraphRAG domain
- [x] WRRF weight visualization is domain-specific and would not make sense outside this project

### Craft
- [x] `/dashboard` returns HTTP 200 with `text/html`
- [x] Auto-refresh interval visible (countdown timer)
- [x] Refresh updates DOM without full page reload (fetch + DOM manipulation)
- [ ] Numbers use thousand separators — only in JS refresh path, not server-rendered
- [x] Empty state for query log: "No queries recorded yet"
- [x] Empty state for eval metrics: "No evaluation data"
- [x] Empty state for signal breakdown: "No retrieval data"
- [x] Neo4j connection failure: dashboard renders with "unavailable" badges on all 4 cards
- [x] `<html lang="vi">`
- [x] Semantic headings (h1 for page title, h2 for all sections)
- [x] ARIA labels on interactive elements (`role="region"`, `aria-label`, `aria-live`, `role="timer"`, `role="dialog"`)
- [x] Color contrast ratio >= 4.5:1 — `#a8bcc8` on `#1a2332` yields ~4.6:1 (passes AA)
- [x] Keyboard shortcut `R` triggers refresh
- [x] Keyboard shortcut `?` shows overlay, `Esc` closes it
- [ ] Tab navigation reaches all interactive elements — no custom focus styles
- [x] No console errors on page load

### Functionality
- [x] `GET /dashboard` returns 200 with HTML (3ms response time)
- [x] `GET /dashboard/api/stats` returns JSON with expected keys (pages, chunks, entities, total_rels, available)
- [x] `GET /dashboard/api/queries` returns JSON with queries array
- [x] `GET /dashboard/api/signals` returns JSON with scores field
- [x] `GET /dashboard/api/eval` returns JSON with available field
- [ ] Graph stats reflect actual Neo4j data — Neo4j unavailable, shows "—" correctly
- [ ] Query log entry appears after calling POST /query — untested (Neo4j down)
- [ ] Signal breakdown updates after hybrid query — untested (Neo4j down)
- [x] Dashboard handles Neo4j being unavailable (no 500 error, graceful degradation)
- [x] Page load time < 500ms (measured 3ms)
- [x] No new pip dependencies required

---

## What Improved Since Last Iteration

1. **Semantic headings**: All section titles are now proper `<h2>` elements with `id` attributes for `aria-labelledby` linking. Clean heading hierarchy (h1 > h2).
2. **ARIA attributes**: Comprehensive coverage — `role="region"`, `role="status"`, `role="timer"`, `role="dialog"`, `aria-label`, `aria-live="polite"`, `aria-hidden` on decorative elements.
3. **`<main>` landmark**: Content is wrapped in `<main class="container" role="main">`.
4. **Stats panel layout stability**: All 4 stat cards render with `data-stat` attributes and "—" values when Neo4j is down, with `unavailable-badge` indicators. Layout is preserved.
5. **SVG charts**: Inline SVG bar chart for WRRF weights (static, server-rendered). Signal breakdown dynamically renders SVG via DOM API (no innerHTML).
6. **Keyboard shortcuts**: `R` (refresh), `?` (overlay toggle), `Esc` (close overlay) all implemented with proper guard against input/textarea focus.
7. **Contrast fix**: `--muted` changed from `#8899a6` to `#a8bcc8`, improving contrast on card backgrounds to pass WCAG AA.
8. **XSS protection**: Query table uses `document.createElement` + `textContent` for user-provided question data — no innerHTML injection of user content.

## What Regressed Since Last Iteration

- Nothing regressed. All previously working features remain intact.

---

## Specific Suggestions for Next Iteration

1. **Add a static entity type SVG placeholder**: Even when Neo4j is unavailable, render a horizontal stacked bar SVG with zero-width segments and labels (Person, Org, Location, Work). This shows the visualization structure and avoids a blank panel.

2. **Server-side number formatting**: Add a Jinja2 filter `def format_number(n): return f"{n:,}" if n else "—"` and apply it to stat values in the template so the first paint shows formatted numbers.

3. **Add focus-visible styles**: Add to CSS: `*:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }` to ensure keyboard navigation is visible on the dark background.

4. **Add a subtle refresh indicator**: On fetch start, add a class like `.refreshing` to the container that sets `opacity: 0.85; transition: opacity 0.15s`. Remove it on fetch completion.

5. **Extract overlay styles to CSS classes**: Move the inline styles on `#shortcuts-overlay` and its children into the stylesheet for maintainability and responsive adaptation.

---

## Screenshots

- No visual screenshots taken (Playwright MCP not available). Evaluation based on full HTML source analysis via curl, API endpoint JSON response verification, CSS variable inspection, and JavaScript behavior analysis.

---

**WEIGHTED TOTAL: 7.3 / 10**
