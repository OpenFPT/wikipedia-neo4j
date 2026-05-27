# MCP Server Optimization Research Report

**Date:** 2026-05-27
**Topic:** Optimizing MCP tools for GraphRAG evaluation

---

## 1. Reference Implementations — GraphRAG MCP Tool Patterns

### Tool Inventory Across Repos

| Tool Category | Our System | claude-graphrag-mcp | graphrag_mcp | pggraphrag-mcp | rag-anythink-mcp | GraphRAG (MvdB) |
|---------------|-----------|--------------------|--------------|-----------------|--------------------|-----------------|
| Hybrid search | `search_knowledge_base` | `query` | `hybrid_search` | `retrieve_hybrid` | `query_knowledge_graph` | `search` |
| Entity explore | `explore_entity` | (via query) | (via graph) | `entity_search` + `entity_expand` | - | `get_related` |
| Path finding | `find_path` | - | - | - | - | - |
| Community | `get_community_summary` | - | - | - | - | - |
| Full agent | `answer_question` | - | - | - | - | - |
| Stats | `get_graph_stats` | - | - | `graph_status` | `get_graph_statistics` | `list_documents` |
| Entity types | `list_entity_types` | - | - | - | - | - |
| **Source trace** | **MISSING** | via EXTRACTED_FROM | - | `source_trace` | - | `get_chunk_context` |
| **Naive/vector search** | **MISSING** | - | `search_documentation` | `retrieve_naive` | mode param | `search` |
| **Fact verification** | **MISSING** | - | - | - | - | - |

### Missing Tools We Should Add

1. **`source_trace`** — Given a chunk or answer, trace back to source page/paragraph. Critical for provenance.
2. **`verify_claim`** — Given a statement, return supporting/contradicting evidence from KG.
3. **`search_by_mode`** — Allow explicit mode selection (bm25-only, vector-only) as separate parameter.

### Best Practices from Reference Repos

- **String-in, evidence-out**: Callers pass natural language; server handles embeddings
- **Fail fast**: Database errors propagate to MCP client so Claude sees "Neo4j unreachable" not empty results
- **Bounded surface**: Don't expose arbitrary Cypher/SQL — keep tools intentionally bounded
- **Chunk context expansion**: Return neighboring chunks around a hit for better context

---

## 2. Tool Description Optimization

### Key Papers

| Paper | Finding |
|-------|---------|
| **Tool Preferences in Agentic LLMs are Unreliable** (2505.18135) | LLMs rely entirely on text descriptions; process is "surprisingly fragile" |
| **Uncovering Tool Selection Bias** (2510.00307) | Semantic alignment between query and metadata is strongest predictor; small perturbations shift selections significantly |
| **Learning to Rewrite Tool Descriptions** (2602.20426) | Tool interfaces are "human-oriented" and become bottleneck; LLM-optimized rewrites improve selection |
| **MetaTool** (2310.03128) | Detailed descriptions with usage scenarios significantly improve tool selection |
| **toolpick** (pontusab/toolpick) | Description enrichment + LLM reranking: 84% to 100% accuracy |

### Principles for Effective Tool Descriptions

1. **Lead with WHEN to use** — First sentence should be the trigger condition, not what the tool does
2. **Include WHEN NOT to use** — Explicitly state boundaries to prevent confusion with similar tools
3. **Use the user's vocabulary** — Match how questions are phrased, not internal system terminology
4. **Add concrete examples** — "e.g., 'Ai la tong thong dau tien?'" helps semantic matching
5. **Keep parameter descriptions minimal** — Only describe non-obvious parameters
6. **Avoid jargon in descriptions** — "WRRF hybrid retrieval" means nothing to the LLM; say "searches across text, meaning, and connections"
7. **Differentiate similar tools clearly** — If two tools could match, the first sentence must disambiguate

### Recommended Tool Description Rewrites

```python
# BEFORE (current)
"Search the Vietnamese Wikipedia knowledge graph. Uses WRRF hybrid retrieval..."

# AFTER (optimized)
"Use for simple factual questions that need 1-2 facts from Vietnamese Wikipedia.
Examples: 'Ai sang lap Dang Cong san?', 'Ha Noi co bao nhieu quan?'
Do NOT use for questions requiring reasoning across multiple facts — use answer_question instead.
Returns text passages ranked by relevance with source page links."
```

```python
# BEFORE
"Explore an entity's neighborhood in the knowledge graph."

# AFTER
"Use when you already know an entity name and want to see what it connects to.
Examples: after finding 'Ho Chi Minh', explore to find related people, places, events.
Do NOT use for searching — use search_knowledge_base first to find entities."
```

```python
# BEFORE
"Find the shortest path between two entities in the knowledge graph."

# AFTER
"Use for multi-hop questions asking HOW two things are connected.
Examples: 'Moi quan he giua X va Y la gi?', 'X lien quan den Y nhu the nao?'
Requires knowing both entity names — use search_knowledge_base first if unsure."
```

### Token Budget Consideration

- Haiku has 200K context but tool descriptions eat tokens on every call
- With 4 tools x ~100 tokens each = ~400 tokens overhead per API call
- Over 100 questions x 2 rounds avg = 800 tokens wasted on verbose descriptions
- **Recommendation**: Keep descriptions under 80 tokens each; move details to parameter descriptions

---

## 3. Evaluation Methodology — Beyond F1/EM

### Framework Comparison

| Framework | Dimensions | Key Insight |
|-----------|-----------|-------------|
| **TRACE** | Efficiency, Hallucination, Adaptivity | Reference-free; uses evidence bank from preceding steps |
| **ToolEyes** | Format, Intent, Planning, Selection, Organization | 5-dimension scoring across 7 real-world scenarios |
| **ToolSandbox** | Stateful execution, state dependencies | Milestones (must-achieve) + Minefields (must-avoid) |
| **ToolHop** | Multi-hop chaining accuracy | Best model (GPT-4o) only 49% — multi-hop is hard |
| **ToolBeHonest** | Hallucination depth + breadth | Solvability detection, missing-tool analysis |
| **ToolLLM** | Pass rate, Win rate | 87% agreement with human annotators |

### Metrics to Add to Our Eval Script

| Metric | What It Measures | Implementation Cost |
|--------|-----------------|-------------------|
| **Tool selection accuracy** | Did Claude pick the right tool first? | Low — tag questions by expected tool |
| **Retrieval efficiency** | How many tool calls needed? | Free — already in trajectory |
| **Faithfulness** | Is answer grounded in retrieved context? | Medium — entity overlap check |
| **Hallucination rate** | Did it fabricate info? | Medium — NER on answer vs context |
| **Recovery rate** | Did it adapt after empty results? | Low — count empty-to-retry patterns |
| **Unsolvable detection** | Does it refuse when answer isn't in KG? | Low — add synthetic questions |

### Implementation Sketch

```python
@dataclass
class EnhancedEvalResult:
    # Answer quality
    f1: float
    em: float
    # Tool behavior
    tool_calls_count: int
    first_tool_correct: bool
    empty_calls: int
    recovery_success: bool
    # Faithfulness
    faithfulness_score: float  # entities_in_context / entities_in_answer
    hallucination_detected: bool
    # Efficiency
    latency_ms: float
    input_tokens: int
    output_tokens: int
```

**Faithfulness computation (cheap, no LLM needed):**
```python
def compute_faithfulness(answer: str, tool_results: list[str]) -> float:
    """Entity-overlap faithfulness."""
    from src.ner import extract_entities
    answer_entities = {e["name"].lower() for e in extract_entities(answer)}
    context = " ".join(tool_results).lower()
    if not answer_entities:
        return 1.0
    grounded = sum(1 for e in answer_entities if e in context)
    return grounded / len(answer_entities)
```

---

## 4. Actionable Recommendations

### Priority 1: Improve Tool Descriptions (High Impact, Low Effort)

- Rewrite all tool descriptions following "WHEN to use / WHEN NOT to use" pattern
- Add Vietnamese example queries to each description
- Remove internal jargon (WRRF, BM25, Louvain)
- Keep each description under 80 tokens

### Priority 2: Add Source Trace Tool (Medium Impact, Low Effort)

```python
@mcp.tool()
def trace_source(chunk_id: str) -> dict:
    """Given a chunk ID from search results, return the full source page context.
    Use after search_knowledge_base to get more context around a relevant passage."""
```

### Priority 3: Enhance Eval Script (High Impact, Medium Effort)

- Add trajectory logging (tool name, args, result length, latency per call)
- Add faithfulness metric (entity overlap between answer and retrieved chunks)
- Add efficiency metric (tool calls per question)
- Add 10-20 "unsolvable" questions to test refusal behavior
- Tag questions by expected tool type for tool selection accuracy

### Priority 4: Description Enrichment (Medium Impact, Medium Effort)

- Use toolpick-style synonym enrichment at startup
- Test with/without enrichment on a 50-question sample
- Measure tool selection accuracy improvement

---

## 5. Key Papers and Repos

### Papers
- TRACE: arxiv.org/abs/2510.02837
- ToolEyes: arxiv.org/abs/2401.00741
- ToolSandbox: arxiv.org/abs/2408.04682
- ToolHop: arxiv.org/abs/2501.02506
- ToolBeHonest: arxiv.org/abs/2406.20015
- ToolLLM: arxiv.org/abs/2307.16789
- Tool Preferences Unreliable: arxiv.org/abs/2505.18135
- Tool Selection Bias: arxiv.org/abs/2510.00307
- Rewrite Tool Descriptions: arxiv.org/abs/2602.20426
- MetaTool: arxiv.org/abs/2310.03128
- KAG (Knowledge Augmented Generation): arxiv.org/abs/2409.13731

### Reference Repos
- leakydata/claude-graphrag-mcp — Neo4j + OpenAI embeddings, KAG-inspired
- rileylemm/graphrag_mcp — Neo4j + Qdrant hybrid
- rioriost/pggraphrag-mcp — PostgreSQL + Apache AGE, source tracing
- ArthurSrz/graphRAGmcp — 29x faster than vector RAG, provenance chains
- serkanyasr/rag-anythink-mcp — Neo4j + pgvector, multimodal
- MvdBerg/GraphRAG — PostgreSQL + AGE, chunk context expansion
- engineering-with-ai/python-mcp-server — Graphiti + pgvector, fact verification
- pontusab/toolpick — Tool selection via embeddings + LLM reranking
- bernard777/tool-selector-cascade — 3-level cascading tool selector
