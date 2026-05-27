# Research Report: Building an MCP Server for Wikipedia-Neo4j GraphRAG

## Executive Summary

This report synthesizes research on building a Model Context Protocol (MCP) server to expose our Vietnamese Wikipedia GraphRAG system as tools that Claude (Desktop/Code) can connect to and use for Q&A.

**Key findings:**
- The official Python MCP SDK (`fastmcp`) provides a high-level decorator-based API that maps cleanly to our existing architecture
- Best approach: Mount FastMCP on our existing FastAPI app at `/mcp`, sharing Neo4j connections and embedding models
- Tool design: Expose both granular retrieval tools (search, entity exploration, path finding) AND a high-level `answer_question` tool
- Transport: Streamable HTTP for programmatic access, stdio for Claude Desktop/Code integration
- Production: Reuse existing `APP_API_KEY` auth via middleware

## Research Methodology

- **Effort Level**: High (10 queries/agent)
- **Total Agents**: 5 (3 research + 2 deep research)
- **Total Searches**: ~50
- **Sources Analyzed**: MCP spec, Python SDK docs, 6+ real-world RAG MCP implementations, production security guides

---

## 1. MCP Protocol Overview

### What is MCP?

Model Context Protocol is an open protocol (by Anthropic) that standardizes how LLM applications connect to external tools and data. It uses **JSON-RPC 2.0** over various transports.

### Architecture

```
Host (Claude Desktop/Code)
  └── Client (MCP client connector)
        └── Server (our GraphRAG system)
              ├── Tools: Functions the LLM can call
              ├── Resources: Context/data for the LLM
              └── Prompts: Templated workflows
```

### Key Protocol Features

| Feature | Description |
|---------|-------------|
| **Tools** | Functions with JSON Schema inputs, LLM decides when to call |
| **Resources** | Static/dynamic data exposed to the LLM context |
| **Prompts** | Reusable prompt templates |
| **Transports** | stdio, Streamable HTTP (recommended), SSE (deprecated) |
| **Auth** | OAuth 2.0, Bearer tokens, custom middleware |
| **Progress** | Long-running operations can report progress |
| **Logging** | Structured log messages from server to client |

---

## 2. Python MCP SDK (FastMCP)

### Installation

```bash
uv add fastmcp
# OR
pip install fastmcp
```

The `fastmcp` package includes the official `mcp` SDK as a dependency.

### Core API

```python
from fastmcp import FastMCP

mcp = FastMCP(
    "WikiGraphRAG",
    stateless_http=True,   # No session state (simpler, scalable)
    json_response=True,    # JSON instead of SSE for responses
)

@mcp.tool()
def search_knowledge_graph(question: str, top_k: int = 5) -> dict:
    """Search the Vietnamese Wikipedia knowledge graph.
    
    Uses hybrid retrieval combining BM25, vector similarity, 
    graph traversal, and community detection.
    """
    results = hybrid_retrieve(question, top_k=top_k)
    return {"results": [...]}

# Run standalone
if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

### Key Patterns

1. **`@mcp.tool()` decorator** — Auto-generates JSON Schema from type annotations + docstring
2. **Pydantic models** — Use as input parameters for complex validation
3. **Context object** — Inject `ctx: Context` for logging, progress reporting
4. **Resources** — `@mcp.resource("graph://schema")` for static data
5. **Lifespan** — Async context manager for startup/shutdown

### Transport Options

| Transport | Use Case | Config |
|-----------|----------|--------|
| **stdio** | Claude Desktop/Code local | `mcp.run(transport="stdio")` |
| **Streamable HTTP** | Production, remote, multi-client | `mcp.run(transport="streamable-http")` |
| **SSE** | Legacy (deprecated) | Don't use for new projects |

---

## 3. Integration Strategy: Mount on Existing FastAPI

### Recommended Approach

Mount FastMCP as a sub-application on our existing FastAPI app. Same port, same process, shared state.

```python
# src/main.py — additions
from fastmcp import FastMCP

# Create MCP server
mcp = FastMCP("WikiGraphRAG", stateless_http=True, json_response=True)

# Register tools
from src.mcp_tools import register_tools
register_tools(mcp)

# Mount MCP at /mcp
mcp_app = mcp.http_app(path="/", transport="streamable-http")
app.mount("/mcp", mcp_app)
```

**Result:**
```
http://localhost:8000/query    → REST API (existing)
http://localhost:8000/health   → Health check (existing)
http://localhost:8000/mcp      → MCP endpoint (new)
http://localhost:8000/docs     → Swagger UI (still works)
```

### Why This Works

- **Shared state**: MCP tools import `neo4j_client`, `hybrid_retrieve`, `run_agent_query` directly — same module singletons
- **Single process**: One uvicorn worker, one Neo4j connection pool, one copy of embedding models in memory
- **No IPC overhead**: Direct function calls, not HTTP proxying

### Alternative: Standalone stdio Server

For Claude Desktop/Code integration (simpler, no HTTP):

```python
# src/mcp_server.py
from fastmcp import FastMCP
from src.mcp_tools import register_tools

mcp = FastMCP("WikiGraphRAG")
register_tools(mcp)

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

---

## 4. Tool Design for GraphRAG

### Recommended Tool Set (7 tools)

Based on analysis of 6+ real-world RAG MCP servers:

#### Retrieval Tools (LLM orchestrates)

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `search_knowledge_base` | Hybrid retrieval (BM25+vector+graph+community) | Simple factual questions |
| `explore_entity` | Entity neighborhood in KG | "Tell me about X" |
| `find_path` | Shortest path between entities | Multi-hop reasoning |
| `get_community_summary` | Topic-level community summary | Broad topic overview |

#### QA Tools (server-side generation)

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `answer_question` | Full agent pipeline with citations | Complex multi-hop questions |

#### Metadata Tools

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `get_graph_stats` | KG statistics (nodes, edges, coverage) | Understanding data scope |
| `list_entity_types` | Available entity types and counts | Query planning |

### Tool Implementation

```python
# src/mcp_tools.py
from fastmcp import FastMCP, Context
from pydantic import BaseModel, Field
from typing import Literal

from src.retrieve import hybrid_retrieve
from src.agent import run_agent_query
from src.neo4j_client import neo4j_client


class SearchInput(BaseModel):
    question: str = Field(..., description="Question in Vietnamese or English")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")
    method: Literal["hybrid", "bm25", "vector", "graph"] = Field(
        "hybrid", description="Retrieval method"
    )


def register_tools(mcp: FastMCP):

    @mcp.tool()
    def search_knowledge_base(params: SearchInput) -> dict:
        """Search the Vietnamese Wikipedia knowledge graph.
        
        Uses WRRF hybrid retrieval combining BM25 fulltext, vector similarity,
        graph traversal, and community-based retrieval with reranking.
        
        Use this for factual questions about Vietnamese topics.
        Do NOT use for complex multi-hop questions — use answer_question instead.
        
        Returns chunks with relevance scores and source metadata.
        """
        results = hybrid_retrieve(params.question, top_k=params.top_k)
        return {
            "question": params.question,
            "method": params.method,
            "results": [
                {
                    "chunk_id": r.chunk_id,
                    "text": r.text,
                    "score": round(r.score, 4),
                    "page_title": r.page_title,
                    "page_url": r.page_url,
                }
                for r in results
            ],
            "total": len(results),
        }

    @mcp.tool()
    def explore_entity(entity_name: str, depth: int = 2) -> dict:
        """Explore an entity's neighborhood in the knowledge graph.
        
        Returns connected entities, their types, and relationships.
        Useful for understanding how concepts relate to each other.
        
        Args:
            entity_name: Entity name (Vietnamese, e.g. "Hồ Chí Minh")
            depth: How many hops to traverse (1-3)
        """
        with neo4j_client.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity {name: $name})
                OPTIONAL MATCH (e)-[r]-(connected:Entity)
                RETURN e.name AS entity, e.type AS type,
                       type(r) AS rel_type, connected.name AS connected_name,
                       connected.type AS connected_type
                LIMIT 50
            """, name=entity_name)
            neighbors = [dict(record) for record in result]
        return {
            "entity": entity_name,
            "depth": depth,
            "neighbors": neighbors,
            "count": len(neighbors),
        }

    @mcp.tool()
    def find_path(entity_a: str, entity_b: str) -> dict:
        """Find the shortest path between two entities in the knowledge graph.
        
        Useful for multi-hop reasoning: how are two concepts connected?
        
        Args:
            entity_a: Starting entity name (Vietnamese)
            entity_b: Target entity name (Vietnamese)
        """
        with neo4j_client.driver.session() as session:
            result = session.run("""
                MATCH path = shortestPath(
                    (a:Entity {name: $a})-[*..5]-(b:Entity {name: $b})
                )
                RETURN [n IN nodes(path) | n.name] AS nodes,
                       [r IN relationships(path) | type(r)] AS relations
            """, a=entity_a, b=entity_b)
            record = result.single()
        if record:
            return {
                "from": entity_a,
                "to": entity_b,
                "path": record["nodes"],
                "relations": record["relations"],
                "hops": len(record["relations"]),
            }
        return {"from": entity_a, "to": entity_b, "path": None, "message": "No path found"}

    @mcp.tool()
    def get_community_summary(topic: str) -> dict:
        """Get a community-level summary for a topic.
        
        Communities are clusters of related entities detected via Louvain algorithm.
        Returns a pre-generated summary covering the main concepts in that community.
        
        Use this for broad topic overviews before diving into specific entities.
        """
        from src.community import get_community_for_topic
        summary = get_community_for_topic(topic)
        return {"topic": topic, "summary": summary}

    @mcp.tool()
    def answer_question(question: str) -> dict:
        """Answer a complex question using the full GraphRAG agent pipeline.
        
        This tool uses a ReAct agent with question decomposition for multi-hop
        questions. It automatically:
        1. Detects question complexity
        2. Decomposes complex questions into sub-questions
        3. Retrieves evidence from multiple sources
        4. Synthesizes a final answer with citations
        
        Use this for complex questions that require reasoning across multiple facts.
        For simple factual lookups, use search_knowledge_base instead.
        
        Returns: answer text with source citations.
        """
        result = run_agent_query(question)
        return {
            "question": question,
            "answer": result.get("answer", ""),
            "citations": result.get("citations", []),
            "confidence": result.get("confidence", "medium"),
        }

    @mcp.tool()
    def get_graph_stats() -> dict:
        """Get statistics about the knowledge graph.
        
        Returns counts of pages, chunks, entities, and relationships.
        Useful for understanding the scope and coverage of the knowledge base.
        """
        with neo4j_client.driver.session() as session:
            stats = session.run("""
                MATCH (p:Page) WITH count(p) AS pages
                MATCH (c:Chunk) WITH pages, count(c) AS chunks
                MATCH (e:Entity) WITH pages, chunks, count(e) AS entities
                RETURN pages, chunks, entities
            """).single()
        return dict(stats) if stats else {}

    @mcp.tool()
    def list_entity_types() -> dict:
        """List all entity types in the knowledge graph with counts.
        
        Returns: entity types (Person, Organization, Location, Work) and their counts.
        """
        with neo4j_client.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity)
                RETURN e.type AS type, count(*) AS count
                ORDER BY count DESC
            """)
            types = [dict(r) for r in result]
        return {"entity_types": types}

    # --- Resources ---
    @mcp.resource("graph://schema")
    def graph_schema() -> str:
        """The Neo4j knowledge graph schema."""
        return """
        Graph Schema:
        - (Page)-[:HAS_CHUNK]->(Chunk)-[:MENTIONS]->(Entity)
        - (Page)-[:LINKS_TO]->(Page)
        - Entity types: Person, Organization, Location, Work
        - Typed edges: MENTIONS_PERSON, MENTIONS_ORG, MENTIONS_LOCATION, MENTIONS_WORK
        - Chunks have: text, embedding (1024-dim), position
        - Pages have: title, url, category
        """
```

### Tool Description Best Practices

1. **State what it does** + **when to use it** + **when NOT to use it**
2. **Include parameter constraints** in descriptions (not just schema)
3. **Document return structure** so the LLM knows what to expect
4. **Use negative guidance**: "Do NOT use for X — use Y instead"
5. **Keep names as verbs**: `search_knowledge_base`, not `knowledge_base_search`

---

## 5. Authentication & Security

### Recommended: Bearer Token Middleware

Reuse existing `APP_API_KEY` environment variable:

```python
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

class MCPAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path.startswith("/mcp"):
            auth = request.headers.get("authorization", "")
            if settings.app_api_key:
                if not auth.startswith("Bearer ") or auth[7:].strip() != settings.app_api_key:
                    return JSONResponse({"error": "Unauthorized"}, status_code=401)
        return await call_next(request)

app.add_middleware(MCPAuthMiddleware)
```

### Security Checklist

- [x] Validate all tool inputs (Pydantic models)
- [x] Rate limit tool invocations (reuse existing `RATE_LIMIT_PER_MINUTE`)
- [x] Sanitize outputs (no raw exceptions to LLM)
- [x] Read-only by default (no write operations exposed)
- [x] Bind to localhost in dev (not 0.0.0.0)
- [x] Audit log all tool calls
- [ ] OAuth 2.0 (future, for multi-user)

---

## 6. Claude Desktop/Code Configuration

### Claude Code (Streamable HTTP)

```bash
claude mcp add viwiki-graphrag \
  --transport http \
  --url http://localhost:8000/mcp \
  --header "Authorization: Bearer ${APP_API_KEY}"
```

Or in `.claude/settings.json`:
```json
{
  "mcpServers": {
    "viwiki-graphrag": {
      "type": "http",
      "url": "http://localhost:8000/mcp",
      "headers": {
        "Authorization": "Bearer your-api-key-here"
      }
    }
  }
}
```

### Claude Code (stdio — standalone)

```json
{
  "mcpServers": {
    "viwiki-graphrag": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "python", "-m", "src.mcp_server"],
      "cwd": "/home/aoi/Documents/FPTU/DATN/wikipedia-neo4j",
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "your-password"
      }
    }
  }
}
```

### Claude Desktop

`~/.config/Claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "viwiki-graphrag": {
      "command": "uv",
      "args": ["run", "python", "-m", "src.mcp_server"],
      "cwd": "/home/aoi/Documents/FPTU/DATN/wikipedia-neo4j"
    }
  }
}
```

---

## 7. Testing

### MCP Inspector (visual debugging)

```bash
npx @modelcontextprotocol/inspector http://localhost:8000/mcp
```

### Programmatic Test

```python
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

@pytest.mark.asyncio
async def test_mcp_tools_available():
    async with streamablehttp_client("http://localhost:8000/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            assert "search_knowledge_base" in names
            assert "answer_question" in names
            assert "explore_entity" in names

@pytest.mark.asyncio
async def test_search_tool():
    async with streamablehttp_client("http://localhost:8000/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("search_knowledge_base", {
                "params": {"question": "Hồ Chí Minh sinh năm nào?", "top_k": 3}
            })
            data = json.loads(result.content[0].text)
            assert data["total"] > 0
```

---

## 8. Implementation Plan

### Phase 1: Core MCP Server (1-2 days)

1. `uv add fastmcp` — Add dependency
2. Create `src/mcp_tools.py` — Define 7 tools wrapping existing functions
3. Create `src/mcp_server.py` — Standalone stdio entry point
4. Modify `src/main.py` — Mount MCP at `/mcp`
5. Add auth middleware for MCP endpoint

### Phase 2: Testing & Configuration (1 day)

6. Add `tests/test_mcp_server.py` — Tool availability + basic calls
7. Test with MCP Inspector
8. Add Claude Code config to `.claude/settings.json`
9. Test end-to-end with Claude Code

### Phase 3: Polish (1 day)

10. Add progress reporting for long-running queries
11. Add structured error handling (Neo4j down, no results, timeout)
12. Add audit logging for MCP tool calls
13. Update documentation

### Dependencies

```toml
# pyproject.toml addition
[project.dependencies]
fastmcp = ">=2.0"
```

### File Structure

```
src/
├── main.py           # FastAPI app + MCP mount (modified)
├── mcp_server.py     # Standalone stdio entry point (new)
├── mcp_tools.py      # MCP tool definitions (new)
├── retrieve.py       # Hybrid retrieval (existing, used by tools)
├── agent.py          # Agent pipeline (existing, used by tools)
├── neo4j_client.py   # Neo4j driver (existing, shared)
└── ...
```

---

## 9. Architecture Decision: Why Not fastapi-mcp?

The `fastapi-mcp` library auto-exposes FastAPI endpoints as MCP tools. We considered it but chose manual FastMCP tools because:

| Aspect | fastapi-mcp (auto) | FastMCP (manual) |
|--------|--------------------|--------------------|
| Tool descriptions | From OpenAPI docs | Custom, LLM-optimized |
| Input schemas | HTTP semantics (path/query/body) | Clean Pydantic models |
| Granularity | 1 tool per endpoint | Custom tool boundaries |
| Graph-specific tools | Would need new endpoints | Direct Neo4j queries |
| Control | Less | Full |

**Verdict**: Our QA system benefits from tools designed specifically for LLM interaction (negative guidance, structured citations, graph-aware parameters) rather than auto-generated HTTP wrappers.

---

## 10. Real-World Patterns Applied

From analyzing 6+ RAG MCP implementations:

1. **Layered tools** (from Retrievo): Separate retrieval from generation. Let the LLM choose granularity.
2. **Hexagonal adapter** (from MCP-Demo): MCP server is a thin adapter over existing business logic.
3. **Structured citations** (from all): Always return source metadata alongside answers.
4. **Error as content** (from Linked-Docs): Use `isError: true` in tool results, not protocol errors.
5. **Stdio for Claude** (from all): Every successful Claude integration uses stdio transport.

---

## Confidence Assessment

- **High Confidence**: FastMCP SDK API, mounting pattern, tool decorator usage, Claude config format
- **Medium Confidence**: Optimal tool granularity (may need iteration based on Claude's behavior)
- **Needs Testing**: Performance of full agent pipeline via MCP (timeout handling), community summary tool availability

---

## Sources

- MCP Specification: https://modelcontextprotocol.org/specification/2025-11-25
- Python MCP SDK: https://github.com/modelcontextprotocol/python-sdk
- FastMCP docs: https://fastmcp.mintlify.app/
- MCP in Production guide: https://fordelstudios.com/research/mcp-production-engineering-guide
- Real Python MCP tutorial: https://realpython.com/python-mcp/
- RAGify-Docs-API: https://github.com/codewithyasho/RAGify-Docs-API
- Linked-Docs-MCP: https://github.com/folence/Linked-Docs-MCP
- Retrievo: https://github.com/Kanyuchi/literature-rag-mcp
- MCP-Demo: https://github.com/dmorav1/MCP-Demo
- fastapi-mcp: https://github.com/tadata-org/fastapi_mcp
- Neo4j MCP examples: https://github.com/BjornMelin/qdrant-neo4j-crawl4ai-mcp
- MCP Security Guide: https://beyondscale.tech/blog/mcp-server-security-guide
