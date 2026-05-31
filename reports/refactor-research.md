# Refactor Research Report: GraphRAG Architecture Patterns

*Generated: 2026-05-28 | Sources: 25+ | Confidence: High*

## Executive Summary

Nghiên cứu 25+ repos và tài liệu liên quan cho thấy hệ thống hiện tại có thể cải thiện đáng kể về modularity, testability và extensibility bằng cách áp dụng các patterns từ Microsoft GraphRAG (33K stars), LightRAG (34K stars, EMNLP 2025), và MODULAR-RAG-MCP-SERVER. Ba thay đổi có impact cao nhất: (1) Strategy pattern cho retrieval pipeline, (2) Explicit state machine cho agent orchestration, (3) Factory pattern cho pluggable components.

---

## 1. Reference Repositories

### Tier 1: Core References

| Repo | Stars | Relevance | Key Pattern |
|------|-------|-----------|-------------|
| [Microsoft GraphRAG](https://github.com/microsoft/graphrag) | 33K | High | Factory pattern, community hierarchy, enterprise structure |
| [LightRAG](https://github.com/HKUDS/LightRAG) | 34K | Very High | Dual-level retrieval, Neo4j backend, reranker integration |
| [nano-graphrag](https://github.com/gusye1234/nano-graphrag) | 4K | Medium | Minimal reference (~1100 lines), clean separation |
| [MODULAR-RAG-MCP-SERVER](https://github.com/nobitalqs/MODULAR-RAG-MCP-SERVER) | - | High | 13 pluggable component families, MCP integration |

### Tier 2: Vietnamese NLP

| Repo | Relevance | Notes |
|------|-----------|-------|
| [undertheseanlp/NLP-Vietnamese-progress](https://github.com/undertheseanlp/NLP-Vietnamese-progress) | SOTA tracking | Benchmark reference |
| [ViWiQA](https://github.com/ViWiQA/ViWiQA) | High | Vietnamese Wikipedia QA, ColBERT/MDR |
| [VIMQA](https://github.com/Nguyen2015/vmqa) | High | 10K+ multi-hop QA pairs |
| [SemViQA](https://github.com/DAVID-NGUYEN-S16/SemViQA) | Medium | Fact-checking, TF-IDF + QATC |
| [VietMedKG](https://github.com/HySonLab/VieMedKG) | Medium | KG + QA benchmark |
| [vndee/awsome-vietnamese-nlp](https://github.com/vndee/awsome-vietnamese-nlp) | Reference | Curated resource list |

### Tier 3: FastAPI + Neo4j RAG

| Repo | Pattern |
|------|---------|
| [FlorentB974/graphrag](https://github.com/FlorentB974/graphrag) | FastAPI + LangGraph + Neo4j |
| [gupta-nakul/langchain-neo4j-rag](https://github.com/gupta-nakul/langchain-neo4j-rag) | Text-to-Cypher, dynamic few-shot |
| [OlaAkindele/rag_chatbot](https://github.com/OlaAkindele/rag_chatbot) | ReAct + Cypher + semantic search |
| [royisme/codebase-rag](https://github.com/royisme/codebase-rag) | MCP + Web UI + REST multi-interface |

---

## 2. Recommended Architecture

### Current vs Proposed Structure

**Current:**
```
src/
├── main.py          (301 lines, mixed API + middleware + MCP)
├── agent.py         (418 lines, ReAct + decomposition + voting)
├── retrieve.py      (monolithic hybrid retrieval)
├── mcp_tools.py     (all MCP tools in one file)
├── ner.py           (pluggable but procedural)
├── llm.py           (Gemini client + embeddings mixed)
└── ...
```

**Proposed:**
```
src/
├── api/
│   ├── routes.py           # FastAPI route handlers (thin)
│   ├── schemas.py          # Pydantic request/response models
│   └── middleware.py       # Auth, rate limiting, error handling
├── orchestration/
│   ├── agent.py            # ReAct agent state machine
│   ├── decomposer.py       # Question decomposition
│   └── voting.py           # Multi-trajectory voting
├── retrieval/
│   ├── base.py             # RetrieverStrategy ABC
│   ├── vector.py           # Vector similarity retriever
│   ├── bm25.py             # BM25 fulltext retriever
│   ├── graph.py            # Graph traversal retriever
│   ├── community.py        # Community-based retriever
│   ├── fusion.py           # WRRF fusion
│   ├── reranker.py         # Cross-encoder reranking
│   └── pipeline.py         # Composable retrieval pipeline
├── extraction/
│   ├── base.py             # NER backend ABC
│   ├── simple.py           # Regex + keyword NER
│   ├── transformer.py      # PhoBERT/ViDeBERTa NER
│   ├── wikilink.py         # Wikipedia hyperlink NER
│   └── postprocess.py      # Noise filtering, dedup
├── generation/
│   ├── llm_client.py       # LLM factory (Gemini/local)
│   ├── embeddings.py       # Embedding generation
│   └── prompts.py          # Prompt templates
├── infrastructure/
│   ├── neo4j.py            # Neo4j driver + schema
│   ├── cache.py            # LLM response caching
│   └── resilience.py       # Retry, circuit breaker
├── mcp/
│   ├── server.py           # MCP server setup
│   └── tools.py            # Tool definitions
├── evaluation/
│   ├── harness.py          # Evaluation runner
│   ├── metrics.py          # Hit rate, MRR, F1
│   └── datasets.py         # Dataset loaders
├── config.py               # Pydantic Settings (centralized)
└── observability/
    ├── logging.py          # Structured logging
    └── tracing.py          # Request tracing
```

---

## 3. Key Design Patterns to Adopt

### 3.1 Strategy Pattern for Retrieval

```python
# src/retrieval/base.py
class RetrieverStrategy(ABC):
    @abstractmethod
    async def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        pass

# src/retrieval/pipeline.py
class HybridRetrievalPipeline:
    def __init__(self, retrievers: dict[str, RetrieverStrategy],
                 fusion: RetrievalFusion, reranker: Reranker | None = None):
        ...

    async def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        # 1. Parallel retrieval
        results = await asyncio.gather(*[r.retrieve(query, 60) for r in self.retrievers.values()])
        # 2. Fuse
        fused = self.fusion.fuse(dict(zip(self.retrievers.keys(), results)))
        # 3. Rerank
        if self.reranker:
            fused = await self.reranker.rerank(query, fused, top_k=20)
        return fused[:top_k]
```

### 3.2 Factory Pattern (from Microsoft GraphRAG)

```python
# src/generation/llm_client.py
class LLMFactory:
    _registry: dict[str, type[BaseLLM]] = {}

    @classmethod
    def register(cls, name: str, llm_class: type[BaseLLM]):
        cls._registry[name] = llm_class

    @classmethod
    def create(cls, config: LLMConfig) -> BaseLLM:
        return cls._registry[config.provider](config)

LLMFactory.register("gemini", GeminiClient)
LLMFactory.register("local", LocalLLMClient)
```

### 3.3 State Machine for Agent (LangGraph-style)

```python
# src/orchestration/agent.py
class AgentState(TypedDict):
    question: str
    complexity: Literal["simple", "complex"]
    sub_questions: list[str]
    retrieved_context: list[RetrievedChunk]
    trajectories: list[Trajectory]
    final_answer: str
    citations: list[Citation]

# Explicit transitions instead of procedural if/else
TRANSITIONS = {
    "classify": lambda s: "decompose" if s["complexity"] == "complex" else "retrieve",
    "retrieve": lambda s: "generate",
    "decompose": lambda s: "multi_retrieve",
    "multi_retrieve": lambda s: "vote",
    "vote": lambda s: "generate",
}
```

### 3.4 Evaluation as First-Class Citizen

```python
# src/evaluation/harness.py
class EvaluationHarness:
    def __init__(self, pipeline: HybridRetrievalPipeline, metrics: list[Metric]):
        ...

    async def evaluate(self, dataset: EvalDataset) -> EvalResults:
        ...

    def compare(self, baseline: EvalResults, candidate: EvalResults) -> ComparisonReport:
        ...
```

---

## 4. LightRAG's Dual-Level Retrieval (Most Relevant)

LightRAG outperforms Microsoft GraphRAG (54.8% vs 45.2%) with a simpler approach:

- **Low-level retrieval**: Entity/relation specific — find exact entities and their connections
- **High-level retrieval**: Topic/theme — find community summaries and broad context
- **Fusion**: Combine both levels for comprehensive answers

**Mapping to your system:**
- Low-level = your graph traversal + entity neighborhood (already have)
- High-level = your community retrieval (already have)
- Missing: explicit separation and dedicated fusion logic per level

---

## 5. Actionable Refactoring Plan

### Phase 1: Quick Wins (1-2 weeks)
1. Move imports to proper locations (done)
2. Extract retrieval strategies into separate files with ABC
3. Enable reranker in pipeline (already implemented, just wire it)
4. Add structured config validation with nested Pydantic models

### Phase 2: Core Refactor (2-4 weeks)
1. Split `agent.py` into orchestration/ package
2. Split `retrieve.py` into retrieval/ package with strategy pattern
3. Create composable pipeline with async parallel retrieval
4. Add experiment tracking for WRRF weight tuning

### Phase 3: Production Hardening (1-2 months)
1. Add circuit breaker + retry for external calls
2. Implement LangGraph-style state machine for agent
3. Continuous evaluation in CI
4. OpenTelemetry tracing

---

## 6. Key Takeaways

1. **LightRAG > Microsoft GraphRAG** for your use case — simpler, better performance, Neo4j support built-in
2. **Strategy pattern** is the single highest-impact refactor — makes retrieval testable and tunable
3. **Reranker integration** is low-hanging fruit — you already have it, just enable it
4. **Vietnamese-specific**: benchmark against ViWiQA and VIMQA datasets for credible evaluation
5. **Don't over-engineer**: nano-graphrag proves ~1100 lines can be production-quality

---

## Sources

### GraphRAG Systems
- [Microsoft GraphRAG](https://github.com/microsoft/graphrag)
- [LightRAG](https://github.com/HKUDS/LightRAG)
- [nano-graphrag](https://github.com/gusye1234/nano-graphrag)
- [MODULAR-RAG-MCP-SERVER](https://github.com/nobitalqs/MODULAR-RAG-MCP-SERVER)

### Vietnamese NLP
- [NLP-Vietnamese-progress](https://github.com/undertheseanlp/NLP-Vietnamese-progress)
- [ViWiQA](https://github.com/ViWiQA/ViWiQA)
- [VIMQA](https://github.com/Nguyen2015/vmqa)
- [SemViQA](https://github.com/DAVID-NGUYEN-S16/SemViQA)
- [awsome-vietnamese-nlp](https://github.com/vndee/awsome-vietnamese-nlp)

### Architecture Patterns
- [AWS Modular RAG](https://aws.amazon.com/blogs/publicsector/use-modular-architecture-for-flexible-and-extensible-rag-based-generative-ai-solutions/)
- [GenAI Patterns: Production Search](https://www.genaipatterns.dev/guides/production-search-pipeline)
- [NVIDIA RAG Blueprint](https://github.com/NVIDIA-AI-Blueprints/rag)
- [LangGraph Agentic RAG](https://docs.langchain.com/oss/python/langgraph/agentic-rag)
- [RAGAS Evaluation](https://docs.ragas.io/en/stable/)
- [Production RAG: 12-Component System](https://nicchin.com/blog/rag-architecture-production)

### FastAPI + Neo4j
- [FlorentB974/graphrag](https://github.com/FlorentB974/graphrag)
- [langchain-neo4j-rag](https://github.com/gupta-nakul/langchain-neo4j-rag)
- [codebase-rag](https://github.com/royisme/codebase-rag)
