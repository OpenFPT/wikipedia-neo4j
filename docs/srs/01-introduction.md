# 1. Introduction

## 1.1 Purpose

This Software Requirements Specification (SRS) defines the functional and non-functional requirements for the **ViWiki-GraphRAG** system — a Graph-based Retrieval-Augmented Generation platform for Multi-hop Question Answering over Vietnamese Wikipedia.

**Intended audience:**
- Capstone review committee (FPT University, AI/SE track)
- Development team (3 members)
- Future maintainers and researchers

## 1.2 Problem Statement

Vietnamese Wikipedia contains **1.29 million articles** covering diverse topics (history, geography, culture, science). However:

1. **No multi-hop QA system exists** for Vietnamese Wikipedia. Users cannot ask questions that require reasoning across multiple articles (e.g., "Ai là người sáng lập trường đại học mà Ngô Bảo Châu từng học?").

2. **Existing Vietnamese QA systems are limited:**
   - UIT-ViQuAD 2.0 (2021): Extractive QA on single paragraphs — cannot handle cross-document reasoning
   - VIMQA (2022): Multi-hop dataset exists but no deployed retrieval system leveraging knowledge graph structure
   - Commercial search (Google): Returns documents, not direct answers with citations

3. **Knowledge graph structure is underutilized:** Wikipedia's hyperlink graph naturally encodes entity relationships (person → organization, location → event) that enable multi-hop traversal, but no Vietnamese system exploits this for QA.

**Pain point:** Vietnamese students and researchers spend significant time manually navigating between Wikipedia articles to answer complex questions that span multiple topics.

## 1.3 Scope

### In Scope

- Ingest Vietnamese Wikipedia articles into a Neo4j knowledge graph
- Extract named entities (Person, Organization, Location, Work) using pluggable NER
- Generate semantic embeddings for vector similarity search
- Hybrid retrieval combining BM25 + vector + graph traversal + community detection
- Answer multi-hop questions using a ReAct agent with tool use
- REST API for programmatic access
- Evaluation pipeline with standard IR/NLP metrics

### Out of Scope

- Real-time Wikipedia synchronization (batch ingestion only)
- Languages other than Vietnamese
- User account management and personalization
- Mobile application frontend
- Production deployment at scale (research prototype)

## 1.4 Definitions and Acronyms

| Term | Definition |
|------|-----------|
| GraphRAG | Graph-based Retrieval-Augmented Generation |
| WRRF | Weighted Reciprocal Rank Fusion — combines multiple retrieval signals |
| NER | Named Entity Recognition — extracting entities from text |
| BM25 | Best Matching 25 — probabilistic text retrieval algorithm |
| MRR | Mean Reciprocal Rank — retrieval quality metric |
| EM | Exact Match — answer accuracy metric |
| F1 | Token-level F1 score — partial answer accuracy |
| KG | Knowledge Graph — structured entity-relationship database |
| ReAct | Reasoning + Acting — agent paradigm combining thought and tool use |
| SLM | Small Language Model (< 10B parameters) |

## 1.5 Assumptions and Dependencies

### Assumptions

1. Neo4j 5.x is available as a local service (systemd) on `bolt://localhost:7687`
2. GPU with ≥ 8GB VRAM available for local LLM inference (Vi-Qwen2-7B-RAG, 4-bit quantized)
3. Vietnamese Wikipedia dump accessible via HuggingFace Datasets (`wikimedia/wikipedia`, `20231101.vi`)
4. Google Gemini API keys available for embedding generation (free tier sufficient for development)
5. Team members have Python 3.11+ development environment with `uv` package manager

### Dependencies

| Dependency | Version | Purpose |
|-----------|---------|---------|
| Neo4j | 5.x | Knowledge graph storage |
| Python | 3.11+ | Runtime |
| PyTorch | 2.x | Local model inference |
| FastAPI | 0.100+ | REST API framework |
| Google Generative AI SDK | latest | Gemini embedding + generation |
| Transformers (HuggingFace) | 4.x | NER and local LLM |

## 1.6 Document Conventions

- Requirements are labeled `FR-XX` (functional) and `NFR-XX` (non-functional)
- Priority uses MoSCoW: Must / Should / Could / Won't
- All performance targets are measurable and include verification method
