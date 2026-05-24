---
title: "An Empirical Study of Multi-Agent RAG for Real-World University Admissions Counseling"
authors: MARAUS Authors (UTT Hanoi collaboration)
year: 2025
url: https://arxiv.org/abs/2507.11272
venue: arXiv preprint
---

tags:: [[paper]], [[vietnamese-nlp]], [[agentic-rag]]

# [[MARAUS Team 2025 - MARAUS]]

## TL;DR
MARAUS (Multi-Agent and Retrieval-Augmented University Admission System) is a real-world deployment of a conversational AI platform for higher education admissions counseling in Vietnam, developed in collaboration with the University of Transport Technology (UTT) in Hanoi. Processing over 6,000 actual user interactions, it achieved 92% accuracy, reduced hallucination rates from 15% to 1.45%, and kept response times under 4 seconds at a two-week deployment cost of 11.58 USD using GPT-4o mini.

## Method
A two-phase study: technical development followed by real-world evaluation.
- **Hybrid retrieval:** Combines sparse and dense retrieval over structured admissions documents (tuition, cut-off scores, majors, deadlines).
- **Multi-agent orchestration:** Six query categories (score lookup, major info, deadlines, eligibility, general info, unanswerable) are routed to specialized sub-agents.
- **LLM-based generation:** GPT-4o mini generates final answers using retrieved context.
- **Verification:** Structured consistency checks cross-reference answers against institutional databases before serving.

## Results
- **92% average accuracy** across six query categories.
- Hallucination rate: **15% → 1.45%** (vs. LLM-only baseline).
- Average response time: **< 4 seconds**.
- Deployment cost: **11.58 USD** for two weeks — demonstrating cost-effective production viability.

## Relevance
This is the closest real-world Vietnamese agentic RAG deployment prior to our system:
- **What we borrow:** The query-type routing pattern (structured categories → specialized retrieval paths) and the verification step (cross-referencing generated answers against a structured data source).
- **What we adapt:** Domain — from closed admissions documents to open-domain Wikipedia. We replace the closed institutional database with our Neo4j Vietnamese Wikipedia KG, and replace specialized sub-agents with a unified 4-tool ReAct loop.
- **What we avoid:** GPT-4o mini at inference time. MARAUS's cost-efficiency argument ($11.58/2 weeks) still requires internet-connected API calls. Our system targets full offline operation.
