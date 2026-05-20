---
title: "SoK: Agentic Retrieval-Augmented Generation (RAG): Taxonomy, Architectures, Evaluation, and Research Directions"
authors: SoK Authors (arXiv 2026)
year: 2026
url: https://arxiv.org/abs/2603.07379
venue: arXiv preprint
---

tags:: [[paper]], [[agentic-rag]], [[survey]]

# [[SoK Team 2026 - Agentic RAG]]

## TL;DR
This Systematization of Knowledge paper provides the first unified framework for Agentic RAG by formally modeling these systems as finite-horizon partially observable Markov decision processes (POMDPs). It develops a comprehensive taxonomy covering planning mechanisms, retrieval orchestration, memory paradigms, and tool-invocation behaviors, while identifying severe systemic risks in autonomous loops that are absent from traditional static RAG evaluation.

## Method
The authors survey and systematize published agentic RAG systems, formalizing them through a single theoretical lens: the agentic retrieval-generation loop as a POMDP with explicit control policies and state transitions. They build a modular architectural decomposition categorizing systems by:
- **Planning:** static pre-defined plans vs. dynamic adaptive plans.
- **Retrieval orchestration:** single-hop lookup vs. iterative multi-step tool calls.
- **Memory:** in-context vs. external vector stores vs. KG stores.
- **Tool invocation:** function-calling APIs, code execution, and structured query generators.

They analyze traditional static evaluation practices and identify their critical insufficiency for agentic settings, then outline research directions.

## Results
- Identified **four severe systemic risks** unique to autonomous loops: compounding hallucination propagation, memory poisoning, retrieval misalignment, and cascading tool-execution vulnerabilities.
- Demonstrated that **existing evaluation benchmarks** are not designed to catch these failure modes since they test outputs, not trajectory correctness.
- Mapped over 50 published architectures onto the unified taxonomy, exposing inconsistencies in how "agentic RAG" is defined across the literature.

## Relevance
This is the theoretical grounding document for our orchestrator design:
- **What we borrow:** The POMDP formalization. Our 4-tool ReAct loop is a finite-horizon POMDP: the SLM acts as the policy, tool outputs are observations, and the final answer is the terminal state. The iteration cap (max 6 steps) is our horizon limit.
- **What we adapt:** The risk taxonomy. We directly address *compounding hallucination propagation* by inserting the `verify_tool` as a mid-trajectory check rather than only evaluating the final output.
- **What we avoid:** Multi-agent architectures (identified in SoK as introducing cascading tool vulnerabilities) and unbounded loops (identified as compounding hallucination risk).
