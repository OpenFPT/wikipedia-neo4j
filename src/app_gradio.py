"""Gradio web demo: query → reasoning trace → cited answer."""

from __future__ import annotations

import json
import time

import gradio as gr

from src.agent_tools import kg_schema, kg_query, text_search, get_passage, ToolResult
from src.config import settings
from src.logging_utils import get_logger

logger = get_logger(__name__)


def format_tool_result(result: ToolResult) -> str:
    """Format a tool result for display."""
    if not result.success:
        return f"[Error] {result.error}"
    if isinstance(result.data, list):
        return json.dumps(result.data[:5], ensure_ascii=False, indent=2)
    if isinstance(result.data, dict):
        return json.dumps(result.data, ensure_ascii=False, indent=2)
    return str(result.data)


def process_query(question: str, top_k: int = 5) -> tuple[str, str, str]:
    """Process a question through the agent pipeline.

    Returns: (answer, reasoning_trace, citations)
    """
    if not question.strip():
        return "Please enter a question.", "", ""

    trace_steps = []
    start = time.time()

    # Step 1: Text search for relevant passages
    trace_steps.append("**Step 1:** Searching for relevant passages...")
    search_result = text_search(question, top_k=top_k)
    passages: list[dict] = search_result.data if search_result.success and isinstance(search_result.data, list) else []

    if passages:
        trace_steps.append(f"  Found {len(passages)} passages")
        for i, p in enumerate(passages[:3]):
            trace_steps.append(f"  [{i+1}] {p.get('article_title', 'Unknown')} (score: {p.get('score', 0):.3f})")

    # Step 2: Try KG query for structured answer
    trace_steps.append("\n**Step 2:** Querying knowledge graph...")
    kg_result = kg_query(
        "MATCH (e)-[r]->(t) WHERE toLower(e.name) CONTAINS toLower($q) "
        "RETURN e.name AS source, type(r) AS relation, t.name AS target LIMIT 5",
        params={"q": question},
    )
    kg_data: list[dict] = kg_result.data if kg_result.success and isinstance(kg_result.data, list) else []

    if kg_data:
        trace_steps.append(f"  Found {len(kg_data)} KG triples")
        for triple in kg_data[:3]:
            trace_steps.append(f"  {triple.get('source', '?')} --[{triple.get('relation', '?')}]--> {triple.get('target', '?')}")
    else:
        trace_steps.append("  No KG results found")

    # Step 3: Synthesize answer
    trace_steps.append("\n**Step 3:** Synthesizing answer...")
    elapsed = time.time() - start

    # Build answer from retrieved context
    answer_parts = []
    if kg_data:
        answer_parts.append("From Knowledge Graph:")
        for triple in kg_data[:3]:
            answer_parts.append(f"  - {triple.get('source', '')} → {triple.get('relation', '')} → {triple.get('target', '')}")

    if passages:
        answer_parts.append("\nFrom Text Retrieval:")
        for p in passages[:2]:
            text_snippet = p.get("text", "")[:200]
            answer_parts.append(f"  [{p.get('article_title', '')}]: {text_snippet}...")

    if not answer_parts:
        answer = "Could not find relevant information. Try rephrasing your question or ingesting more data."
    else:
        answer = "\n".join(answer_parts)

    trace_steps.append(f"\n**Done** in {elapsed:.2f}s")
    reasoning_trace = "\n".join(trace_steps)

    # Citations
    citations_parts = []
    if passages:
        for p in passages[:5]:
            citations_parts.append(f"- **{p.get('article_title', 'Unknown')}** (paragraph: {p.get('paragraph_id', '')[:8]}...)")
    citations = "\n".join(citations_parts) if citations_parts else "No citations available."

    return answer, reasoning_trace, citations


def build_demo() -> gr.Blocks:
    """Build the Gradio interface."""
    with gr.Blocks(title="ViWiki-MHR Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# ViWiki-MHR: Vietnamese Multi-Hop Reasoning")
        gr.Markdown("Ask questions about Vietnamese Wikipedia. The system uses Knowledge Graph + Text Retrieval.")

        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label="Question (Vietnamese)",
                    placeholder="Ví dụ: Ai là người sáng lập Đảng Cộng sản Việt Nam?",
                    lines=2,
                )
                top_k_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Top-K Results")
                submit_btn = gr.Button("Ask", variant="primary")

            with gr.Column(scale=3):
                answer_output = gr.Textbox(label="Answer", lines=8)

        with gr.Row():
            with gr.Column():
                trace_output = gr.Markdown(label="Reasoning Trace")
            with gr.Column():
                citations_output = gr.Markdown(label="Citations")

        submit_btn.click(
            fn=process_query,
            inputs=[question_input, top_k_slider],
            outputs=[answer_output, trace_output, citations_output],
        )
        question_input.submit(
            fn=process_query,
            inputs=[question_input, top_k_slider],
            outputs=[answer_output, trace_output, citations_output],
        )

        gr.Markdown("---")
        gr.Markdown("Built with Gradio | ViWiki-MHR Project")

    return demo


def main() -> None:
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
