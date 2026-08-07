"""Gradio chat demo backed by the GraphRAG pipeline (Neo4j + Qdrant + LLM)."""

from __future__ import annotations

import sys
from pathlib import Path
import os

import gradio as gr

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.retrieval.hybrid import query_graph


def _answer(question: str) -> str:
    # Keep retrieval grounded: use only the latest user message.
    result = query_graph(question, top_k=4)
    # De-dup citations by page_url for a cleaner UI.
    seen: set[str] = set()
    deduped = []
    for c in (result.citations or []):
        url = str(c.get("page_url", "") or "")
        if not url or url in seen:
            continue
        seen.add(url)
        deduped.append(c)

    citations = "\n".join(
        f"- {c.get('page_title','')} ({c.get('page_url','')})" for c in deduped
    ) or "No citations."
    return f"{result.answer}\n\nCitations:\n{citations}"


def _normalize_history(chat_history):
    # Gradio Chatbot expects list[{"role","content"}], but legacy tuple history can
    # slip through depending on wiring/version. Normalize to avoid postprocess errors.
    if not chat_history:
        return []

    normalized: list[dict] = []
    for item in chat_history:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if role in ("user", "assistant") and content is not None:
                normalized.append({"role": role, "content": str(content)})
            continue

        if isinstance(item, (tuple, list)) and len(item) == 2:
            user_msg, bot_msg = item
            if user_msg is not None and str(user_msg).strip():
                normalized.append({"role": "user", "content": str(user_msg)})
            if bot_msg is not None and str(bot_msg).strip():
                normalized.append({"role": "assistant", "content": str(bot_msg)})
            continue

    return normalized


def _on_submit(message: str, chat_history):
    message = (message or "").strip()
    if not message:
        return "", chat_history
    resp = _answer(message)
    chat_history = _normalize_history(chat_history) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": resp},
    ]
    return "", chat_history


def main() -> None:
    with gr.Blocks(title="Wikipedia Neo4j GraphRAG Chat") as demo:
        gr.Markdown("# Wikipedia Neo4j GraphRAG Chat")
        gr.Markdown(
            "Hỏi bằng tiếng Việt/English đều được. Câu trả lời luôn kèm trích dẫn từ Wikipedia (Neo4j)."
        )
        gr.Markdown("UI build: `dedup-citations-v2`")

        # Explicitly use messages format to match our {role,content} history payload.
        chatbot = gr.Chatbot(label="Chat", height=520, type="messages")
        msg = gr.Textbox(
            label="Câu hỏi",
            placeholder="Ví dụ: Hồ Chí Minh là ai?",
            lines=2,
        )
        send = gr.Button("Gửi", variant="primary")

        send.click(_on_submit, inputs=[msg, chatbot], outputs=[msg, chatbot])
        msg.submit(_on_submit, inputs=[msg, chatbot], outputs=[msg, chatbot])

    # Prefer 7860, but fall back if something else is already bound.
    base_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    last_err: Exception | None = None
    for port in range(base_port, base_port + 11):
        try:
            demo.launch(server_name="0.0.0.0", server_port=port, share=False)
            return
        except OSError as e:
            last_err = e
            continue
    raise last_err or RuntimeError("Failed to launch Gradio server.")


if __name__ == "__main__":
    main()
