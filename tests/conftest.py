from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _force_api_mode(monkeypatch):
    """Force model_mode=api in tests to avoid loading the local quantized model."""
    from src.config import settings

    monkeypatch.setattr(settings, "model_mode", "api")


@pytest.fixture(autouse=True)
def _mock_reranker(monkeypatch):
    """Bypass cross-encoder model loading in tests."""
    import src.reranker as reranker_mod

    def _passthrough(_query, documents, text_key="chunk_text", top_k=5):
        return documents[:top_k]

    monkeypatch.setattr(reranker_mod, "rerank", _passthrough)
