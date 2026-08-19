from __future__ import annotations

from pathlib import Path

import pytest

from src.infrastructure import speech_to_text as stt


def test_resolve_model_source_prefers_local_path(tmp_path: Path, monkeypatch) -> None:
    model_dir = tmp_path / "faster-whisper-small"
    model_dir.mkdir()

    monkeypatch.setattr(stt.settings, "stt_model_path", str(model_dir))
    monkeypatch.setattr(stt.settings, "stt_model_size", "small")

    source, display_name = stt._resolve_model_source()

    assert source == str(model_dir)
    assert display_name == "faster-whisper-small"


def test_resolve_model_source_raises_for_missing_path(monkeypatch) -> None:
    monkeypatch.setattr(stt.settings, "stt_model_path", "missing-local-model")

    with pytest.raises(RuntimeError, match="STT_MODEL_PATH does not exist"):
        stt._resolve_model_source()


def test_resolve_model_source_falls_back_to_model_size(monkeypatch) -> None:
    monkeypatch.setattr(stt.settings, "stt_model_path", None)
    monkeypatch.setattr(stt.settings, "stt_model_size", "small")

    source, display_name = stt._resolve_model_source()

    assert source == "small"
    assert display_name == "small"
