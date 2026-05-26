"""Tests for config module — validators, key loading, runtime validation."""

from __future__ import annotations

import pytest

from src.config import Settings, load_gemini_api_keys, validate_runtime_settings, settings


class TestSettingsValidators:
    def test_invalid_model_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="model_mode"):
            Settings(model_mode="invalid")

    def test_valid_model_modes(self) -> None:
        s = Settings(model_mode="local")
        assert s.model_mode == "local"
        s2 = Settings(model_mode="api")
        assert s2.model_mode == "api"

    def test_invalid_embedding_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="embedding_backend"):
            Settings(embedding_backend="invalid")

    def test_invalid_ner_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="ner_backend"):
            Settings(ner_backend="invalid")

    def test_rate_limit_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="rate_limit_per_minute"):
            Settings(rate_limit_per_minute=0)

    def test_rate_limit_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="rate_limit_per_minute"):
            Settings(rate_limit_per_minute=-5)

    def test_valid_embedding_backends(self) -> None:
        s = Settings(embedding_backend="local")
        assert s.embedding_backend == "local"
        s2 = Settings(embedding_backend="gemini")
        assert s2.embedding_backend == "gemini"

    def test_valid_ner_backends(self) -> None:
        for backend in ("simple", "underthesea", "phonlp"):
            s = Settings(ner_backend=backend)
            assert s.ner_backend == backend

    def test_strip_surrounding_quotes_double(self) -> None:
        s = Settings(neo4j_uri='"bolt://localhost:7687"')
        assert s.neo4j_uri == "bolt://localhost:7687"

    def test_strip_surrounding_quotes_single(self) -> None:
        s = Settings(neo4j_password="'my-password'")
        assert s.neo4j_password == "my-password"

    def test_no_strip_when_no_quotes(self) -> None:
        s = Settings(neo4j_username="neo4j")
        assert s.neo4j_username == "neo4j"

    def test_strip_quotes_non_string_passthrough(self) -> None:
        from src.config import Settings
        # Directly test the validator method
        result = Settings._strip_surrounding_quotes(42)  # type: ignore[arg-type]
        assert result == 42

    def test_videberta_ner_backend_valid(self) -> None:
        s = Settings(ner_backend="videberta")
        assert s.ner_backend == "videberta"

    def test_phobert_ner_backend_valid(self) -> None:
        s = Settings(ner_backend="phobert")
        assert s.ner_backend == "phobert"

    def test_wikilink_ner_backend_valid(self) -> None:
        s = Settings(ner_backend="wikilink")
        assert s.ner_backend == "wikilink"


class TestValidateRuntimeSettings:
    def test_empty_neo4j_uri_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "neo4j_uri", "   ")
        with pytest.raises(RuntimeError, match="NEO4J_URI"):
            validate_runtime_settings()

    def test_empty_neo4j_username_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "neo4j_uri", "bolt://localhost:7687")
        monkeypatch.setattr(settings, "neo4j_username", "  ")
        with pytest.raises(RuntimeError, match="NEO4J_USERNAME"):
            validate_runtime_settings()

    def test_empty_neo4j_password_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "neo4j_uri", "bolt://localhost:7687")
        monkeypatch.setattr(settings, "neo4j_username", "neo4j")
        monkeypatch.setattr(settings, "neo4j_password", "")
        with pytest.raises(RuntimeError, match="NEO4J_PASSWORD"):
            validate_runtime_settings()

    def test_missing_gemini_key_file_raises_when_required(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "neo4j_uri", "bolt://localhost:7687")
        monkeypatch.setattr(settings, "neo4j_username", "neo4j")
        monkeypatch.setattr(settings, "neo4j_password", "pass")
        monkeypatch.setattr(settings, "require_gemini_key_on_startup", True)
        monkeypatch.setattr(settings, "gemini_key_file", "/nonexistent/path.txt")
        with pytest.raises(RuntimeError, match="key file not found"):
            validate_runtime_settings()


class TestLoadGeminiApiKeys:
    def test_loads_multiple_keys(self, tmp_path) -> None:
        key_file = tmp_path / "keys.txt"
        key_file.write_text("key1\nkey2\n# comment\n\nkey3\n")

        from unittest.mock import patch
        with patch.object(settings, "gemini_key_file", str(key_file)):
            keys = load_gemini_api_keys()

        assert keys == ["key1", "key2", "key3"]

    def test_empty_file_raises(self, tmp_path) -> None:
        key_file = tmp_path / "empty.txt"
        key_file.write_text("# only comments\n\n")

        from unittest.mock import patch
        with patch.object(settings, "gemini_key_file", str(key_file)):
            with pytest.raises(RuntimeError, match="empty"):
                load_gemini_api_keys()

    def test_missing_file_raises(self) -> None:
        from unittest.mock import patch
        with patch.object(settings, "gemini_key_file", "/nonexistent/keys.txt"):
            with pytest.raises(RuntimeError, match="not found"):
                load_gemini_api_keys()

    def test_load_gemini_api_key_returns_first(self, tmp_path) -> None:
        from unittest.mock import patch
        from src.config import load_gemini_api_key

        key_file = tmp_path / "keys.txt"
        key_file.write_text("first-key\nsecond-key\n")
        with patch.object(settings, "gemini_key_file", str(key_file)):
            assert load_gemini_api_key() == "first-key"


class TestLoggingUtils:
    def test_configure_logging_json_with_file(self, tmp_path, monkeypatch) -> None:
        import src.logging_utils as lu
        monkeypatch.setattr(lu, "_CONFIGURED", False)
        lu.configure_logging(
            level_name="DEBUG",
            json_logs=True,
            log_dir=str(tmp_path),
            task_name="test_task",
        )
        monkeypatch.setattr(lu, "_CONFIGURED", False)

    def test_set_and_reset_request_id(self) -> None:
        from src.logging_utils import set_request_id, reset_request_id
        token = set_request_id("req-123")
        reset_request_id(token)

    def test_json_formatter_with_exception(self) -> None:
        import logging
        from src.logging_utils import JsonFormatter
        fmt = JsonFormatter()
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="", lineno=0,
            msg="fail", args=None, exc_info=None,
        )
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            record.exc_info = sys.exc_info()
        output = fmt.format(record)
        assert "boom" in output
