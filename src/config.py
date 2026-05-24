"""Configuration and runtime validation helpers."""

from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "please-change-me"
    qdrant_url: str = "http://localhost:6333"
    ner_backend: str = "simple"
    openai_api_key: str | None = None

    gemini_key_file: str = ".gemini_key.txt"
    gemini_model_text: str = "gemini-2.0-flash"
    gemini_model_embedding: str = "gemini-embedding-001"
    embedding_backend: str = "local"
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    phonlp_model_dir: str = ".phonlp"
    vncorenlp_dir: str = ".vncorenlp"

    model_mode: str = "local"
    local_model_id: str = "Qwen/Qwen2.5-7B-Instruct"

    app_api_key: str | None = None
    rate_limit_per_minute: int = 120
    log_level: str = "INFO"
    json_logs: bool = False
    require_gemini_key_on_startup: bool = False

    multi_hop_expansion: bool = True
    rerank_min_score: float = 0.1

    min_text_length: int = 200
    ingest_batch_size: int = 100
    embed_batch_size: int = 50
    neo4j_page_batch: int = 5000
    neo4j_chunk_batch: int = 2000
    neo4j_entity_batch: int = 1000
    log_dir: str = "logs"

    @field_validator("rate_limit_per_minute")
    @classmethod
    def validate_rate_limit_per_minute(cls, value: int) -> int:
        """Ensure configured rate limit is positive."""
        if value < 1:
            raise ValueError("rate_limit_per_minute must be >= 1")
        return value

    @field_validator("embedding_backend")
    @classmethod
    def validate_embedding_backend(cls, value: str) -> str:
        backend = (value or "").strip().lower()
        if backend not in {"gemini", "local"}:
            raise ValueError("embedding_backend must be 'gemini' or 'local'")
        return backend

    @field_validator("ner_backend")
    @classmethod
    def validate_ner_backend(cls, value: str) -> str:
        backend = (value or "").strip().lower()
        if backend not in {"simple", "underthesea", "phonlp"}:
            raise ValueError("ner_backend must be 'simple', 'underthesea', or 'phonlp'")
        return backend

    @field_validator("model_mode")
    @classmethod
    def validate_model_mode(cls, value: str) -> str:
        mode = (value or "").strip().lower()
        if mode not in {"local", "api"}:
            raise ValueError("model_mode must be 'local' or 'api'")
        return mode

    @field_validator("neo4j_uri", "neo4j_username", "neo4j_password", mode="before")
    @classmethod
    def _strip_surrounding_quotes(cls, value: str) -> str:
        """Allow values in .env to be wrapped in quotes by stripping them."""
        if not isinstance(value, str):
            return value
        s = value.strip()
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            return s[1:-1]
        return s


settings = Settings()


def validate_runtime_settings() -> None:
    """Validate critical settings that should fail fast on startup."""
    if not settings.neo4j_uri.strip():
        raise RuntimeError("NEO4J_URI cannot be empty")
    if not settings.neo4j_username.strip():
        raise RuntimeError("NEO4J_USERNAME cannot be empty")
    if not settings.neo4j_password.strip():
        raise RuntimeError("NEO4J_PASSWORD cannot be empty")
    if settings.require_gemini_key_on_startup:
        key_path = Path(settings.gemini_key_file)
        if not key_path.exists():
            raise RuntimeError(
                f"Gemini key file not found: {settings.gemini_key_file}. "
                "Create it with your API key or disable REQUIRE_GEMINI_KEY_ON_STARTUP."
            )


def load_gemini_api_keys() -> list[str]:
    """Load one or more Gemini API keys from configured key file."""
    try:
        with open(settings.gemini_key_file, "r", encoding="utf-8") as f:
            raw = f.read()
            lines = [
                ln.strip()
                for ln in raw.splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            ]
            if not lines:
                raise RuntimeError(
                    f"Gemini key file is empty: {settings.gemini_key_file}. "
                    "Put only the API key content inside it."
                )
            return lines
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Gemini key file not found: {settings.gemini_key_file}. Create it with your API key."
        ) from exc


def load_gemini_api_key() -> str:
    """Load the first configured Gemini API key."""
    return load_gemini_api_keys()[0]
