"""Speech-to-text helpers backed by faster-whisper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import threading

from src.config import settings

_MODEL = None
_MODEL_LOCK = threading.Lock()


@dataclass
class SpeechTranscriptionResult:
    text: str
    model: str
    language: str | None = None
    language_probability: float | None = None


def _resolve_model_source() -> tuple[str, str]:
    """Return model source and display name."""
    configured_path = (settings.stt_model_path or "").strip()
    if configured_path:
        model_path = Path(configured_path)
        if not model_path.exists():
            raise RuntimeError(f"Configured STT_MODEL_PATH does not exist: {configured_path}")
        return str(model_path), model_path.name or str(model_path)
    return settings.stt_model_size, settings.stt_model_size


def _import_whisper_model():
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Speech-to-text is unavailable because 'faster-whisper' is not installed."
        ) from exc
    return WhisperModel


def _get_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    with _MODEL_LOCK:
        if _MODEL is None:
            try:
                model_source, display_name = _resolve_model_source()
                WhisperModel = _import_whisper_model()
                _MODEL = (
                    WhisperModel(
                        model_source,
                        device=settings.stt_device,
                        compute_type=settings.stt_compute_type,
                    ),
                    display_name,
                )
            except RuntimeError:
                raise
            except Exception as exc:
                raise RuntimeError(f"Speech-to-text model initialization failed: {exc}") from exc
    return _MODEL


def _suffix_for_content_type(content_type: str | None) -> str:
    lowered = (content_type or "").lower()
    if "ogg" in lowered:
        return ".ogg"
    if "mp4" in lowered or "m4a" in lowered:
        return ".m4a"
    if "wav" in lowered:
        return ".wav"
    if "mpeg" in lowered or "mp3" in lowered:
        return ".mp3"
    return ".webm"


def transcribe_audio_bytes(
    audio_bytes: bytes,
    *,
    content_type: str | None = None,
) -> SpeechTranscriptionResult:
    """Transcribe uploaded audio bytes into plain text."""
    if not audio_bytes:
        raise RuntimeError("Audio payload is empty")

    model, model_name = _get_model()
    suffix = _suffix_for_content_type(content_type)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = Path(tmp.name)

    try:
        segments, info = model.transcribe(
            str(tmp_path),
            language=settings.stt_language,
            vad_filter=True,
        )
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
        return SpeechTranscriptionResult(
            text=text,
            model=model_name,
            language=getattr(info, "language", None),
            language_probability=getattr(info, "language_probability", None),
        )
    except Exception as exc:
        raise RuntimeError(f"Speech transcription failed: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)
