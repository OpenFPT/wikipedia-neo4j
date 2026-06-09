from __future__ import annotations

from dataclasses import dataclass

import src.ingestion.pipeline as ingest


@dataclass
class _CapturedUpsertCall:
    page_id: str
    title: str
    url: str
    text: str
    summary: str


class _FakeDataset:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def select(self, indexes) -> list[dict]:
        return [self._rows[i] for i in indexes]


def test_ingest_from_hf_streaming_respects_sample_size_and_progress(monkeypatch) -> None:
    rows = [
        {"id": "1", "title": "A", "url": "https://a", "text": "Alpha text"},
        {"id": "2", "title": "B", "url": "https://b", "text": "Beta text"},
        {"id": "3", "title": "C", "url": "https://c", "text": "Gamma text"},
    ]
    calls: list[_CapturedUpsertCall] = []
    progress: list[tuple[int, int | None, str]] = []

    def _fake_load_dataset(*_args, **kwargs):
        assert kwargs.get("streaming") is True
        return iter(rows)

    def _fake_upsert(*, page_id: str, title: str, url: str, text: str, summary: str):
        calls.append(_CapturedUpsertCall(page_id, title, url, text, summary))
        return ingest.IngestResult(
            topic=title,
            page_id=page_id,
            title=title,
            url=url,
            chunk_count=1,
            entity_count=1,
        )

    monkeypatch.setattr(ingest, "load_dataset", _fake_load_dataset)
    monkeypatch.setattr(ingest, "_upsert_page_from_text", _fake_upsert)

    results = ingest.ingest_from_hf(
        config_name="20231101.simple",
        split="train",
        sample_size=2,
        streaming=True,
        on_progress=lambda p, t, title: progress.append((p, t, title)),
    )

    assert len(results) == 2
    assert [c.title for c in calls] == ["A", "B"]
    assert progress == [(1, 2, "A"), (2, 2, "B")]


def test_ingest_from_hf_non_streaming_uses_selected_subset(monkeypatch) -> None:
    rows = [
        {"id": "1", "title": "A", "url": "https://a", "text": "Alpha"},
        {"id": "2", "title": "B", "url": "https://b", "text": "Beta"},
        {"id": "3", "title": "C", "url": "https://c", "text": "Gamma"},
    ]
    captured_titles: list[str] = []

    def _fake_load_dataset(*_args, **kwargs):
        assert "streaming" not in kwargs or kwargs.get("streaming") is not True
        return _FakeDataset(rows)

    def _fake_upsert(*, page_id: str, title: str, url: str, text: str, summary: str):
        captured_titles.append(title)
        return ingest.IngestResult(
            topic=title,
            page_id=page_id,
            title=title,
            url=url,
            chunk_count=1,
            entity_count=1,
        )

    monkeypatch.setattr(ingest, "load_dataset", _fake_load_dataset)
    monkeypatch.setattr(ingest, "_upsert_page_from_text", _fake_upsert)

    results = ingest.ingest_from_hf(sample_size=2, streaming=False)

    assert len(results) == 2
    assert captured_titles == ["A", "B"]


def test_ingest_from_hf_respects_stop_signal(monkeypatch) -> None:
    rows = [
        {"id": "1", "title": "A", "url": "https://a", "text": "Alpha text"},
        {"id": "2", "title": "B", "url": "https://b", "text": "Beta text"},
    ]
    processed_count = {"n": 0}

    def _fake_load_dataset(*_args, **_kwargs):
        return iter(rows)

    def _fake_upsert(*, page_id: str, title: str, url: str, text: str, summary: str):
        processed_count["n"] += 1
        return ingest.IngestResult(
            topic=title,
            page_id=page_id,
            title=title,
            url=url,
            chunk_count=1,
            entity_count=1,
        )

    monkeypatch.setattr(ingest, "load_dataset", _fake_load_dataset)
    monkeypatch.setattr(ingest, "_upsert_page_from_text", _fake_upsert)

    should_stop_calls = {"n": 0}

    def _should_stop() -> bool:
        should_stop_calls["n"] += 1
        return should_stop_calls["n"] > 1

    results = ingest.ingest_from_hf(sample_size=5, streaming=True, should_stop=_should_stop)

    assert len(results) == 1
    assert processed_count["n"] == 1


def test_ingest_from_hf_skips_malformed_rows_and_continues(monkeypatch) -> None:
    rows = [
        {"id": "1", "title": "Good", "url": "https://good", "text": "Useful text"},
        object(),  # Will fail dict(raw_row)
        {"id": "2", "title": "AlsoGood", "url": "https://good2", "text": "More text"},
    ]
    titles: list[str] = []

    def _fake_load_dataset(*_args, **_kwargs):
        return iter(rows)

    def _fake_upsert(*, page_id: str, title: str, url: str, text: str, summary: str):
        titles.append(title)
        return ingest.IngestResult(
            topic=title,
            page_id=page_id,
            title=title,
            url=url,
            chunk_count=1,
            entity_count=1,
        )

    monkeypatch.setattr(ingest, "load_dataset", _fake_load_dataset)
    monkeypatch.setattr(ingest, "_upsert_page_from_text", _fake_upsert)

    results = ingest.ingest_from_hf(sample_size=3, streaming=True)

    assert len(results) == 2
    assert titles == ["Good", "AlsoGood"]


def test_ingest_from_hf_generates_defaults_for_missing_fields(monkeypatch) -> None:
    rows = [{"title": "", "url": "", "text": "Body"}]
    captured: list[_CapturedUpsertCall] = []

    def _fake_load_dataset(*_args, **_kwargs):
        return iter(rows)

    def _fake_upsert(*, page_id: str, title: str, url: str, text: str, summary: str):
        captured.append(_CapturedUpsertCall(page_id, title, url, text, summary))
        return ingest.IngestResult(
            topic=title,
            page_id=page_id,
            title=title,
            url=url,
            chunk_count=1,
            entity_count=1,
        )

    monkeypatch.setattr(ingest, "load_dataset", _fake_load_dataset)
    monkeypatch.setattr(ingest, "_upsert_page_from_text", _fake_upsert)
    monkeypatch.setattr(ingest.uuid, "uuid4", lambda: "test-uuid")

    results = ingest.ingest_from_hf(sample_size=1, streaming=True)

    assert len(results) == 1
    assert captured[0].title == "untitled-test-uuid"
    assert captured[0].url == "https://example.org/untitled-test-uuid"
    assert captured[0].page_id
    assert captured[0].summary == "Body"
