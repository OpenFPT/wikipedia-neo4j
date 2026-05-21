"""Tests for KG walk extraction and question template engine."""

from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager

import src.dataset_gen as dg


class _FakeRecords:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


class _FakeSession:
    def __init__(self, rows=None):
        self._rows = rows or []

    def run(self, cypher, **params):
        return _FakeRecords(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def _mock_session(rows):
    @contextmanager
    def _s():
        yield _FakeSession(rows)
    return _s


class TestExtract2HopWalks:
    def test_extracts_walks(self, monkeypatch) -> None:
        rows = [
            {
                "page1": "Internet", "url1": "http://a",
                "page2": "Mạng máy tính", "url2": "http://b",
                "entity_name": "Vint Cerf", "entity_type": "Person",
                "chunk1_id": "c1", "chunk1_text": "text1",
                "chunk2_id": "c2", "chunk2_text": "text2",
            }
        ]
        monkeypatch.setattr(dg.neo4j_client, "session", _mock_session(rows))
        walks = dg.extract_2hop_walks(limit=10)
        assert len(walks) == 1
        assert walks[0].hops == 2
        assert walks[0].pages == ["Internet", "Mạng máy tính"]
        assert walks[0].entities[0]["name"] == "Vint Cerf"

    def test_deduplicates(self, monkeypatch) -> None:
        row = {
            "page1": "A", "url1": "http://a",
            "page2": "B", "url2": "http://b",
            "entity_name": "X", "entity_type": "Person",
            "chunk1_id": "c1", "chunk1_text": "t1",
            "chunk2_id": "c2", "chunk2_text": "t2",
        }
        monkeypatch.setattr(dg.neo4j_client, "session", _mock_session([row, row]))
        walks = dg.extract_2hop_walks(limit=10)
        assert len(walks) == 1


class TestExtract3HopWalks:
    def test_extracts_walks(self, monkeypatch) -> None:
        rows = [
            {
                "page1": "A", "page2": "B",
                "entity1_name": "E1", "entity1_type": "Person",
                "entity2_name": "E2", "entity2_type": "Location",
                "chunk1_id": "c1", "chunk1_text": "t1",
                "chunk3_id": "c3", "chunk3_text": "t3",
            }
        ]
        monkeypatch.setattr(dg.neo4j_client, "session", _mock_session(rows))
        walks = dg.extract_3hop_walks(limit=10)
        assert len(walks) == 1
        assert walks[0].hops == 3
        assert len(walks[0].entities) == 2


class TestGenerateQAFromWalks:
    def test_generates_2hop_qa(self, monkeypatch) -> None:
        walk = dg.KGWalk(
            walk_id="w1", hops=2,
            pages=["Internet", "Mạng"],
            entities=[{"name": "Vint Cerf", "type": "Person"}],
            path_description="Internet → Vint Cerf → Mạng",
            evidence_chunks=["c1", "c2"],
        )
        monkeypatch.setattr(dg.neo4j_client, "session", _mock_session([
            {"id": "c1", "text": "Vint Cerf là cha đẻ của Internet"},
            {"id": "c2", "text": "Mạng máy tính kết nối toàn cầu"},
        ]))
        qa = dg.generate_qa_from_walks([walk])
        assert len(qa) == 1
        assert "Vint Cerf" in qa[0].question
        assert qa[0].question_type == "2hop_person"

    def test_generates_3hop_qa(self, monkeypatch) -> None:
        walk = dg.KGWalk(
            walk_id="w2", hops=3,
            pages=["A", "B"],
            entities=[{"name": "E1", "type": "Person"}, {"name": "E2", "type": "Location"}],
            path_description="A → E1 → E2 → B",
            evidence_chunks=["c1", "c2"],
        )
        monkeypatch.setattr(dg.neo4j_client, "session", _mock_session([
            {"id": "c1", "text": "chunk 1 text"},
            {"id": "c2", "text": "chunk 2 text"},
        ]))
        qa = dg.generate_qa_from_walks([walk])
        assert len(qa) == 1
        assert qa[0].question_type == "3hop_bridge"


class TestExtractBrokenWalks:
    def test_extracts_broken(self, monkeypatch) -> None:
        rows = [
            {"page1": "Test", "entity_name": "Orphan", "entity_type": "Person", "chunk1_id": "c1"}
        ]
        monkeypatch.setattr(dg.neo4j_client, "session", _mock_session(rows))
        walks = dg.extract_broken_walks(limit=10)
        assert len(walks) == 1
        assert walks[0].pages[1] == "[NOT_INGESTED]"


class TestGenerateUnanswerableQA:
    def test_generates_unanswerable(self) -> None:
        walk = dg.KGWalk(
            walk_id="w3", hops=2,
            pages=["Test", "[NOT_INGESTED]"],
            entities=[{"name": "Orphan", "type": "Person"}],
            path_description="Test → Orphan → [broken]",
            evidence_chunks=["c1"],
        )
        qa = dg.generate_unanswerable_qa([walk])
        assert len(qa) == 1
        assert qa[0].question_type == "unanswerable"
        assert "Không thể trả lời" in qa[0].answer


class TestSaveDataset:
    def test_writes_jsonl(self) -> None:
        qa = dg.QAPair(
            qa_id="q1", question="Test?", answer="Answer",
            walk_id="w1", hops=2, question_type="2hop_person",
            evidence_chunk_ids=["c1"], source_pages=["A", "B"],
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name

        dg.save_dataset([qa], output_path=path)

        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["id"] == "q1"
        assert record["question"] == "Test?"
        assert record["metadata"]["hops"] == 2
