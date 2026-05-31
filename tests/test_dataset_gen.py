"""Tests for KG walk extraction and question template engine."""

from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager

import src.ingestion.dataset_gen as dg


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


class TestIsValidRewrite:
    def test_valid_rewrite(self) -> None:
        assert dg._is_valid_rewrite("Câu hỏi gốc?", "Câu hỏi đã viết lại?")

    def test_empty_rewrite(self) -> None:
        assert not dg._is_valid_rewrite("Câu hỏi gốc?", "")

    def test_too_short(self) -> None:
        assert not dg._is_valid_rewrite("Câu hỏi gốc?", "Ngắn?")

    def test_same_as_original(self) -> None:
        assert not dg._is_valid_rewrite("Câu hỏi gốc?", "Câu hỏi gốc?")

    def test_too_long(self) -> None:
        original = "Câu hỏi ngắn?"
        rewritten = "A" * 200 + "?"
        assert not dg._is_valid_rewrite(original, rewritten)


class TestCheckWellFormed:
    def test_valid_qa(self) -> None:
        qa = dg.QAPair(
            qa_id="q1", question="Ai là tổng thống đầu tiên?",
            answer="Hồ Chí Minh là chủ tịch đầu tiên.",
            walk_id="w1", hops=2, question_type="2hop_person",
            evidence_chunk_ids=["c1"], source_pages=["A"],
        )
        assert dg._check_well_formed(qa)

    def test_short_question(self) -> None:
        qa = dg.QAPair(
            qa_id="q1", question="Ai?",
            answer="Hồ Chí Minh là chủ tịch đầu tiên.",
            walk_id="w1", hops=2, question_type="2hop_person",
            evidence_chunk_ids=["c1"], source_pages=["A"],
        )
        assert not dg._check_well_formed(qa)

    def test_no_question_mark(self) -> None:
        qa = dg.QAPair(
            qa_id="q1", question="Đây không phải câu hỏi",
            answer="Hồ Chí Minh là chủ tịch đầu tiên.",
            walk_id="w1", hops=2, question_type="2hop_person",
            evidence_chunk_ids=["c1"], source_pages=["A"],
        )
        assert not dg._check_well_formed(qa)


class TestCheckNoDuplicate:
    def test_unique(self) -> None:
        qa = dg.QAPair(
            qa_id="q1", question="Câu hỏi duy nhất?",
            answer="answer", walk_id="w1", hops=2,
            question_type="2hop_person", evidence_chunk_ids=["c1"], source_pages=["A"],
        )
        seen: set[str] = set()
        assert dg._check_no_duplicate(qa, seen)

    def test_duplicate(self) -> None:
        qa = dg.QAPair(
            qa_id="q1", question="Câu hỏi trùng?",
            answer="answer", walk_id="w1", hops=2,
            question_type="2hop_person", evidence_chunk_ids=["c1"], source_pages=["A"],
        )
        seen: set[str] = {"câu hỏi trùng?"}
        assert not dg._check_no_duplicate(qa, seen)


class TestRunQCPipeline:
    def test_filters_bad_qa(self, monkeypatch) -> None:
        good = dg.QAPair(
            qa_id="q1", question="Ai là người sáng lập Internet?",
            answer="Vint Cerf được coi là cha đẻ của Internet.",
            walk_id="w1", hops=2, question_type="2hop_person",
            evidence_chunk_ids=["c1"], source_pages=["A", "B"],
        )
        bad_short = dg.QAPair(
            qa_id="q2", question="Ai?",
            answer="X", walk_id="w2", hops=2,
            question_type="2hop_person", evidence_chunk_ids=["c2"], source_pages=["C"],
        )
        monkeypatch.setattr(dg.neo4j_client, "session", _mock_session([
            {"id": "c1", "text": "Vint Cerf là cha đẻ của Internet và mạng máy tính toàn cầu"},
        ]))
        result = dg.run_qc_pipeline([good, bad_short])
        assert len(result) == 1
        assert result[0].qa_id == "q1"


class TestRewriteQuestionsWithLLM:
    def test_rewrites_questions(self, monkeypatch) -> None:
        qa = dg.QAPair(
            qa_id="q1", question="Ai là người sáng lập Internet?",
            answer="Vint Cerf", walk_id="w1", hops=2,
            question_type="2hop_person", evidence_chunk_ids=["c1"], source_pages=["A"],
        )

        def _fake_chat(messages, max_new_tokens=128, temperature=0.3):
            return "Người sáng lập Internet là ai?"

        import src.infrastructure.local_llm
        monkeypatch.setattr(src.infrastructure.local_llm, "chat", _fake_chat)

        result = dg.rewrite_questions_with_llm([qa], batch_size=1)
        assert result[0].question == "Người sáng lập Internet là ai?"

    def test_skips_unanswerable(self, monkeypatch) -> None:
        qa = dg.QAPair(
            qa_id="q1", question="Ai là người sáng lập XYZ?",
            answer="Không có thông tin", walk_id="w1", hops=2,
            question_type="unanswerable", evidence_chunk_ids=["c1"], source_pages=["A"],
        )

        call_count = [0]

        def _fake_chat(messages, max_new_tokens=128, temperature=0.3):
            call_count[0] += 1
            return "rewritten"

        import src.infrastructure.local_llm
        monkeypatch.setattr(src.infrastructure.local_llm, "chat", _fake_chat)

        result = dg.rewrite_questions_with_llm([qa], batch_size=1)
        assert call_count[0] == 0
        assert result[0].question == "Ai là người sáng lập XYZ?"

    def test_skips_when_not_local(self, monkeypatch) -> None:
        qa = dg.QAPair(
            qa_id="q1", question="Ai là người sáng lập Internet?",
            answer="Vint Cerf", walk_id="w1", hops=2,
            question_type="2hop_person", evidence_chunk_ids=["c1"], source_pages=["A"],
        )
        result = dg.rewrite_questions_with_llm([qa], use_local_model=False)
        assert result[0].question == "Ai là người sáng lập Internet?"

    def test_handles_chat_error(self, monkeypatch) -> None:
        qa = dg.QAPair(
            qa_id="q1", question="Ai là người sáng lập Internet?",
            answer="Vint Cerf", walk_id="w1", hops=2,
            question_type="2hop_person", evidence_chunk_ids=["c1"], source_pages=["A"],
        )

        def _fail_chat(messages, max_new_tokens=128, temperature=0.3):
            raise RuntimeError("model error")

        import src.infrastructure.local_llm
        monkeypatch.setattr(src.infrastructure.local_llm, "chat", _fail_chat)

        result = dg.rewrite_questions_with_llm([qa], batch_size=1)
        assert result[0].question == "Ai là người sáng lập Internet?"


class TestIsValidRewriteExtra:
    def test_appends_question_mark(self) -> None:
        assert dg._is_valid_rewrite("Ai là người sáng lập?", "Người sáng lập là ai")


class TestCheckGroundingExtra:
    def test_unanswerable_always_passes(self) -> None:
        qa = dg.QAPair(
            qa_id="q1", question="Ai là người sáng lập XYZ?",
            answer="Không có thông tin", walk_id="w1", hops=2,
            question_type="unanswerable", evidence_chunk_ids=[], source_pages=["A"],
        )
        assert dg._check_grounding(qa, {}) is True

    def test_grounded_answer(self) -> None:
        qa = dg.QAPair(
            qa_id="q1", question="Ai là người sáng lập Internet?",
            answer="Vint Cerf là cha đẻ của Internet",
            walk_id="w1", hops=2, question_type="2hop_person",
            evidence_chunk_ids=["c1"], source_pages=["A"],
        )
        chunks = {"c1": "Vint Cerf là cha đẻ của Internet và mạng máy tính toàn cầu"}
        assert dg._check_grounding(qa, chunks) is True

    def test_ungrounded_answer(self) -> None:
        qa = dg.QAPair(
            qa_id="q1", question="Ai là người sáng lập Internet?",
            answer="Xyz abc def ghij",
            walk_id="w1", hops=2, question_type="2hop_person",
            evidence_chunk_ids=["c1"], source_pages=["A"],
        )
        chunks = {"c1": "Vint Cerf là cha đẻ của Internet"}
        assert dg._check_grounding(qa, chunks) is False


class TestExtractKeyTerms:
    def test_extracts_capitalized_terms(self) -> None:
        terms = dg._extract_key_terms("Vint Cerf là cha đẻ của Internet")
        assert "Vint Cerf" in terms or "Internet" in terms

    def test_empty_text(self) -> None:
        terms = dg._extract_key_terms("")
        assert terms == []


class TestCheckWellFormedExtra:
    def test_question_equals_answer(self) -> None:
        qa = dg.QAPair(
            qa_id="q1", question="Ai là người sáng lập Internet?",
            answer="Ai là người sáng lập Internet?",
            walk_id="w1", hops=2, question_type="2hop_person",
            evidence_chunk_ids=["c1"], source_pages=["A"],
        )
        assert dg._check_well_formed(qa) is False


class TestRunQCPipelineExtra:
    def test_rejects_grounding_and_duplicate(self, monkeypatch) -> None:
        good = dg.QAPair(
            qa_id="q1", question="Ai là người sáng lập Internet?",
            answer="Vint Cerf là cha đẻ của Internet.",
            walk_id="w1", hops=2, question_type="2hop_person",
            evidence_chunk_ids=["c1"], source_pages=["A"],
        )
        duplicate = dg.QAPair(
            qa_id="q2", question="Ai là người sáng lập Internet?",
            answer="Vint Cerf là cha đẻ của Internet.",
            walk_id="w2", hops=2, question_type="2hop_person",
            evidence_chunk_ids=["c1"], source_pages=["A"],
        )
        ungrounded = dg.QAPair(
            qa_id="q3", question="Thủ đô của Pháp là gì?",
            answer="Xyz abc short",
            walk_id="w3", hops=2, question_type="2hop_location",
            evidence_chunk_ids=["c2"], source_pages=["B"],
        )
        monkeypatch.setattr(dg.neo4j_client, "session", _mock_session([
            {"id": "c1", "text": "Vint Cerf là cha đẻ của Internet và mạng máy tính toàn cầu"},
            {"id": "c2", "text": "Paris là thủ đô của nước Pháp"},
        ]))
        result = dg.run_qc_pipeline([good, duplicate, ungrounded])
        assert len(result) == 1
        assert result[0].qa_id == "q1"


class TestGenerateDataset:
    def test_full_pipeline(self, monkeypatch, tmp_path) -> None:
        walks_2hop = [
            dg.KGWalk(
                walk_id="w1", hops=2, pages=["A", "B"],
                entities=[{"name": "E1", "type": "Person"}],
                path_description="A → E1 → B",
                evidence_chunks=["c1"],
            )
        ]
        walks_3hop = []
        broken_walks = []

        monkeypatch.setattr(dg, "extract_2hop_walks", lambda limit: walks_2hop)
        monkeypatch.setattr(dg, "extract_3hop_walks", lambda limit: walks_3hop)
        monkeypatch.setattr(dg, "extract_broken_walks", lambda limit: broken_walks)

        qa_list = [
            dg.QAPair(
                qa_id="q1", question="Ai là người sáng lập Internet?",
                answer="Vint Cerf là cha đẻ của Internet.",
                walk_id="w1", hops=2, question_type="2hop_person",
                evidence_chunk_ids=["c1"], source_pages=["A", "B"],
            )
        ]
        monkeypatch.setattr(dg, "generate_qa_from_walks", lambda w: qa_list)
        monkeypatch.setattr(dg, "generate_unanswerable_qa", lambda w: [])

        out = str(tmp_path / "output.jsonl")
        stats = dg.generate_dataset(
            two_hop_limit=10, three_hop_limit=10, broken_limit=10,
            output_path=out, rewrite=False, qc=False,
        )
        assert stats["total"] == 1
        assert stats["2hop"] == 1
        assert stats["output_path"] == out

    def test_with_rewrite_and_qc(self, monkeypatch, tmp_path) -> None:
        walks_2hop = []
        monkeypatch.setattr(dg, "extract_2hop_walks", lambda limit: walks_2hop)
        monkeypatch.setattr(dg, "extract_3hop_walks", lambda limit: [])
        monkeypatch.setattr(dg, "extract_broken_walks", lambda limit: [])

        qa_list = [
            dg.QAPair(
                qa_id="q1", question="Ai là người sáng lập Internet?",
                answer="Vint Cerf là cha đẻ của Internet.",
                walk_id="w1", hops=2, question_type="2hop_person",
                evidence_chunk_ids=["c1"], source_pages=["A"],
            )
        ]
        monkeypatch.setattr(dg, "generate_qa_from_walks", lambda w: qa_list)
        monkeypatch.setattr(dg, "generate_unanswerable_qa", lambda w: [])
        monkeypatch.setattr(dg, "rewrite_questions_with_llm", lambda qa: qa)
        monkeypatch.setattr(dg, "run_qc_pipeline", lambda qa: qa)

        out = str(tmp_path / "output2.jsonl")
        stats = dg.generate_dataset(
            two_hop_limit=5, three_hop_limit=5, broken_limit=5,
            output_path=out, rewrite=True, qc=True,
        )
        assert stats["total"] == 1


class TestGenerateAnswerFromChunks:
    def test_fallback_when_no_chunks(self) -> None:
        walk = dg.KGWalk(
            walk_id="w1", hops=2, pages=["A"],
            entities=[{"name": "E1", "type": "Person"}],
            path_description="A → E1",
            evidence_chunks=["missing_chunk"],
        )
        answer = dg._generate_answer_from_chunks(walk, {})
        assert "E1" in answer
