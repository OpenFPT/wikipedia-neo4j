from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.orchestration.agent_loop as agent_loop


_DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "viwikihop"
    / "data"
    / "intermediate"
    / "uit_viquad_seed42_8500.jsonl"
)


def _load_record(record_id: str) -> dict:
    with _DATASET_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("_id") == record_id:
                return record
    raise AssertionError(f"Missing viwikihop test record: {record_id}")


def _rows_from_context(record: dict) -> list[dict]:
    rows: list[dict] = []
    for context in record.get("context", []):
        title = context.get("title", "")
        for segment in context.get("segments", []):
            rows.append(
                {
                    "page_title": title,
                    "chunk_id": f"{record.get('_id')}:{segment.get('segment_id')}",
                    "chunk_text": segment.get("text", ""),
                }
            )
    return rows


@pytest.mark.parametrize(
    ("record_id", "expected_answer"),
    [
        ("uvq-000010", "rất căng thẳng và phức tạp"),
        (
            "uvq-020633",
            "Việt Nam lại bị chia cắt thành hai miền và cuốn vào một cuộc chiến tàn khốc mới theo Hiệp định Genève và việc Hoa Kỳ can thiệp vào miền Nam: Bắc Việt Nam (Việt Nam Dân chủ Cộng hòa) và Nam Việt Nam (Việt Nam Cộng hòa với hỗ trợ của Hoa Kỳ).",
        ),
    ],
)
def test_compose_fallback_answer_matches_real_viwikihop_answers(
    record_id: str,
    expected_answer: str,
) -> None:
    record = _load_record(record_id)

    answer = agent_loop._compose_fallback_answer(
        record["question"],
        _rows_from_context(record),
    )

    assert answer == expected_answer


def test_compose_fallback_answer_abstains_for_unanswerable_viwikihop_case() -> None:
    record = _load_record("uvq-020719")

    answer = agent_loop._compose_fallback_answer(
        record["question"],
        _rows_from_context(record),
    )

    assert "Không đủ thông tin" in answer
