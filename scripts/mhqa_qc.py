"""Stage 3: Quality control pipeline for generated MHQA pairs (5-layer filtering)."""

from __future__ import annotations

import argparse
import json
import re
import signal
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.logging_utils import configure_logging, get_logger

configure_logging(settings.log_level, settings.json_logs, log_dir=settings.log_dir, task_name="mhqa_qc")
logger = get_logger(__name__)

_STOP = False


def _handle_sigint(sig, frame):
    global _STOP
    _STOP = True
    print("\nGraceful shutdown requested, finishing current batch...")


# -- Normalization utilities --


def _normalize_text(text: str) -> str:
    """Normalize Vietnamese text: NFC, lowercase, strip extra whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _strip_punctuation(text: str) -> str:
    """Remove punctuation for comparison."""
    return re.sub(r"[^\w\s]", "", text)


# -- Layer 1: Well-formedness --


def check_well_formedness(record: dict) -> tuple[bool, str]:
    """Layer 1: Check structural validity of the QA record."""
    question = record.get("question", "")
    answer = record.get("answer", "")
    decomposition = record.get("decomposition", [])
    supporting_facts = record.get("supporting_facts", [])
    context = record.get("context", [])
    num_hops = record.get("num_hops", 2)

    # Question length and ending
    if len(question) < 15:
        return False, "question_too_short"
    if not question.strip().endswith("?"):
        return False, "question_no_question_mark"

    # Answer length
    if len(answer) < 2:
        return False, "answer_too_short"
    if len(answer) > 50:
        return False, "answer_too_long"

    # Decomposition count must match num_hops
    if len(decomposition) != num_hops:
        return False, f"decomposition_count_mismatch (got {len(decomposition)}, expected {num_hops})"

    # Supporting facts must reference valid sent_ids
    if not supporting_facts:
        return False, "no_supporting_facts"

    for sf in supporting_facts:
        if "title" not in sf or "sent_id" not in sf:
            return False, "supporting_fact_missing_fields"
        # Find matching context page
        matching_pages = [c for c in context if c.get("title") == sf["title"]]
        if not matching_pages:
            return False, f"supporting_fact_title_not_in_context: {sf['title']}"
        page_ctx = matching_pages[0]
        if sf["sent_id"] >= len(page_ctx.get("sentences", [])):
            return False, f"sent_id_out_of_range: {sf['sent_id']} >= {len(page_ctx.get('sentences', []))}"

    # Required fields
    required = ["question", "answer", "supporting_facts", "context", "type", "num_hops", "decomposition"]
    for field in required:
        if field not in record:
            return False, f"missing_field: {field}"

    return True, "ok"


# -- Layer 2: Grounding --


def check_grounding(record: dict) -> tuple[bool, str]:
    """Layer 2: Check that answer and sub-answers are grounded in context."""
    answer = _normalize_text(record.get("answer", ""))
    context = record.get("context", [])
    decomposition = record.get("decomposition", [])

    # Answer must appear in at least one context sentence
    answer_found = False
    for page_ctx in context:
        for sent in page_ctx.get("sentences", []):
            if answer in _normalize_text(sent):
                answer_found = True
                break
        if answer_found:
            break

    if not answer_found:
        return False, "answer_not_grounded"

    # Each sub_answer must be grounded in its corresponding page
    for i, decomp in enumerate(decomposition):
        sub_answer = _normalize_text(decomp.get("sub_answer", ""))
        page_id = decomp.get("page_id")

        # Find page context by page_id
        target_pages = [c for c in context if str(c.get("page_id")) == str(page_id)]
        if not target_pages:
            # Try matching by index
            if i < len(context):
                target_pages = [context[i]]
            else:
                return False, f"decomp_{i}_page_not_found"

        sub_found = False
        for page_ctx in target_pages:
            for sent in page_ctx.get("sentences", []):
                if sub_answer in _normalize_text(sent):
                    sub_found = True
                    break
            if sub_found:
                break

        if not sub_found:
            return False, f"sub_answer_{i}_not_grounded"

    return True, "ok"


# -- Layer 3: Deduplication --


def check_dedup(
    record: dict,
    existing_questions: list[str],
    embeddings_cache: dict | None = None,
) -> tuple[bool, str]:
    """Layer 3: Check for duplicate questions via normalized text similarity."""
    question_norm = _strip_punctuation(_normalize_text(record.get("question", "")))

    # Exact normalized match
    for existing in existing_questions:
        existing_norm = _strip_punctuation(_normalize_text(existing))
        if question_norm == existing_norm:
            return False, "exact_duplicate"

        # Simple word-level Jaccard similarity as lightweight check
        q_words = set(question_norm.split())
        e_words = set(existing_norm.split())
        if q_words and e_words:
            intersection = len(q_words & e_words)
            union = len(q_words | e_words)
            jaccard = intersection / union if union > 0 else 0
            if jaccard > 0.85:
                return False, f"near_duplicate (jaccard={jaccard:.3f})"

    return True, "ok"


# -- Layer 4: Answerability (multi-hop check) --


def check_answerability(record: dict) -> tuple[bool, str]:
    """Layer 4: Check that the question truly requires both pages."""
    answer = _normalize_text(record.get("answer", ""))
    context = record.get("context", [])

    if len(context) < 2:
        return False, "single_page_context"

    # Check if answer is derivable from a single page alone
    pages_with_answer = []
    for i, page_ctx in enumerate(context):
        for sent in page_ctx.get("sentences", []):
            if answer in _normalize_text(sent):
                pages_with_answer.append(i)
                break

    # If answer appears in exactly one page, check if question references the other
    if len(pages_with_answer) == 1:
        answer_page_idx = pages_with_answer[0]
        other_page_idx = 1 - answer_page_idx
        other_page = context[other_page_idx]

        # Heuristic: does the question contain words unique to the other page?
        question_norm = _normalize_text(record.get("question", ""))
        other_text = " ".join(_normalize_text(s) for s in other_page.get("sentences", []))
        answer_text = " ".join(
            _normalize_text(s) for s in context[answer_page_idx].get("sentences", [])
        )

        # Get words unique to other page (not in answer page)
        other_words = set(other_text.split()) - set(answer_text.split())
        question_words = set(question_norm.split())

        # If question has no meaningful overlap with the other page, it's single-hop
        overlap = question_words & other_words
        meaningful_overlap = {w for w in overlap if len(w) > 2}

        if not meaningful_overlap:
            return False, "answerable_from_single_page"

    return True, "ok"


# -- Layer 5: Schema Validation --


def check_schema(record: dict) -> tuple[bool, str]:
    """Layer 5: Validate against template.json structure."""
    # Type check
    if record.get("type") not in ("bridge", "comparison"):
        return False, f"invalid_type: {record.get('type')}"

    # Num hops
    num_hops = record.get("num_hops")
    if not isinstance(num_hops, int) or num_hops < 2 or num_hops > 3:
        return False, f"invalid_num_hops: {num_hops}"

    # Supporting facts structure
    for sf in record.get("supporting_facts", []):
        if not isinstance(sf.get("sent_id"), int):
            return False, "supporting_fact_sent_id_not_int"
        if not sf.get("title"):
            return False, "supporting_fact_missing_title"

    # Decomposition structure
    for decomp in record.get("decomposition", []):
        if not decomp.get("sub_question"):
            return False, "decomposition_missing_sub_question"
        if not decomp.get("sub_answer"):
            return False, "decomposition_missing_sub_answer"

    # Context structure
    for ctx in record.get("context", []):
        if not ctx.get("title"):
            return False, "context_missing_title"
        if not isinstance(ctx.get("sentences"), list):
            return False, "context_sentences_not_list"

    return True, "ok"


# -- Main QC Pipeline --


def run_qc(
    input_path: Path,
    verified_path: Path,
    rejected_path: Path,
    report_path: Path,
    seeds_path: Path,
):
    """Run the full 5-layer QC pipeline."""
    # Load raw questions
    raw_records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_records.append(json.loads(line))
    logger.info(f"Loaded {len(raw_records)} raw records from {input_path}")

    # Load seed questions for dedup checking
    seed_questions: list[str] = []
    if seeds_path.exists():
        with open(seeds_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    seed = json.loads(line)
                    seed_questions.append(seed.get("question", ""))

    # Stats tracking
    stats: dict = {
        "total_input": len(raw_records),
        "layer1_well_formedness": {"pass": 0, "fail": 0, "reasons": {}},
        "layer2_grounding": {"pass": 0, "fail": 0, "reasons": {}},
        "layer3_dedup": {"pass": 0, "fail": 0, "reasons": {}},
        "layer4_answerability": {"pass": 0, "fail": 0, "reasons": {}},
        "layer5_schema": {"pass": 0, "fail": 0, "reasons": {}},
        "total_verified": 0,
        "total_rejected": 0,
    }

    verified = []
    rejected = []
    accepted_questions: list[str] = list(seed_questions)

    for record in raw_records:
        if _STOP:
            logger.warning("Interrupted during QC processing")
            break

        reject_reason = None

        # Layer 1
        ok, reason = check_well_formedness(record)
        if not ok:
            stats["layer1_well_formedness"]["fail"] += 1
            stats["layer1_well_formedness"]["reasons"][reason] = (
                stats["layer1_well_formedness"]["reasons"].get(reason, 0) + 1
            )
            reject_reason = f"L1:{reason}"
        else:
            stats["layer1_well_formedness"]["pass"] += 1

        # Layer 2
        if not reject_reason:
            ok, reason = check_grounding(record)
            if not ok:
                stats["layer2_grounding"]["fail"] += 1
                stats["layer2_grounding"]["reasons"][reason] = (
                    stats["layer2_grounding"]["reasons"].get(reason, 0) + 1
                )
                reject_reason = f"L2:{reason}"
            else:
                stats["layer2_grounding"]["pass"] += 1

        # Layer 3
        if not reject_reason:
            ok, reason = check_dedup(record, accepted_questions)
            if not ok:
                stats["layer3_dedup"]["fail"] += 1
                stats["layer3_dedup"]["reasons"][reason] = (
                    stats["layer3_dedup"]["reasons"].get(reason, 0) + 1
                )
                reject_reason = f"L3:{reason}"
            else:
                stats["layer3_dedup"]["pass"] += 1

        # Layer 4
        if not reject_reason:
            ok, reason = check_answerability(record)
            if not ok:
                stats["layer4_answerability"]["fail"] += 1
                stats["layer4_answerability"]["reasons"][reason] = (
                    stats["layer4_answerability"]["reasons"].get(reason, 0) + 1
                )
                reject_reason = f"L4:{reason}"
            else:
                stats["layer4_answerability"]["pass"] += 1

        # Layer 5
        if not reject_reason:
            ok, reason = check_schema(record)
            if not ok:
                stats["layer5_schema"]["fail"] += 1
                stats["layer5_schema"]["reasons"][reason] = (
                    stats["layer5_schema"]["reasons"].get(reason, 0) + 1
                )
                reject_reason = f"L5:{reason}"
            else:
                stats["layer5_schema"]["pass"] += 1

        if reject_reason:
            record["_reject_reason"] = reject_reason
            rejected.append(record)
        else:
            verified.append(record)
            accepted_questions.append(record.get("question", ""))

    # Assign sequential IDs to verified records
    for i, record in enumerate(verified):
        record["_id"] = f"mhqa-{i:05d}"
        record.pop("walk_id", None)

    stats["total_verified"] = len(verified)
    stats["total_rejected"] = len(rejected)
    stats["pass_rate"] = f"{len(verified) / max(len(raw_records), 1) * 100:.1f}%"

    # Write outputs
    verified_path.parent.mkdir(parents=True, exist_ok=True)

    with open(verified_path, "w", encoding="utf-8") as f:
        for record in verified:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(rejected_path, "w", encoding="utf-8") as f:
        for record in rejected:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    report_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    logger.info(f"QC complete: {len(verified)} verified, {len(rejected)} rejected ({stats['pass_rate']} pass)")
    logger.info(f"Verified: {verified_path}")
    logger.info(f"Rejected: {rejected_path}")
    logger.info(f"Report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Run 5-layer QC on generated MHQA pairs")
    parser.add_argument("--input", type=str, default="data/mhqa/raw_questions.jsonl", help="Input JSONL")
    parser.add_argument("--verified", type=str, default="data/mhqa/verified.jsonl", help="Output verified JSONL")
    parser.add_argument("--rejected", type=str, default="data/mhqa/rejected.jsonl", help="Output rejected JSONL")
    parser.add_argument("--report", type=str, default="data/mhqa/qc_report.json", help="QC report JSON")
    parser.add_argument("--seeds", type=str, default="seeds.jsonl", help="Seed JSONL for dedup")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_sigint)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}. Run mhqa_generate.py first.")
        sys.exit(1)

    run_qc(
        input_path=input_path,
        verified_path=Path(args.verified),
        rejected_path=Path(args.rejected),
        report_path=Path(args.report),
        seeds_path=Path(args.seeds),
    )


if __name__ == "__main__":
    main()
