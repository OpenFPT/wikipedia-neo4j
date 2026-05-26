"""Evaluate NER quality in the knowledge graph.

Performs three types of evaluation:
1. Noise detection — heuristic filters for non-entity strings
2. Type accuracy — cross-reference entity types against Wikidata
3. Deduplication — detect fragmented/duplicate entities

Usage:
    uv run python scripts/evaluate_ner.py
    uv run python scripts/evaluate_ner.py --wikidata        # include Wikidata lookup
    uv run python scripts/evaluate_ner.py --sample 200      # limit Wikidata queries
    uv run python scripts/evaluate_ner.py --output reports/ner-eval.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import requests

from src.logging_utils import get_logger

logger = get_logger(__name__)

ENTITIES_PATH = Path("data/export/entities.jsonl")
CHUNKS_PATH = Path("data/export/chunks.jsonl")
MENTIONS_PATH = Path("data/export/mentions.csv")
GOLD_PATH = Path("data/gold/ner_gold_sample.json")

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
USER_AGENT = "ViWikiNER-Eval/1.0 (academic research)"

WIKIDATA_TYPE_MAP = {
    "Q5": "Person",
    "Q515": "Location",
    "Q6256": "Location",
    "Q35657": "Location",
    "Q3624078": "Location",
    "Q486972": "Location",
    "Q532": "Location",
    "Q5119": "Location",
    "Q15284": "Location",
    "Q43229": "Organization",
    "Q4830453": "Organization",
    "Q476028": "Organization",
    "Q847017": "Organization",
    "Q6881511": "Organization",
    "Q7278": "Organization",
    "Q3918": "Organization",
    "Q11424": "Work",
    "Q7725634": "Work",
    "Q482994": "Work",
    "Q134556": "Work",
    "Q5398426": "Work",
}

NOISE_PATTERNS = [
    re.compile(r"^[,.\-;:\s]"),
    re.compile(r"^(năm|mùa giải|vòng|thế kỷ|tháng)\b", re.IGNORECASE),
    re.compile(r"^\d{4}$"),
    re.compile(r"^(áo số|trận đấu|bàn thắng)\b"),
    re.compile(r"^(sự kiện|khoảnh khắc|việc|bối cảnh|bên cạnh|ngày)\b"),
    re.compile(r"^(trước|sau|trong|ngoài|khoảng)\b"),
    re.compile(r"^(chứng|trực chứng|khai mở|kinh điển)\b"),
    re.compile(r"\d+\s*(tháng|năm|tuổi|km|m²)"),
]

ORG_SURFACE_PATTERNS = [
    re.compile(r"\b(F\.?C\.?|United|City|FC|SC|CF|AFC|SFC)\b", re.IGNORECASE),
    re.compile(r"\b(Club|Team|Academy|Athletic)\b", re.IGNORECASE),
    re.compile(r"\b(Inc|Ltd|Corp|Co\.|LLC|GmbH|S\.A\.)\b", re.IGNORECASE),
    re.compile(r"\b(University|Institute|Foundation|Association)\b", re.IGNORECASE),
    re.compile(r"\b(Đại học|Công ty|Tập đoàn|Câu lạc bộ|Hiệp hội)\b"),
]


@dataclass
class EvalResult:
    total_entities: int = 0
    type_distribution: dict[str, int] = field(default_factory=dict)
    noise_count: int = 0
    noise_rate: float = 0.0
    noise_examples: list[dict[str, str]] = field(default_factory=list)
    type_errors: list[dict[str, str]] = field(default_factory=list)
    type_error_count: int = 0
    type_accuracy: float = 0.0
    wikidata_checked: int = 0
    wikidata_matched: int = 0
    duplicates: list[dict[str, Any]] = field(default_factory=list)
    duplicate_count: int = 0
    fragmentation_rate: float = 0.0
    lowercase_start_count: int = 0
    short_name_count: int = 0
    summary: dict[str, Any] = field(default_factory=dict)


def load_entities() -> list[dict[str, str]]:
    entities = []
    with open(ENTITIES_PATH) as f:
        for line in f:
            entities.append(json.loads(line))
    return entities


def detect_noise(entities: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split entities into clean and noise based on heuristic patterns."""
    clean, noise = [], []
    for e in entities:
        name = e["name"]
        is_noise = False

        if len(name) <= 1:
            is_noise = True
        elif name[0].islower() and not any(c.isupper() for c in name[1:]):
            is_noise = True
        else:
            for pat in NOISE_PATTERNS:
                if pat.search(name):
                    is_noise = True
                    break

        if is_noise:
            noise.append(e)
        else:
            clean.append(e)

    return clean, noise


def detect_type_errors_heuristic(entities: list[dict[str, str]]) -> list[dict[str, str]]:
    """Detect obvious type misclassifications using surface patterns."""
    errors = []
    for e in entities:
        name = e["name"]
        label = e["label"]

        for pat in ORG_SURFACE_PATTERNS:
            if pat.search(name) and label != "Organization":
                errors.append({
                    "name": name,
                    "current_type": label,
                    "suggested_type": "Organization",
                    "reason": f"surface pattern: {pat.pattern}",
                })
                break

    return errors


def detect_duplicates(entities: list[dict[str, str]], threshold: float = 0.85) -> list[dict[str, Any]]:
    """Find likely duplicate entities via normalized name similarity."""
    def normalize(name: str) -> str:
        name = unicodedata.normalize("NFKC", name).lower().strip()
        name = re.sub(r"[.\-,;:()'\"]", "", name)
        name = re.sub(r"\s+", " ", name)
        return name

    normalized = [(normalize(e["name"]), e) for e in entities]
    normalized.sort(key=lambda x: x[0])

    duplicates = []
    seen: set[int] = set()

    for i in range(len(normalized)):
        if i in seen:
            continue
        group = [normalized[i][1]]
        for j in range(i + 1, min(i + 50, len(normalized))):
            if j in seen:
                continue
            ratio = SequenceMatcher(None, normalized[i][0], normalized[j][0]).ratio()
            if ratio >= threshold:
                group.append(normalized[j][1])
                seen.add(j)

        if len(group) > 1:
            seen.add(i)
            types_in_group = set(e["label"] for e in group)
            duplicates.append({
                "canonical": group[0]["name"],
                "variants": [e["name"] for e in group],
                "types": list(types_in_group),
                "type_conflict": len(types_in_group) > 1,
            })

    return duplicates


def query_wikidata_types(names: list[str], batch_size: int = 20) -> dict[str, str | None]:
    """Query Wikidata for entity types by Vietnamese label."""
    results: dict[str, str | None] = {}

    for i in range(0, len(names), batch_size):
        batch = names[i:i + batch_size]
        values = " ".join(f'"{n}"@vi' for n in batch)

        query = f"""
        SELECT ?item ?itemLabel ?type WHERE {{
          VALUES ?label {{ {values} }}
          ?item rdfs:label ?label .
          ?item wdt:P31 ?type .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "vi,en". }}
        }}
        LIMIT 200
        """

        try:
            headers = {
                "Accept": "application/sparql-results+json",
                "User-Agent": USER_AGENT,
            }
            resp = requests.get(
                WIKIDATA_SPARQL_URL,
                params={"query": query},
                headers=headers,
                timeout=30,
            )

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", "60"))
                logger.warning(f"Rate limited, sleeping {retry_after}s")
                time.sleep(retry_after)
                continue

            if resp.status_code != 200:
                logger.warning(f"Wikidata returned {resp.status_code}")
                continue

            bindings = resp.json().get("results", {}).get("bindings", [])

            for b in bindings:
                label = b.get("itemLabel", {}).get("value", "")
                type_uri = b.get("type", {}).get("value", "")
                qid = type_uri.split("/")[-1] if type_uri else ""

                if qid in WIKIDATA_TYPE_MAP and label:
                    results[label] = WIKIDATA_TYPE_MAP[qid]

        except Exception as exc:
            logger.warning(f"Wikidata query failed: {exc}")

        time.sleep(1.5)

    return results


def evaluate(
    use_wikidata: bool = False,
    sample_size: int = 100,
    output_path: Path | None = None,
) -> EvalResult:
    """Run full NER evaluation pipeline."""
    entities = load_entities()
    result = EvalResult(total_entities=len(entities))

    type_counts = Counter(e["label"] for e in entities)
    result.type_distribution = dict(type_counts.most_common())

    clean, noise = detect_noise(entities)
    result.noise_count = len(noise)
    result.noise_rate = len(noise) / len(entities) if entities else 0
    result.noise_examples = [{"name": e["name"], "type": e["label"]} for e in noise[:50]]

    result.lowercase_start_count = sum(1 for e in entities if e["name"][0:1].islower())
    result.short_name_count = sum(1 for e in entities if len(e["name"]) <= 2)

    type_errors = detect_type_errors_heuristic(clean)
    result.type_errors = type_errors[:50]
    result.type_error_count = len(type_errors)

    duplicates = detect_duplicates(clean)
    result.duplicates = duplicates[:30]
    result.duplicate_count = len(duplicates)
    result.fragmentation_rate = len(duplicates) / len(clean) if clean else 0

    if use_wikidata:
        names_to_check = [e["name"] for e in clean[:sample_size]]
        wikidata_types = query_wikidata_types(names_to_check)
        result.wikidata_checked = len(names_to_check)
        result.wikidata_matched = len(wikidata_types)

        name_to_label = {e["name"]: e["label"] for e in clean}
        wikidata_errors = []
        for name, wd_type in wikidata_types.items():
            our_type = name_to_label.get(name)
            if our_type and our_type != wd_type:
                wikidata_errors.append({
                    "name": name,
                    "our_type": our_type,
                    "wikidata_type": wd_type,
                })

        result.type_errors.extend(wikidata_errors)
        result.type_error_count += len(wikidata_errors)

        matched_correct = result.wikidata_matched - len(wikidata_errors)
        result.type_accuracy = matched_correct / result.wikidata_matched if result.wikidata_matched else 0

    clean_count = len(clean)
    heuristic_type_error_rate = len(type_errors) / clean_count if clean_count else 0

    result.summary = {
        "total_entities": len(entities),
        "noise_entities": len(noise),
        "noise_rate_pct": round(result.noise_rate * 100, 1),
        "clean_entities": clean_count,
        "heuristic_type_errors": len(type_errors),
        "heuristic_type_error_rate_pct": round(heuristic_type_error_rate * 100, 1),
        "duplicate_groups": len(duplicates),
        "fragmentation_rate_pct": round(result.fragmentation_rate * 100, 1),
        "estimated_precision_pct": round(
            (1 - result.noise_rate) * (1 - heuristic_type_error_rate) * 100, 1
        ),
    }

    if use_wikidata:
        result.summary["wikidata_checked"] = result.wikidata_checked
        result.summary["wikidata_matched"] = result.wikidata_matched
        result.summary["wikidata_type_accuracy_pct"] = round(result.type_accuracy * 100, 1)

    report = asdict(result)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"Report written to {output_path}")

    return result


def print_report(result: EvalResult) -> None:
    """Print human-readable evaluation report."""
    s = result.summary
    print("\n" + "=" * 60)
    print("  NER EVALUATION REPORT")
    print("=" * 60)

    print(f"\n{'Total entities:':<30} {s['total_entities']}")
    print(f"{'Type distribution:':<30}")
    for t, c in result.type_distribution.items():
        pct = c / s["total_entities"] * 100
        print(f"  {t:<20} {c:>6} ({pct:.1f}%)")

    print(f"\n--- NOISE DETECTION ---")
    print(f"{'Noise entities:':<30} {s['noise_entities']} ({s['noise_rate_pct']}%)")
    print(f"{'Clean entities:':<30} {s['clean_entities']}")
    if result.noise_examples:
        print(f"\n  Top noise examples:")
        for ex in result.noise_examples[:15]:
            print(f"    [{ex['type']:<12}] {ex['name']}")

    print(f"\n--- TYPE ACCURACY ---")
    print(f"{'Heuristic type errors:':<30} {s['heuristic_type_errors']} ({s['heuristic_type_error_rate_pct']}%)")
    if result.type_errors:
        print(f"\n  Type error examples:")
        for err in result.type_errors[:15]:
            current = err.get("current_type", err.get("our_type", "?"))
            suggested = err.get("suggested_type", err.get("wikidata_type", "?"))
            print(f"    {err['name']:<30} {current} -> {suggested}")

    if "wikidata_type_accuracy_pct" in s:
        print(f"\n{'Wikidata checked:':<30} {s['wikidata_checked']}")
        print(f"{'Wikidata matched:':<30} {s['wikidata_matched']}")
        print(f"{'Wikidata type accuracy:':<30} {s['wikidata_type_accuracy_pct']}%")

    print(f"\n--- DEDUPLICATION ---")
    print(f"{'Duplicate groups:':<30} {s['duplicate_groups']} ({s['fragmentation_rate_pct']}%)")
    if result.duplicates:
        print(f"\n  Duplicate examples:")
        for dup in result.duplicates[:10]:
            conflict = " [TYPE CONFLICT]" if dup["type_conflict"] else ""
            print(f"    {dup['variants']}{conflict}")

    print(f"\n--- OVERALL ---")
    print(f"{'Estimated precision:':<30} {s['estimated_precision_pct']}%")
    print(f"  (clean rate x type correctness rate)")
    print("=" * 60 + "\n")


def evaluate_against_gold(backends: list[str] | None = None) -> dict[str, Any]:
    """Evaluate NER backends against gold-standard annotations.

    Returns per-backend precision, recall, F1 (entity-level and typed).
    """
    import importlib
    import os

    if not GOLD_PATH.exists():
        logger.warning(f"Gold file not found: {GOLD_PATH}")
        return {}

    with open(GOLD_PATH) as f:
        gold_data = json.load(f)

    chunk_texts = {}
    with open(CHUNKS_PATH) as f:
        for line in f:
            c = json.loads(line)
            chunk_texts[c["chunk_id"]] = c["text"]

    if backends is None:
        backends = ["simple", "underthesea"]

    results: dict[str, Any] = {}

    for backend in backends:
        os.environ["NER_BACKEND"] = backend

        import src.config as cfg_mod
        import src.ner as ner_mod
        importlib.reload(cfg_mod)
        importlib.reload(ner_mod)
        from src.ner import extract_entities

        tp_entity = 0
        fp_entity = 0
        fn_entity = 0
        tp_typed = 0
        fp_typed = 0
        fn_typed = 0
        per_type_tp: Counter = Counter()
        per_type_fp: Counter = Counter()
        per_type_fn: Counter = Counter()

        for sample in gold_data:
            chunk_id = sample["chunk_id"]
            text = chunk_texts.get(chunk_id, "")
            if not text:
                continue

            gold_entities = {(e["name"], e["type"]) for e in sample["entities"]}
            gold_names = {e["name"] for e in sample["entities"]}

            predicted = extract_entities(text)
            pred_set = set(predicted)
            pred_names = {name for name, _ in predicted}

            # Entity-level (name match only, ignoring type)
            tp_entity += len(pred_names & gold_names)
            fp_entity += len(pred_names - gold_names)
            fn_entity += len(gold_names - pred_names)

            # Typed (name + type must match)
            tp_typed += len(pred_set & gold_entities)
            fp_typed += len(pred_set - gold_entities)
            fn_typed += len(gold_entities - pred_set)

            # Per-type metrics
            for name, etype in gold_entities:
                if (name, etype) in pred_set:
                    per_type_tp[etype] += 1
                else:
                    per_type_fn[etype] += 1
            for name, etype in pred_set:
                if (name, etype) not in gold_entities:
                    per_type_fp[etype] += 1

        def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}

        entity_metrics = _prf(tp_entity, fp_entity, fn_entity)
        typed_metrics = _prf(tp_typed, fp_typed, fn_typed)

        all_types = set(per_type_tp) | set(per_type_fp) | set(per_type_fn)
        per_type_metrics = {}
        for t in sorted(all_types):
            per_type_metrics[t] = _prf(per_type_tp[t], per_type_fp[t], per_type_fn[t])

        results[backend] = {
            "entity_level": entity_metrics,
            "typed": typed_metrics,
            "per_type": per_type_metrics,
            "counts": {
                "gold_total": tp_typed + fn_typed,
                "predicted_total": tp_typed + fp_typed,
            },
        }

    return results


def print_gold_report(gold_results: dict[str, Any]) -> None:
    """Print gold-standard evaluation results."""
    if not gold_results:
        print("\n  No gold-standard evaluation (missing gold file)")
        return

    print("\n" + "=" * 60)
    print("  GOLD-STANDARD EVALUATION (Precision / Recall / F1)")
    print("=" * 60)

    for backend, metrics in gold_results.items():
        print(f"\n  Backend: {backend}")
        print(f"  {'─' * 50}")

        em = metrics["entity_level"]
        tm = metrics["typed"]
        counts = metrics["counts"]

        print(f"  Gold entities: {counts['gold_total']}  |  Predicted: {counts['predicted_total']}")
        print(f"\n  {'Metric':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"  {'─' * 50}")
        print(f"  {'Entity (name only)':<20} {em['precision']:>10.1%} {em['recall']:>10.1%} {em['f1']:>10.1%}")
        print(f"  {'Typed (name+type)':<20} {tm['precision']:>10.1%} {tm['recall']:>10.1%} {tm['f1']:>10.1%}")

        print(f"\n  Per-type breakdown:")
        for t, m in metrics["per_type"].items():
            print(f"    {t:<15} P={m['precision']:.1%}  R={m['recall']:.1%}  F1={m['f1']:.1%}")

    print("\n" + "=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate NER quality in the knowledge graph")
    parser.add_argument("--wikidata", action="store_true", help="Enable Wikidata cross-reference")
    parser.add_argument("--sample", type=int, default=100, help="Sample size for Wikidata queries")
    parser.add_argument("--output", type=str, default="reports/ner-eval.json", help="Output JSON path")
    parser.add_argument("--gold", action="store_true", help="Run gold-standard P/R/F1 evaluation")
    parser.add_argument("--backends", nargs="+", default=None, help="Backends to evaluate (default: simple underthesea)")
    args = parser.parse_args()

    output_path = Path(args.output)
    result = evaluate(
        use_wikidata=args.wikidata,
        sample_size=args.sample,
        output_path=output_path,
    )
    print_report(result)

    if args.gold:
        gold_results = evaluate_against_gold(backends=args.backends)
        print_gold_report(gold_results)

        if output_path:
            report_data = json.loads(output_path.read_text())
            report_data["gold_evaluation"] = gold_results
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
