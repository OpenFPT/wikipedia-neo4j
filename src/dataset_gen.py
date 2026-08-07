"""Backward-compatibility shim: imports redirect to src.ingestion.dataset_gen."""

from src.ingestion.dataset_gen import (  # noqa: F401
    KGWalk,
    QAPair,
    extract_2hop_walks,
    extract_3hop_walks,
    generate_qa_from_walks,
    extract_broken_walks,
    generate_unanswerable_qa,
    rewrite_questions_with_llm,
    run_qc_pipeline,
    save_dataset,
)
