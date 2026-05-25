"""Unit tests for WRRF (Weighted Reciprocal Rank Fusion) in src/retrieve.py."""

from __future__ import annotations

from src.retrieve import _wrrf_fusion


def _make_row(chunk_id: str, page_title: str = "P", score: float = 1.0) -> dict:
    return {
        "page_title": page_title,
        "page_url": f"https://vi.wikipedia.org/wiki/{page_title}",
        "page_id": f"pid_{chunk_id}",
        "chunk_id": chunk_id,
        "chunk_text": f"Text for {chunk_id}",
        "score": score,
    }


class TestWRRFFusion:
    """Tests for _wrrf_fusion function."""

    def test_basic_fusion_with_known_inputs(self) -> None:
        """Verify WRRF formula produces expected scores."""
        bm25 = [_make_row("c1"), _make_row("c2")]
        vector = [_make_row("c2"), _make_row("c3")]
        graph = [_make_row("c1"), _make_row("c3")]

        k = 60
        weights = [0.4, 0.4, 0.2]

        results = _wrrf_fusion([bm25, vector, graph], weights, k=k)

        # c1: bm25 rank=1, graph rank=1 -> 0.4/(60+1) + 0.2/(60+1)
        expected_c1 = 0.4 / (60 + 1) + 0.2 / (60 + 1)
        # c2: bm25 rank=2, vector rank=1 -> 0.4/(60+2) + 0.4/(60+1)
        expected_c2 = 0.4 / (60 + 2) + 0.4 / (60 + 1)
        # c3: vector rank=2, graph rank=2 -> 0.4/(60+2) + 0.2/(60+2)
        expected_c3 = 0.4 / (60 + 2) + 0.2 / (60 + 2)

        scores = {r["chunk_id"]: r["score"] for r in results}

        assert abs(scores["c1"] - expected_c1) < 1e-9
        assert abs(scores["c2"] - expected_c2) < 1e-9
        assert abs(scores["c3"] - expected_c3) < 1e-9

    def test_deduplication_by_chunk_id(self) -> None:
        """Same chunk_id appearing in multiple signals produces one output row."""
        bm25 = [_make_row("c1", "PageA"), _make_row("c2", "PageB")]
        vector = [_make_row("c1", "PageA"), _make_row("c2", "PageB")]
        graph = [_make_row("c1", "PageA")]

        results = _wrrf_fusion([bm25, vector, graph], [0.4, 0.4, 0.2], k=60)

        chunk_ids = [r["chunk_id"] for r in results]
        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk_ids in output"
        assert set(chunk_ids) == {"c1", "c2"}

    def test_weights_affect_ranking(self) -> None:
        """Higher weight on a signal boosts items ranked highly in that signal."""
        # c1 is rank 1 in bm25 only, c2 is rank 1 in vector only
        bm25 = [_make_row("c1")]
        vector = [_make_row("c2")]
        graph: list[dict] = []

        # Heavy BM25 weight -> c1 should rank higher
        results_bm25_heavy = _wrrf_fusion(
            [bm25, vector, graph], [0.9, 0.1, 0.0], k=60
        )
        assert results_bm25_heavy[0]["chunk_id"] == "c1"

        # Heavy vector weight -> c2 should rank higher
        results_vec_heavy = _wrrf_fusion(
            [bm25, vector, graph], [0.1, 0.9, 0.0], k=60
        )
        assert results_vec_heavy[0]["chunk_id"] == "c2"

    def test_empty_inputs_returns_empty(self) -> None:
        """All empty signal lists produce empty output."""
        results = _wrrf_fusion([[], [], []], [0.4, 0.4, 0.2], k=60)
        assert results == []

    def test_single_signal_nonempty(self) -> None:
        """Fusion works when only one signal has results."""
        bm25 = [_make_row("c1"), _make_row("c2"), _make_row("c3")]
        vector: list[dict] = []
        graph: list[dict] = []

        results = _wrrf_fusion([bm25, vector, graph], [0.4, 0.4, 0.2], k=60)

        assert len(results) == 3
        # Order should match bm25 order since it's the only signal
        assert results[0]["chunk_id"] == "c1"
        assert results[1]["chunk_id"] == "c2"
        assert results[2]["chunk_id"] == "c3"

    def test_metadata_preserved_from_first_occurrence(self) -> None:
        """Metadata comes from the first signal list that contains the chunk."""
        bm25 = [
            {
                "page_title": "BM25 Title",
                "page_url": "https://bm25.example.com",
                "page_id": "p1",
                "chunk_id": "c1",
                "chunk_text": "BM25 text",
                "score": 5.0,
            }
        ]
        vector = [
            {
                "page_title": "Vector Title",
                "page_url": "https://vector.example.com",
                "page_id": "p1_vec",
                "chunk_id": "c1",
                "chunk_text": "Vector text",
                "score": 0.95,
            }
        ]

        results = _wrrf_fusion([bm25, vector, []], [0.5, 0.5, 0.0], k=60)

        assert len(results) == 1
        # Metadata should come from bm25 (first signal list)
        assert results[0]["page_title"] == "BM25 Title"
        assert results[0]["page_url"] == "https://bm25.example.com"
        assert results[0]["chunk_text"] == "BM25 text"

    def test_k_parameter_affects_score_magnitude(self) -> None:
        """Smaller k gives higher scores (more weight to top ranks)."""
        bm25 = [_make_row("c1")]

        results_small_k = _wrrf_fusion([bm25, [], []], [1.0, 0.0, 0.0], k=10)
        results_large_k = _wrrf_fusion([bm25, [], []], [1.0, 0.0, 0.0], k=100)

        # score = 1.0 / (k + 1), so smaller k -> higher score
        assert results_small_k[0]["score"] > results_large_k[0]["score"]

    def test_scores_are_descending(self) -> None:
        """Output is sorted by score in descending order."""
        bm25 = [_make_row("c1"), _make_row("c2"), _make_row("c3")]
        vector = [_make_row("c3"), _make_row("c1"), _make_row("c2")]
        graph = [_make_row("c2"), _make_row("c3"), _make_row("c1")]

        results = _wrrf_fusion([bm25, vector, graph], [0.4, 0.4, 0.2], k=60)

        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)
