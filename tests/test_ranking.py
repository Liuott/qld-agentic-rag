"""Tests for the pure retrieval-selection helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.rag.ranking import (
    ScoredDocument,
    assemble_context,
    dedupe,
    filter_by_threshold,
    head,
    rerank,
    sort_by_score,
)


@dataclass
class FakeDoc:
    page_content: str
    metadata: dict


def doc(text: str, source: str = "a.pdf") -> FakeDoc:
    return FakeDoc(text, {"source": source})


class TestFilterByThreshold:
    def test_keeps_scores_at_or_above_threshold(self):
        candidates = [
            ScoredDocument(doc("low"), 0.1),
            ScoredDocument(doc("edge"), 0.25),
            ScoredDocument(doc("high"), 0.9),
        ]
        kept = filter_by_threshold(candidates, 0.25)
        assert [c.document.page_content for c in kept] == ["edge", "high"]

    def test_none_threshold_disables_filtering(self):
        candidates = [ScoredDocument(doc("a"), 0.0), ScoredDocument(doc("b"), 0.5)]
        assert len(filter_by_threshold(candidates, None)) == 2

    def test_drops_missing_and_nan_scores(self):
        candidates = [
            ScoredDocument(doc("missing"), None),  # type: ignore[arg-type]
            ScoredDocument(doc("nan"), float("nan")),
            ScoredDocument(doc("ok"), 0.4),
        ]
        kept = filter_by_threshold(candidates, 0.2)
        assert [c.document.page_content for c in kept] == ["ok"]

    def test_drops_non_numeric_scores(self):
        candidates = [ScoredDocument(doc("weird"), "not-a-number")]  # type: ignore[arg-type]
        assert filter_by_threshold(candidates, 0.0) == []


class TestSortAndHead:
    def test_sorted_descending_by_score(self):
        candidates = [
            ScoredDocument(doc("mid"), 0.5),
            ScoredDocument(doc("high"), 0.9),
            ScoredDocument(doc("low"), 0.1),
        ]
        assert [d.page_content for d in sort_by_score(candidates)] == ["high", "mid", "low"]

    def test_head_respects_k(self):
        docs = [doc("1"), doc("2"), doc("3")]
        assert [d.page_content for d in head(docs, 2)] == ["1", "2"]

    @pytest.mark.parametrize("k", [0, -1])
    def test_head_with_non_positive_k_is_empty(self, k):
        assert head([doc("1")], k) == []


class TestDedupe:
    def test_collapses_identical_chunks(self):
        text = "x" * 200
        docs = [doc(text, "a.pdf"), doc(text, "a.pdf"), doc("y" * 200, "a.pdf")]
        assert len(dedupe(docs, k=10)) == 2

    def test_same_text_different_source_is_kept(self):
        text = "y" * 200
        docs = [doc(text, "a.pdf"), doc(text, "b.pdf")]
        assert len(dedupe(docs, k=10)) == 2

    def test_stops_at_k(self):
        docs = [doc(f"text-{i}" + "z" * 200) for i in range(5)]
        assert len(dedupe(docs, k=2)) == 2


class TestRerank:
    def test_reorders_by_external_score(self):
        docs = [doc("short"), doc("much longer text")]
        scores = {"short": 0.1, "much longer text": 0.9}

        def scorer(_query, texts):
            return [scores[text] for text in texts]

        assert [d.page_content for d in rerank(docs, scorer, k=1, query="q")] == ["much longer text"]

    def test_falls_back_when_scorer_raises(self):
        docs = [doc("first"), doc("second")]

        def broken(_query, _texts):
            raise RuntimeError("model exploded")

        assert [d.page_content for d in rerank(docs, broken, k=1)] == ["first"]

    def test_falls_back_when_score_count_mismatches(self):
        docs = [doc("first"), doc("second")]
        assert [d.page_content for d in rerank(docs, lambda _q, _t: [0.5], k=2)] == ["first", "second"]

    def test_empty_input(self):
        assert rerank([], lambda _q, _t: [], k=3) == []


class TestAssembleContext:
    def test_truncates_each_chunk_then_joins_then_trims(self):
        docs = [doc("A" * 10), doc("B" * 10)]
        chunks, joined = assemble_context(docs, per_chunk=5, trim=100)
        assert chunks == ["AAAAA", "BBBBB"]
        assert joined == "AAAAA\n\nBBBBB"

    def test_global_trim_applies_after_join(self):
        docs = [doc("A" * 10), doc("B" * 10)]
        chunks, joined = assemble_context(docs, per_chunk=10, trim=8)
        assert len(chunks) == 2
        assert joined == "AAAAAAAA"

    def test_empty_documents(self):
        assert assemble_context([], per_chunk=10, trim=10) == ([], "")
