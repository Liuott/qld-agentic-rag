"""Pure retrieval-selection helpers.

This module deliberately has **no third-party imports** so the ranking rules can
be unit tested without LangChain, Chroma or a model download.  The behaviour is
identical to the original inline implementation in ``src/eval/eval_ragas.py``:

1. drop candidates below the relevance threshold;
2. sort by score, descending;
3. take ``k`` documents -- optionally de-duplicated, optionally re-ordered by an
   external re-ranker (cross encoder).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

DEDUPE_PREFIX = 120


@dataclass(frozen=True)
class ScoredDocument:
    """A document plus its relevance score."""

    document: Any
    score: float


def filter_by_threshold(candidates: Iterable[ScoredDocument], threshold: float | None) -> list[ScoredDocument]:
    """Keep candidates whose score is not below ``threshold``.

    ``threshold is None`` disables filtering.  Candidates whose score is not a
    finite number are dropped, because ``similarity_search_with_relevance_scores``
    can return ``None``/``nan`` when the store is misconfigured.
    """
    kept: list[ScoredDocument] = []
    for candidate in candidates:
        score = candidate.score
        if score is None:
            continue
        try:
            value = float(score)
        except (TypeError, ValueError):
            continue
        if math.isnan(value) or math.isinf(value):
            continue
        if threshold is not None and value < threshold:
            continue
        kept.append(ScoredDocument(candidate.document, value))
    return kept


def sort_by_score(candidates: Iterable[ScoredDocument]) -> list[Any]:
    """Return documents ordered by descending relevance score."""
    return [c.document for c in sorted(candidates, key=lambda c: c.score, reverse=True)]


def head(documents: Sequence[Any], k: int) -> list[Any]:
    """First ``k`` documents (``k <= 0`` yields an empty list)."""
    if k <= 0:
        return []
    return list(documents)[:k]


def dedupe(documents: Sequence[Any], k: int, prefix: int = DEDUPE_PREFIX) -> list[Any]:
    """Drop near-duplicate chunks, keeping original order, capped at ``k``.

    Two chunks collide when their first ``prefix`` characters *and* their source
    metadata match -- the same rule the original "MMR" branch used.
    """
    unique: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for doc in documents:
        content = getattr(doc, "page_content", "") or ""
        metadata = getattr(doc, "metadata", None) or {}
        key = (content[:prefix].strip(), str(metadata.get("source", "")))
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
        if len(unique) >= k:
            break
    return unique


def rerank(
    documents: Sequence[Any],
    score_fn: Callable[[str, Sequence[str]], Sequence[float]],
    k: int,
    *,
    query: str = "",
) -> list[Any]:
    """Re-order ``documents`` with an external scorer and keep the best ``k``.

    ``score_fn`` receives ``(query, [document_text, ...])`` and returns one score
    per document.  Any failure falls back to plain truncation, matching the
    original behaviour.
    """
    if not documents:
        return []
    texts = [getattr(doc, "page_content", "") or "" for doc in documents]
    try:
        scores = list(score_fn(query, texts))
    except Exception:  # noqa: BLE001 - a broken re-ranker must not break retrieval
        return head(documents, k)
    if len(scores) != len(documents):
        return head(documents, k)
    ranked = sorted(zip(documents, (float(s) for s in scores), strict=False), key=lambda item: item[1], reverse=True)
    return [doc for doc, _ in ranked][:k]


def assemble_context(documents: Sequence[Any], per_chunk: int, trim: int) -> tuple[list[str], str]:
    """Turn documents into ``(chunks, joined_context)``.

    ``per_chunk`` truncates every chunk, ``trim`` truncates the joined string.
    Both are character budgets, exactly like the original implementation.
    """
    chunks = [(getattr(doc, "page_content", "") or "")[:per_chunk] for doc in documents]
    joined = "\n\n".join(chunks)[:trim]
    return chunks, joined


__all__ = [
    "DEDUPE_PREFIX",
    "ScoredDocument",
    "assemble_context",
    "dedupe",
    "filter_by_threshold",
    "head",
    "rerank",
    "sort_by_score",
]
