"""Retrieval: similarity search -> filtering -> re-ranking -> context assembly.

This is the single implementation of the retrieval pipeline.  Before the
refactor the exact same 35 lines existed in ``src/eval/eval_ragas.py``,
``src/eval/eval_newv.py``, ``src/eval/ask_one.py`` and (indirectly, through an
import of the eval module) in ``serve_api.py``.

Behavioural contract kept from the original code:

* candidates come from ``similarity_search_with_relevance_scores`` with
  ``fetch_k`` (never fewer than ``k``);
* ``score_threshold=None`` disables thresholding;
* ``rerank=True`` uses a sentence-transformers cross encoder;
* ``use_mmr=True`` de-duplicates near-identical chunks (content prefix +
  source).  True MMR was never implemented upstream; ``mmr_lambda`` stays in the
  signature for CLI compatibility.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any

from src.config import get_settings
from src.rag.ranking import (
    ScoredDocument,
    assemble_context,
    dedupe,
    filter_by_threshold,
    head,
    rerank,
    sort_by_score,
)


class _UseDefault:
    """Sentinel: "read the threshold from the configuration"."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "USE_DEFAULT"


USE_DEFAULT = _UseDefault()


def _cross_encoder_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
    except Exception:  # noqa: BLE001 - optional dependency
        return False
    return True


@lru_cache(maxsize=2)
def get_cross_encoder(model_name: str) -> Any:
    """Load (once per process) and cache a cross-encoder re-ranker.

    The previous code instantiated ``CrossEncoder(...)`` inside
    ``retrieve_context``, i.e. on *every* question: the weights were re-read
    from disk for each evaluated row.
    """
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def _make_scorer(model_name: str) -> Callable[[str, Sequence[str]], Sequence[float]]:
    encoder = get_cross_encoder(model_name)

    def score(query: str, texts: Sequence[str]) -> Sequence[float]:
        pairs = [(query, text) for text in texts]
        return [float(value) for value in encoder.predict(pairs)]

    return score


def retrieve_documents(
    vectorstore: Any,
    query: str,
    k: int | None = None,
    *,
    fetch_k: int | None = None,
    score_threshold: float | _UseDefault | None = USE_DEFAULT,
    rerank_enabled: bool | None = None,
    use_mmr: bool = False,
    cross_encoder_name: str | None = None,
) -> list[Any]:
    """Return the ``k`` most relevant documents for ``query``."""
    settings = get_settings()
    top_k = settings.retriever_k if k is None else k
    if top_k <= 0:
        return []

    pool = settings.fetch_k if fetch_k is None else fetch_k
    threshold = settings.score_threshold if isinstance(score_threshold, _UseDefault) else score_threshold
    do_rerank = settings.rerank if rerank_enabled is None else rerank_enabled
    ce_model = cross_encoder_name or settings.cross_encoder_model

    raw = vectorstore.similarity_search_with_relevance_scores(query, k=max(pool, top_k))
    candidates = [
        ScoredDocument(document=doc, score=score)  # type: ignore[arg-type]
        for doc, score in raw
    ]
    candidates = filter_by_threshold(candidates, threshold)
    if not candidates:
        return []

    ordered = sort_by_score(candidates)

    if do_rerank and len(ordered) > top_k and _cross_encoder_available():
        return rerank(ordered, _make_scorer(ce_model), top_k, query=query)
    if use_mmr and len(ordered) > top_k:
        return dedupe(ordered, top_k)
    return head(ordered, top_k)


def retrieve_context(
    vectorstore: Any,
    query: str,
    k: int = 6,
    trim: int = 1200,
    per_chunk: int = 300,
    use_mmr: bool = False,
    mmr_lambda: float = 0.85,
    fetch_k: int = 50,
    score_threshold: float | _UseDefault | None = USE_DEFAULT,
    rerank: bool = False,
    cross_encoder_name: str | None = None,
) -> tuple[list[str], str]:
    """Retrieve and flatten context, returning ``(chunks, joined_context)``.

    Signature-compatible with the original ``eval_ragas.retrieve_context``.
    """
    del mmr_lambda  # accepted for backwards compatibility, unused downstream
    documents = retrieve_documents(
        vectorstore,
        query,
        k=k,
        fetch_k=fetch_k,
        score_threshold=score_threshold,
        rerank_enabled=rerank,
        use_mmr=use_mmr,
        cross_encoder_name=cross_encoder_name,
    )
    if not documents:
        return [], ""
    return assemble_context(documents, per_chunk=per_chunk, trim=trim)


__all__ = [
    "USE_DEFAULT",
    "get_cross_encoder",
    "retrieve_context",
    "retrieve_documents",
]
