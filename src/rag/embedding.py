"""Embedding model factory.

The old code built a fresh ``HuggingFaceEmbeddings`` in six different places
(``ingest``, ``app_graph``, ``serve_api``, ``ask_one``, ``gen_testset`` and twice
inside the two eval scripts).  Each construction re-loads the sentence
transformer from disk, so a single evaluation run loaded the same 90 MB model
several times.  Everything now goes through :func:`get_embeddings`, which keeps
one instance per model name.
"""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import get_settings


@lru_cache(maxsize=4)
def get_embeddings(model_name: str | None = None) -> HuggingFaceEmbeddings:
    """Return a process-wide embedding model for ``model_name``.

    The model is loaded lazily on first use; importing this module costs
    nothing. If ``HF_ENDPOINT`` is configured (see ``.env.example``) it is
    exported before the model is fetched, which is what makes the first run work
    behind networks where ``huggingface.co`` is unreachable.
    """
    settings = get_settings()
    if settings.hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)
    name = model_name or settings.embedding_model
    return HuggingFaceEmbeddings(model_name=name)


def clear_embedding_cache() -> None:
    """Release cached models (used by tests and by ``ingest --reset``)."""
    get_embeddings.cache_clear()


__all__ = ["clear_embedding_cache", "get_embeddings"]
