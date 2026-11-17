"""Core RAG building blocks shared by the CLI, the HTTP API and the eval suite.

Import from here rather than reaching into ``src.eval``::

    from src.rag import answer_question, get_llm, get_vectorstore, retrieve_context

The exports are resolved lazily (PEP 562) so that importing a light submodule
such as :mod:`src.rag.ranking` does not drag in LangChain, Chroma or torch --
which is what makes the ranking rules unit-testable in a bare environment.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY: dict[str, tuple[str, str]] = {
    # answering
    "answer_question": ("src.rag.answering", "answer_question"),
    "get_llm": ("src.rag.answering", "get_llm"),
    # embeddings
    "clear_embedding_cache": ("src.rag.embedding", "clear_embedding_cache"),
    "get_embeddings": ("src.rag.embedding", "get_embeddings"),
    # retrieval
    "get_cross_encoder": ("src.rag.retriever", "get_cross_encoder"),
    "retrieve_context": ("src.rag.retriever", "retrieve_context"),
    "retrieve_documents": ("src.rag.retriever", "retrieve_documents"),
    # vector store
    "clear_vectorstore_cache": ("src.rag.store", "clear_vectorstore_cache"),
    "get_vectorstore": ("src.rag.store", "get_vectorstore"),
    "open_vectorstore": ("src.rag.store", "open_vectorstore"),
    "vectorstore_is_empty": ("src.rag.store", "vectorstore_is_empty"),
}

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _LAZY[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value  # cache for subsequent lookups
    return value


def __dir__() -> list[str]:  # pragma: no cover - introspection helper
    return sorted(list(globals()) + __all__)
