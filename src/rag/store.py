"""Chroma vector-store helpers.

Nothing is opened at import time any more: ``src.app_graph`` used to construct a
``Chroma`` client and a sentence-transformer model simply by being imported,
which made ``python -c "import src.app_graph"`` slow and crashed with a cryptic
error when the index had not been built yet.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma

from src.config import get_settings
from src.rag.embedding import get_embeddings


def open_vectorstore(
    persist_dir: str | Path | None = None,
    collection_name: str | None = None,
    embedding_model: str | None = None,
) -> Chroma:
    """Open (creating if needed) a Chroma collection. Un-cached."""
    settings = get_settings()
    directory = Path(persist_dir) if persist_dir is not None else settings.chroma_dir
    directory.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection_name or settings.collection_name,
        embedding_function=get_embeddings(embedding_model or settings.embedding_model),
        persist_directory=str(directory),
    )


@lru_cache(maxsize=2)
def get_vectorstore(
    persist_dir: str | None = None,
    collection_name: str | None = None,
    embedding_model: str | None = None,
) -> Chroma:
    """Process-wide vector store (lazily created, one instance per location)."""
    return open_vectorstore(persist_dir, collection_name, embedding_model)


def clear_vectorstore_cache() -> None:
    get_vectorstore.cache_clear()


def vectorstore_is_empty(store: Chroma) -> bool:
    """True when the collection holds no documents."""
    try:
        return store._collection.count() == 0  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - different chromadb versions expose different APIs
        try:
            return len(store.get(limit=1).get("ids", [])) == 0
        except Exception:  # noqa: BLE001
            return True


__all__ = [
    "clear_vectorstore_cache",
    "get_vectorstore",
    "open_vectorstore",
    "vectorstore_is_empty",
]
