"""Chunk the source documents and (re)build the Chroma index.

Improvements over the original
------------------------------
* ``--reset`` wipes the collection first, so re-running ingest no longer
  duplicates every chunk (the original ``add_documents`` appended blindly);
* chunk ids are deterministic (``source-page-start``), which makes re-ingest
  idempotent and lets Chroma skip unchanged chunks;
* empty pages (common in scanned PDFs) are dropped instead of being indexed as
  blank context;
* every chunk carries ``source`` / ``page`` / ``chunk_index`` metadata, which is
  what the API uses to show citations;
* the batch size is configurable and progress is reported per batch.

    python -m src.ingest               # incremental
    python -m src.ingest --reset       # rebuild from scratch
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_settings
from src.rag import open_vectorstore, vectorstore_is_empty

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DEFAULT_BATCH = 64

LOADERS = {
    ".pdf": PyPDFLoader,
    ".md": TextLoader,
    ".txt": TextLoader,
}


def load_sources(sources_dir: Path | None = None) -> list[Document]:
    """Load every supported file under ``data/sources``."""
    directory = sources_dir or get_settings().sources_dir
    if not directory.exists():
        raise SystemExit(f"Source directory not found: {directory}")

    documents: list[Document] = []
    for path in sorted(directory.iterdir()):
        loader_cls = LOADERS.get(path.suffix.lower())
        if loader_cls is None:
            continue
        try:
            if loader_cls is TextLoader:
                loaded = loader_cls(str(path), encoding="utf-8").load()
            else:
                loaded = loader_cls(str(path)).load()
        except Exception as exc:  # noqa: BLE001 - one broken PDF must not stop the run
            print(f"[Ingest] skip {path.name}: {exc}")
            continue
        kept = [doc for doc in loaded if (doc.page_content or "").strip()]
        print(f"[Ingest] {path.name}: {len(kept)}/{len(loaded)} non-empty pages")
        documents.extend(kept)

    if not documents:
        raise SystemExit(f"No loadable documents found in {directory}")
    return documents


def chunk_documents(documents: Sequence[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True)
    chunks = splitter.split_documents(list(documents))
    for index, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata or {})
        metadata["chunk_index"] = index
        chunk.metadata = metadata
    print(f"[Ingest] Split into {len(chunks)} chunks")
    return chunks


def chunk_id(chunk: Document) -> str:
    """Deterministic id: same input document -> same id."""
    metadata = chunk.metadata or {}
    source = Path(str(metadata.get("source", "unknown"))).name
    page = metadata.get("page", "-")
    start = metadata.get("start_index", metadata.get("chunk_index", "-"))
    digest = hashlib.sha1((chunk.page_content or "").encode("utf-8")).hexdigest()[:8]
    return f"{source}:{page}:{start}:{digest}"


def ingest(
    *,
    reset: bool = False,
    batch_size: int = DEFAULT_BATCH,
    sources_dir: Path | None = None,
) -> int:
    """Rebuild/refresh the index; returns the number of indexed chunks."""
    vectorstore = open_vectorstore()
    if reset:
        existing = vectorstore.get()
        ids: Iterable[str] = existing.get("ids", []) or []
        ids = list(ids)
        if ids:
            vectorstore.delete(ids=ids)
            print(f"[Ingest] --reset removed {len(ids)} chunks")
        else:
            print("[Ingest] --reset: collection already empty")
    elif not vectorstore_is_empty(vectorstore):
        print(
            "[Ingest] WARNING: the collection is not empty. Re-running without --reset "
            "adds chunks with deterministic ids (unchanged chunks are skipped)."
        )

    documents = load_sources(sources_dir)
    chunks = chunk_documents(documents)
    ids = [chunk_id(chunk) for chunk in chunks]

    total = len(chunks)
    for start in range(0, total, batch_size):
        batch = chunks[start : start + batch_size]
        batch_ids = ids[start : start + batch_size]
        vectorstore.add_documents(batch, ids=batch_ids)
        print(f"[Ingest] Embedded {min(start + batch_size, total)}/{total}", flush=True)

    print(f"[Ingest] Done. {total} chunks persisted at {get_settings().chroma_dir}")
    return total


def main(argv: Sequence[str] | None = None) -> int:
    settings = get_settings()
    cli = argparse.ArgumentParser(description="Build the Chroma index from data/sources.")
    cli.add_argument("--reset", action="store_true", help="delete existing chunks first")
    cli.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    cli.add_argument("--sources", type=Path, default=settings.sources_dir)
    args = cli.parse_args(argv)

    ingest(reset=args.reset, batch_size=args.batch_size, sources_dir=args.sources)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
