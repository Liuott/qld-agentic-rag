"""FastAPI entrypoint -- OpenWebUI / curl compatible.

Fixes vs. the original
----------------------
* no longer imports ``src.eval.eval_ragas`` (a 600-line evaluation script that
  dragged in ``ragas``, ``datasets`` and ``pandas`` just to expose two helpers);
  it now talks to :mod:`src.rag` directly;
* ``on_event("startup")`` (removed in modern FastAPI) replaced by a ``lifespan``
  handler;
* CORS is configured from ``CORS_ORIGINS`` instead of hard-coding a personal GCP
  IP and ``"*"`` (``allow_origins=["*"]`` + ``allow_credentials=True`` is
  rejected by browsers anyway);
* optional ``SERVE_API_KEY`` bearer protection for public deployments;
* responses report the retrieved sources so answers are auditable.

Run it::

    uvicorn serve_api:app --host 127.0.0.1 --port 8008
    # or
    python -m serve_api
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import get_settings
from src.rag import (
    answer_question,
    get_llm,
    get_vectorstore,
    retrieve_documents,
    vectorstore_is_empty,
)
from src.rag.ranking import assemble_context

logger = logging.getLogger("qld_agentic_rag.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class AskIn(BaseModel):
    question: str = Field(..., min_length=2, description="User question")
    k: int | None = Field(None, ge=1, le=50, description="Documents to keep")
    trim_ctx: int | None = Field(None, ge=200, le=8000, description="Max characters of context")
    per_chunk: int | None = Field(None, ge=100, le=4000, description="Max characters per chunk")
    fetch_k: int | None = Field(None, ge=1, le=500, description="Candidates pulled from the index")
    thr: float | None = Field(None, ge=0.0, le=1.0, description="Relevance threshold (0 disables)")
    rerank: bool | None = Field(None, description="Enable cross-encoder re-ranking")
    mmr: bool | None = Field(None, description="De-duplicate near-identical chunks")


class Source(BaseModel):
    source: str
    page: int | None = None
    preview: str


class AskOut(BaseModel):
    answer: str
    context: str
    chunks: list[str]
    sources: list[Source]


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the heavy objects once, at startup (not at import).

    Startup must not take the process down. Loading the embedding model needs
    the HuggingFace weights on disk, and that download is the single most common
    failure on a fresh machine (see ``HF_ENDPOINT`` in ``.env.example``). When it
    fails we keep the process up, record why, and let ``/readyz`` and ``/ask``
    report 503 with an actionable message instead of a bare connection error.
    """
    settings = get_settings()
    if settings.langsmith_tracing and settings.langsmith_api_key:
        import os

        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
        os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)

    app.state.error: str | None = None
    app.state.vectorstore = None
    app.state.llm = None
    try:
        logger.info("loading vector store from %s", settings.chroma_dir)
        app.state.vectorstore = get_vectorstore()
        app.state.llm = get_llm(temperature=0.0)
        if vectorstore_is_empty(app.state.vectorstore):
            logger.warning("vector store is empty -- run `python -m src.ingest` before serving")
    except Exception as exc:  # noqa: BLE001 - startup must survive a broken environment
        app.state.error = f"{type(exc).__name__}: {exc}"
        logger.error(
            "startup degraded, /ask will answer 503: %s\n"
            "hint: the embedding model (%s) is downloaded from HuggingFace on first use; "
            "set HF_ENDPOINT=https://hf-mirror.com if huggingface.co is unreachable, "
            "and run `python -m src.ingest` to build the index.",
            app.state.error,
            settings.embedding_model,
        )
    yield
    logger.info("shutting down")


app = FastAPI(title="QLD Agentic RAG", version="2.0.0", lifespan=lifespan)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(_settings.cors_origins) or ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Reject unauthenticated calls when ``SERVE_API_KEY`` is configured."""
    expected = get_settings().serve_api_key
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


@app.get("/healthz")
def healthz() -> dict[str, bool]:
    return {"ok": True}


@app.get("/readyz")
def readyz(request: Request) -> dict[str, Any]:
    store = getattr(request.app.state, "vectorstore", None)
    return {
        "ready": store is not None and not vectorstore_is_empty(store),
        "chroma_dir": str(get_settings().chroma_dir),
        "error": getattr(request.app.state, "error", None),
    }


@app.post("/ask", response_model=AskOut, dependencies=[Depends(require_api_key)])
def ask(payload: AskIn, request: Request) -> AskOut:
    settings = get_settings()
    store = getattr(request.app.state, "vectorstore", None)
    llm = getattr(request.app.state, "llm", None)
    if store is None or llm is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "service not ready: "
                f"{getattr(request.app.state, 'error', None) or 'startup did not complete'}. "
                "Run `python -m src.ingest` and make sure the embedding model is available "
                "(set HF_ENDPOINT=https://hf-mirror.com if huggingface.co is unreachable)."
            ),
        )

    llm = llm or get_llm(temperature=0.0)

    documents = retrieve_documents(
        store,
        payload.question,
        k=payload.k if payload.k is not None else settings.retriever_k,
        fetch_k=payload.fetch_k if payload.fetch_k is not None else settings.fetch_k,
        score_threshold=payload.thr if payload.thr is not None else settings.score_threshold,
        rerank_enabled=payload.rerank if payload.rerank is not None else settings.rerank,
        use_mmr=bool(payload.mmr),
    )
    chunks, joined = assemble_context(
        documents,
        per_chunk=payload.per_chunk if payload.per_chunk is not None else settings.per_chunk,
        trim=payload.trim_ctx if payload.trim_ctx is not None else settings.trim_ctx,
    )

    if not joined:
        logger.info("no context above threshold for %r", payload.question[:80])
        return AskOut(
            answer="I don't know. No clause in the indexed corpus matched this question.",
            context="",
            chunks=[],
            sources=[],
        )

    answer = answer_question(llm, joined, payload.question)
    return AskOut(
        answer=answer,
        context=joined,
        chunks=chunks,
        sources=[
            Source(
                source=str(doc.metadata.get("source", "unknown")),
                page=doc.metadata.get("page"),
                preview=(doc.page_content or "")[:160],
            )
            for doc in documents
        ],
    )


def main() -> None:  # pragma: no cover - convenience entrypoint
    import uvicorn

    settings = get_settings()
    uvicorn.run("serve_api:app", host=settings.api_host, port=settings.api_port, reload=False)


if __name__ == "__main__":  # pragma: no cover
    main()
