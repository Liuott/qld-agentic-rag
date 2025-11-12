# Agentic RAG for Queensland Road Rules

An **Agentic RAG** stack for Queensland road rules. It follows a rewrite → retrieve → rerank → verify → generate loop to keep answers grounded, auditable, and reproducible.

---

## Prerequisites
- Python 3.10+ (recommended) or GCP VM instances
- Optional: **Ollama** for local LLM judging in evaluation (e.g., `qwen2.5:3b` or larger)
- Optional: LangSmith (if you run `eval_langsmith`)
- Corpus/config files expected by the scripts under `src/`

> Example Ollama setup:
> ```bash
> ollama pull qwen2.5:3b
> export EVAL_OLLAMA_MODEL=qwen2.5:3b
> ```

---

Quick Start
# 1) Create & activate a virtual env
python -m venv .venv && source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Ingest corpus (chunking, indexing, etc.)
python -u -m src.ingest

# 4) Run the Agentic RAG app (interactive)
python -u -m src.app_graph
# e.g. ask:
# What should I do when crossing the road?

# 5) Generate a test set (randomly create N questions)
python -u -m src.eval.gen_testset --n 30

RAGAS evaluation (quantitative)

Environment variables (optional but recommended):

export EVAL_OLLAMA_MODEL=qwen2.5:3b
export RAGAS_CONCURRENCY=1
export RAGAS_TIMEOUT=600


Common runs:

A) Fast metrics only (no LLM judge)

python -u -m src.eval.eval_ragas \
  --limit 30 \
  --full false


B) Full metrics (LLM judge; needs EVAL_OLLAMA_MODEL)

python -u -m src.eval.eval_ragas \
  --limit 30 \
  --k 6 \
  --trim_ctx 1200 \
  --per_chunk 360 \
  --fetch_k 60 \
  --full true \
  --show_debug


C) Single row by index

python -u -m src.eval.eval_ragas --row_idx 2 --full false


Optional convenience alias

base() {
  python -u -m src.eval.eval_ragas \
    --limit 30 \
    --k 6 \
    --trim_ctx 1200 \
    --per_chunk 360 \
    --fetch_k 60 \
    --full true \
    --show_debug \
    --no_cache \
    "$@"
}

LangSmith evaluation (optional)

Most recent 30

python -u -m src.eval.eval_langsmith --limit 30 --tail


Single row by index

python -u -m src.eval.eval_langsmith --row_idx 2


Filter by keyword (substring match on question)

python -u -m src.eval.eval_langsmith --question_substr "site access only"

Batch loop run (agentic iterations)
python -u -m src.eval.loop_run --limit 30 --start 0 --max_steps 3

Expose as an API (for tools / OpenWebUI connector)
# 1) Free the port if occupied
sudo fuser -k 8008/tcp || true

# 2) Ensure package path
export PYTHONPATH="$(pwd):$PYTHONPATH"

# 3) Start the server
python -m uvicorn serve_api:app --host 0.0.0.0 --port 8008 --reload


Quick smoke test:

curl -X POST "http://localhost:8008/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"Who has priority at a give way sign in QLD?"}'


In OpenWebUI you can add this as a custom connector (OpenAI-compatible proxy or simple HTTP hook, depending on your setup), then route chats to your API.

Troubleshooting

RAGAS timeouts → Increase RAGAS_TIMEOUT (e.g., 1200) or start with --full false.

Faithfulness ≈ 0 with tiny local model → Try a larger model (e.g., qwen2.5:7b) and avoid over-truncating context (--trim_ctx).

Port conflicts → sudo fuser -k 8008/tcp then restart.

Low retrieval quality → Raise --fetch_k and --k. If your scripts support it, enable cross-encoder reranking.

GPU/CPU considerations for Ollama → Model size vs. latency trade-offs follow the LangChain/Ollama patterns used in teaching.
Project layout (snippet)
src/
  app_graph.py         # Agentic routing & chat entry
  ingest.py            # Chunking / indexing pipeline
  eval/
    gen_testset.py     # Generate eval set
    eval_ragas.py      # RAGAS metrics
    eval_langsmith.py  # LangSmith evaluation
    loop_run.py        # Batch/loop execution
serve_api.py           # FastAPI/Uvicorn entrypoint
