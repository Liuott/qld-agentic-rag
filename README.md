# QLD Agentic RAG

[![CI](https://github.com/Liuott/qld-agentic-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Liuott/qld-agentic-rag/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An **agentic RAG** system for Queensland road rules. It answers questions about
the *Queensland Road Rules 2009* and the *Your Keys to Driving in Queensland*
handbook with grounded, citation-bearing answers, and it ships with a
reproducible evaluation harness (RAGAS metrics + an LLM judge).

The pipeline is a **rewrite → retrieve → grade → generate** loop implemented as a
LangGraph state machine:

```
                 ┌──────────────────────────────────────────┐
                 │                                          │
  question ──► agent ──(tool call)──► retrieve ──► grade ────┤
                 │                      │           │         │
                 │ (no tool call)       │           ├─ generate ──► answer
                 ▼                      │           │
              generate                  └── documents└─ rewrite ──┘
                                                     (budget left)
```

Everything runs locally: **Ollama** for generation, **Chroma** for the vector
store, `sentence-transformers` for embeddings and optional cross-encoder
re-ranking. No API keys are required.

---

## Quick start

```bash
# 1. environment
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[serve,dev]"                          # exact pins: pip install -r requirements.txt

# 2. local model
ollama pull qwen2.5:3b

# 3. configuration
cp .env.example .env        # then edit if needed

# 4. build the index (PDFs live in data/sources/)
python -m src.ingest --reset

# 5. ask
python -m src.app_graph --question "Who has priority at a give way sign?" --show-sources

# 6. serve
uvicorn serve_api:app --host 127.0.0.1 --port 8008
curl -X POST http://127.0.0.1:8008/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "Who has priority at a give way sign in QLD?"}'
```

---

## Configuration

All knobs are read once by [`src/config.py`](src/config.py) and validated at
startup; a malformed value fails fast with the variable name.

| Variable | Default | Meaning |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | sentence-transformers model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `qwen2.5:3b` | model used for answering |
| `EVAL_OLLAMA_MODEL` | value of `OLLAMA_MODEL` | model used by the eval scripts |
| `CHROMA_DIR` | `storage/chroma_db` | index location (relative to the repo root) |
| `RETRIEVER_K` | `6` | documents kept per question |
| `FETCH_K` | `80` | candidates pulled from the index |
| `THR` | `0.20` | relevance threshold (`0` disables) |
| `RERANK` | `true` | cross-encoder re-ranking |
| `TRIM_CTX` / `PER_CHUNK` | `1200` / `360` | context character budgets |
| `MAX_AGENT_STEPS` | `4` | hard cap on rewrite/retrieve iterations |
| `CORS_ORIGINS` | localhost:3000,127.0.0.1:3000 | allowed browser origins |
| `SERVE_API_KEY` | *(empty)* | when set, `/ask` requires `X-API-Key` |
| `LANGSMITH_*` | off | optional tracing |

---

## Evaluation

The harness writes machine-readable artefacts so experiments stay comparable:

```bash
# fast: retrieval metrics only (no LLM calls while scoring)
python -m src.eval.eval_ragas --limit 30 --full false

# full: adds faithfulness + answer relevance via the local model
python -m src.eval.eval_ragas --limit 30 --full true

# one row, for debugging
python -m src.eval.eval_ragas --row_idx 2 --full false

# self-evaluation loop: critic -> wider retrieval -> regenerate
python -m src.eval.eval_ragas --limit 30 --loop --judge
```

| Artefact | Content |
|---|---|
| `data/eval/testset.csv` | question / ground_truth / optional ref_context |
| `data/eval/run_answers.csv` | answer cache keyed by question + retrieval settings |
| `experiments/ragas_row_scores.csv` | per-row context precision / recall |
| `experiments/ab_results.csv` | one aggregated row per run |
| `experiments/langsmith_scores.csv` | LLM-judge grades |

Generating a fresh test set:

```bash
python -m src.eval.gen_testset --n 30
```

---

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | liveness |
| `GET` | `/readyz` | is the index populated? |
| `POST` | `/ask` | `{question, k?, thr?, rerank?, mmr?, ...}` → `{answer, context, chunks, sources}` |

`/ask` returns the retrieved sources with page numbers, so every answer is
auditable. Set `SERVE_API_KEY` before exposing the service: unauthenticated
`/ask` calls are then rejected with 401.

---

## Project layout

```
.
├── src/
│   ├── config.py            # single source of truth for paths + settings
│   ├── prompt.py            # prompt templates
│   ├── ingest.py            # chunking + idempotent indexing
│   ├── app_graph.py         # LangGraph agent (CLI entrypoint)
│   ├── rag/                 # reusable core: embeddings, store, retriever, answering
│   │   ├── ranking.py       # pure selection rules (unit tested, no deps)
│   │   ├── embedding.py
│   │   ├── store.py
│   │   ├── retriever.py
│   │   └── answering.py
│   └── eval/
│       ├── common.py        # pure helpers shared by all eval scripts
│       ├── eval_ragas.py    # the evaluation pipeline (--loop / --judge)
│       ├── eval_newv.py     # alias: eval_ragas with the loop enabled
│       ├── eval_langsmith.py# LLM judge over the answer cache
│       ├── gen_testset.py   # synthetic test-set generator
│       ├── ask_one.py       # one-question CLI
│       └── loop_run.py      # batch plan sweeper
├── serve_api.py             # FastAPI entrypoint
├── data/sources/            # source PDFs (see the layout in the README of that folder)
├── storage/chroma_db/       # generated index (git-ignored)
├── experiments/             # evaluation outputs (tracked on purpose)
├── tests/                   # pytest suite (no GPU / no model download needed)
└── docs/                    # optimisation log, security notes
```

---

## Development

```bash
make install     # pip install -e ".[serve,dev]"
make test        # pytest
make lint        # ruff check
make format      # ruff format
make ingest      # rebuild the index
make serve       # uvicorn
```

The test suite is intentionally dependency-light: the ranking rules, the score
normalisation and the configuration parser are pure Python, so `pytest` runs in
under a second on a bare interpreter and in CI without downloading models.

See [`docs/OPTIMIZATION.md`](docs/OPTIMIZATION.md) for what changed in v2 and
why, and [`docs/SECURITY.md`](docs/SECURITY.md) before making the repository
public again.

---

## 中文速览

- **定位**：昆士兰道路规则（Road Rules 2009 + 驾驶手册）的 Agentic RAG 问答系统，
  本地 Ollama + Chroma，无外部 API 依赖。
- **流程**：改写 → 检索 → 判定 → 生成，由 LangGraph 状态机编排，带步数上限，不会死循环。
- **评测**：`python -m src.eval.eval_ragas --limit 30 --full false`（快）或
  `--full true`（加 LLM 判分）；结果写入 `experiments/*.csv`。
- **服务**：`uvicorn serve_api:app --port 8008`，`POST /ask` 返回答案与引用页码。
- **注意**：`.env` 不再入库，请从 `.env.example` 复制并填自己的 key。

---

## License

MIT — see [LICENSE](LICENSE).
