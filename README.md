# Agentic RAG for Queensland Road Rules

## Quick Start
1)python -m venv .venv && source .venv/bin/activate
2)pip install -r requirements.txt
3)python -u -m src.ingest
4)python -u -m src.app_graph
 What should I do when crossing the road?
5)python -u -m src.eval.gen_testset --n 30

6)
export EVAL_OLLAMA_MODEL=qwen2.5:3b
export RAGAS_CONCURRENCY=1
export RAGAS_TIMEOUT=600


python -u -m src.eval.eval_ragas --full false --row_idx 

7)

python -u -m src.eval.eval_langsmith --limit 30 -tail

eval index2）：
python -u -m src.eval.eval_langsmith --row_idx 

fliter by key word：
python -u -m src.eval.eval_langsmith --question_substr "site access only"


8) quantitative analysis
base() {
  python -u -m src.eval.eval_ragas \
    --limit 30 \
    --k 6 \
    --trim_ctx 1200 \
    --per_chunk 360 \
    --fetch_k 60 \
    --full \
    --show_debug \
    --no_cache \
    "$@"
}

9)loop run
python -u -m src.eval.loop_run --limit 30 --start 0 --max_steps 3

10) connect to openwebui
sudo fuser -k 8008/tcp || true
export PYTHONPATH=$(pwd):$PYTHONPATH 
python -m uvicorn serve_api:app --host 0.0.0.0 --port 8008 --reload