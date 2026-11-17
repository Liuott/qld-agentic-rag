"""Batch driver: evaluate one row, grade it, retry with a wider retrieval plan.

This is the "experiment sweeper".  It shells out to the two eval modules so each
subprocess gets a clean interpreter (Ollama clients and RAGAS hold a lot of
state).

The old version had a subtle bug: it invoked ``sys.executable`` but relied on
being launched from the repository root, and it hard-coded
``"src.eval.eval_ragas"`` strings plus relative CSV paths.  Paths and modules
now come from :mod:`src.config`, and the subprocess runs with ``cwd`` pinned to
the repository root.

    python -m src.eval.loop_run --start 0 --limit 20 --max_steps 3
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import pandas as pd

from src.config import EXPERIMENTS_DIR, REPO_ROOT

ANSWER_MODULE = "src.eval.eval_ragas"
JUDGE_MODULE = "src.eval.eval_langsmith"

SCORE_CSV = EXPERIMENTS_DIR / "final_scores.csv"

#: Progressive retrieval plans: (threshold, fetch_k, mmr, rerank, trim_ctx).
PLANS: tuple[dict[str, object], ...] = (
    {"thr": 0.20, "fetch_k": 80, "mmr": False, "rerank": True, "trim_ctx": 1200},
    {"thr": 0.12, "fetch_k": 160, "mmr": True, "rerank": True, "trim_ctx": 1400},
    {"thr": 0.00, "fetch_k": 240, "mmr": True, "rerank": True, "trim_ctx": 1800},
)


def run_module(module: str, extra: list[str]) -> None:
    subprocess.run([sys.executable, "-m", module, *extra], check=True, cwd=REPO_ROOT)


def run_answer(row_idx: int, plan: dict[str, object]) -> None:
    extra = ["--row_idx", str(row_idx), "--thr", str(plan["thr"]), "--fetch_k", str(plan["fetch_k"])]
    if plan.get("mmr"):
        extra.append("--mmr")
    if plan.get("rerank"):
        extra.append("--rerank")
    extra += ["--no_cache", "--trim_ctx", str(plan["trim_ctx"])]
    run_module(ANSWER_MODULE, extra)


def run_judge() -> tuple[bool, str]:
    """Grade the most recent cached answer; returns ``(faithful, relevance)``."""
    run_module(JUDGE_MODULE, ["--limit", "1", "--tail"])
    frame = pd.read_csv(SCORE_CSV)
    last = frame.iloc[-1]
    faithful = str(last["Faithfulness"]).strip().lower() in ("true", "1", "yes")
    relevance = str(last["Relevance"]).strip().lower()
    return faithful, relevance


def score(ok: bool, similarity: str) -> float:
    """Reward signal used to pick the best plan."""
    return (1.0 if ok else 0.0) + (0.5 if similarity == "high" else 0.2 if similarity == "medium" else 0.0)


def loop_one(row_idx: int, max_steps: int = 3) -> dict[str, object]:
    best: dict[str, object] = {"idx": row_idx, "step": -1, "ok": False, "score": -1.0}
    for step, plan in enumerate(PLANS[:max_steps], 1):
        run_answer(row_idx, plan)
        faithful, relevance = run_judge()
        current = score(faithful, relevance)
        accepted = faithful and relevance != "low"
        if current > float(best["score"]):
            best.update({"step": step, "ok": accepted, "score": current})
        print(f"[row {row_idx}] step {step}: faithful={faithful}, relevance={relevance}, score={current:.2f}")
        if best["ok"]:
            break
    return best


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description="Sweep retrieval plans over a test set.")
    cli.add_argument("--limit", type=int, default=20)
    cli.add_argument("--start", type=int, default=0)
    cli.add_argument("--max_steps", type=int, default=3)
    args = cli.parse_args(argv)

    results: list[dict[str, object]] = []
    for index in range(args.start, args.start + args.limit):
        try:
            results.append(loop_one(index, max_steps=args.max_steps))
        except subprocess.CalledProcessError as exc:
            print(f"[row {index}] subprocess failed: {exc}")
        except Exception as exc:
            print(f"[row {index}] error: {exc}")

    passed = sum(1 for result in results if result["ok"])
    average = sum(float(result["score"]) for result in results) / max(1, len(results))
    print(f"\n[summary] passed={passed}/{len(results)}; avg score={average:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
