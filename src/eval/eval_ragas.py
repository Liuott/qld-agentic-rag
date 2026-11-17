"""RAGAS evaluation for the QLD Agentic RAG corpus (no OpenAI dependency).

Two modes
---------
* default -- fast, non-LLM metrics only (context precision / recall against the
  reference contexts);
* ``--full`` -- additionally asks Ollama to score faithfulness and answer
  relevance.

Optional self-evaluation loop
-----------------------------
``--loop`` turns on the "critic" that ``eval_newv.py`` used to duplicate: after
an answer is produced, the evidence bullets are checked against the retrieved
context; if the hit ratio is too low (or ``--judge`` says the answer is not
grounded) the retrieval is widened and the answer is regenerated.  The variant
that scores better wins.

Outputs
-------
* ``data/eval/run_answers.csv`` -- answer cache (keyed by question + settings);
* ``experiments/ragas_row_scores.csv`` -- per-row context precision / recall;
* ``experiments/ab_results.csv`` -- one aggregated row per run.

Examples
--------
    python -m src.eval.eval_ragas --limit 30 --full false
    python -m src.eval.eval_ragas --row_idx 2 --full false
    python -m src.eval.eval_ragas --limit 30 --loop --judge
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

# ---- RAGAS (hard dependency of this script) -------------------------------- #
from datasets import Dataset
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings as LcHFEmb
from pandas.errors import EmptyDataError
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from tqdm import tqdm

from src.config import EVAL_DIR, EVAL_OLLAMA_MODEL, EXPERIMENTS_DIR, get_settings
from src.eval.common import (
    cache_key,
    evidence_hit_ratio,
    format_scores,
    is_bad_answer,
    nan_safe_scores,
    normalize_ragas_scores,
    shrink_contexts,
    str2bool,
)
from src.rag import answer_question, get_llm, get_vectorstore, retrieve_context

try:  # pragma: no cover - depends on the installed ragas version
    from ragas.metrics import NonLLMContextPrecisionWithReference, NonLLMContextRecall
except Exception:  # pragma: no cover
    from ragas.metrics import context_precision as NonLLMContextPrecisionWithReference  # type: ignore
    from ragas.metrics import context_recall as NonLLMContextRecall  # type: ignore

try:  # pragma: no cover
    from ragas.metrics import answer_relevancy, faithfulness
except Exception:  # pragma: no cover
    faithfulness = None  # type: ignore
    answer_relevancy = None  # type: ignore

try:  # pragma: no cover
    from ragas.llms import LangchainLLMWrapper as LangchainLLM
except Exception:  # pragma: no cover
    try:
        from ragas.integrations.langchain import LangchainLLM  # type: ignore
    except Exception:
        LangchainLLM = None  # type: ignore

# --------------------------------------------------------------------------- #
# Defaults (overridable through the environment)
# --------------------------------------------------------------------------- #
_settings = get_settings()

DEFAULT_LIMIT = int(os.getenv("LIMIT", "12"))
DEFAULT_TRIM_CTX = int(os.getenv("TRIM_CTX", "1200"))
DEFAULT_K = int(os.getenv("RETRIEVER_K", str(_settings.retriever_k)))
DEFAULT_PER_CHUNK = int(os.getenv("PER_CHUNK", "360"))
DEFAULT_FETCH_K = int(os.getenv("FETCH_K", "80"))
DEFAULT_THR = float(os.getenv("THR", "0.20"))
DEFAULT_FULL = str(os.getenv("FULL", "true")).lower() in ("1", "true", "yes")
DEFAULT_SHOW_DEBUG = str(os.getenv("SHOW_DEBUG", "true")).lower() in ("1", "true", "yes")
DEFAULT_RERANK = str(os.getenv("RERANK", "true")).lower() in ("1", "true", "yes")
DEFAULT_NO_CACHE = str(os.getenv("NO_CACHE", "false")).lower() in ("1", "true", "yes")

EVAL_MODEL = EVAL_OLLAMA_MODEL

TESTSET_CSV = EVAL_DIR / "testset.csv"
ANSWERS_CSV = EVAL_DIR / "run_answers.csv"
ROW_SCORES_CSV = EXPERIMENTS_DIR / "ragas_row_scores.csv"
AB_RESULTS_CSV = EXPERIMENTS_DIR / "ab_results.csv"

JUDGE_PROMPT = ChatPromptTemplate.from_template(
    """
You are a strict ground-truth judge.
Given the retrieved CONTEXT and the model's ANSWER (with Evidence bullets),
tell if the answer is fully grounded in the context.

Reply only JSON with keys:
- grounded: true/false
- reason: short text
- need_more_context: true/false

CONTEXT:
{context}

ANSWER:
{answer}
"""
)


# --------------------------------------------------------------------------- #
# Per-row export
# --------------------------------------------------------------------------- #
def _pick_column(df: pd.DataFrame, must_have: Sequence[str], fuzzy: Sequence[str]) -> str | None:
    lower_map = {column.lower(): column for column in df.columns}
    for name in must_have:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    for column in df.columns:
        lowered = column.lower()
        if all(needle in lowered for needle in fuzzy):
            return column
    return None


def export_per_row_scores(result: Any, questions: Sequence[str], out_csv: Path = ROW_SCORES_CSV) -> None:
    """Append per-row context precision / recall for error analysis."""
    import numpy as np

    try:
        frame = result.to_pandas()
    except Exception:
        if hasattr(result, "results") and isinstance(result.results, list):
            frame = pd.DataFrame(result.results)
        else:
            print("[warn] ragas per-row export: cannot obtain dataframe")
            return

    precision_col = _pick_column(
        frame,
        must_have=[
            "context_precision",
            "context_precision_with_reference",
            "nonllm_context_precision_with_reference",
            "non_llm_context_precision_with_reference",
        ],
        fuzzy=("context", "precision"),
    )
    recall_col = _pick_column(
        frame,
        must_have=[
            "context_recall",
            "context_recall_with_reference",
            "nonllm_context_recall",
            "non_llm_context_recall",
        ],
        fuzzy=("context", "recall"),
    )

    out = pd.DataFrame({"question": list(questions)})
    out["Context_precision"] = pd.to_numeric(frame[precision_col], errors="coerce") if precision_col else np.nan
    out["Context_recall"] = pd.to_numeric(frame[recall_col], errors="coerce") if recall_col else np.nan

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header_needed = True
    if out_csv.exists():
        try:
            header_needed = list(pd.read_csv(out_csv, nrows=0).columns) != list(out.columns)
        except Exception:
            header_needed = True
    out.to_csv(out_csv, mode="a", index=False, header=header_needed)
    print(f"[Saved] per-row RAGAS -> {out_csv} (precision_col={precision_col}, recall_col={recall_col})")


# --------------------------------------------------------------------------- #
# Optional LLM metrics
# --------------------------------------------------------------------------- #
def evaluate_llm_metric(
    dataset: Dataset,
    metric: Any,
    ragas_llm: Any,
    ragas_emb: Any,
    max_retries: int = 3,
    base_wait: float = 8.0,
) -> dict[str, float] | None:
    """Score one LLM-based metric with exponential backoff."""
    metric_key = getattr(metric, "name", None) or metric.__class__.__name__
    attempt = 0
    while True:
        try:
            result = evaluate(dataset, metrics=[metric], llm=ragas_llm, embeddings=ragas_emb)
            try:
                frame = result.to_pandas()
                numbers = frame.select_dtypes(include=["number"])
                aggregate = numbers.mean(numeric_only=True) if len(frame) > 1 else numbers.iloc[0]
                for key, value in aggregate.items():
                    if str(key).lower() == str(metric_key).lower():
                        return {metric_key: float(value)}
                for value in aggregate.values:
                    if isinstance(value, int | float):
                        return {metric_key: float(value)}
                return None
            except Exception:
                scores = getattr(result, "scores", {}) or {}
                if metric_key in scores:
                    return {metric_key: float(scores[metric_key])}
                return None
        except Exception as exc:
            attempt += 1
            if attempt > max_retries:
                print(f"[LLM metric] {metric_key} failed after retries: {exc}")
                return None
            wait = base_wait * (2 ** (attempt - 1)) + random.uniform(0, 2)
            print(f"[LLM metric] {metric_key} retry {attempt}/{max_retries} after {wait:.1f}s ({exc})")
            time.sleep(wait)


# --------------------------------------------------------------------------- #
# Self-evaluation loop
# --------------------------------------------------------------------------- #
def llm_judge(llm: Any, context_joined: str, answer: str) -> dict[str, Any]:
    """Ask the model whether the answer is grounded (fallback: assume ok)."""
    try:
        raw = (JUDGE_PROMPT | llm | StrOutputParser()).invoke({"context": context_joined, "answer": answer}).strip()
        try:
            payload = json.loads(raw)
        except Exception:
            payload = json.loads(raw.splitlines()[-1])
        return {
            "grounded": bool(payload.get("grounded", False)),
            "need_more_context": bool(payload.get("need_more_context", False)),
            "reason": str(payload.get("reason", ""))[:200],
        }
    except Exception:
        return {"grounded": True, "need_more_context": False, "reason": "judge-fallback"}


def retry_with_broader_retrieval(vectorstore: Any, question: str, args: argparse.Namespace):
    """Widen the search: lower threshold, bigger pool, de-duplication + rerank."""
    return retrieve_context(
        vectorstore,
        question,
        k=args.k,
        trim=args.trim_ctx,
        per_chunk=args.per_chunk,
        use_mmr=True,
        mmr_lambda=0.70,
        fetch_k=max(args.fetch_k, 120),
        score_threshold=min(args.thr, 0.10) if args.thr is not None else None,
        rerank=True,
        cross_encoder_name=args.ce_model,
    )


def improve_answer(llm: Any, vectorstore: Any, question: str, answer: str, chunks, joined, args):
    """Run one critic pass; return ``(answer, chunks, joined)``."""
    hit = evidence_hit_ratio(joined, answer)
    need_retry = hit < args.critic_thr
    if need_retry and args.judge:
        verdict = llm_judge(llm, joined, answer)
        need_retry = (not verdict.get("grounded", True)) or verdict.get("need_more_context", False)
    if not need_retry:
        return answer, chunks, joined

    wider_chunks, wider_joined = retry_with_broader_retrieval(vectorstore, question, args)
    if not wider_joined:
        return answer, chunks, joined

    retried = answer_question(llm, wider_joined, question)
    if evidence_hit_ratio(wider_joined, retried) >= hit:
        return retried, wider_chunks, wider_joined
    return answer, chunks, joined


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #
_CACHE_COLUMNS = ["question", "answer", "context", "ckey"]


def load_answer_cache(path: Path) -> dict[str, tuple[str, str, str]]:
    """Read the answer cache, tolerating legacy / empty / broken files."""
    cache: dict[str, tuple[str, str, str]] = {}
    if not path.exists():
        return cache
    try:
        if path.stat().st_size == 0:
            raise EmptyDataError("empty run_answers.csv")
        frame = pd.read_csv(path, engine="python", on_bad_lines="skip")
    except EmptyDataError:
        return cache
    except Exception:
        return cache

    if frame.shape[1] >= 3:
        frame = frame.iloc[:, :4].copy()
        if frame.shape[1] == 3:
            frame.columns = ["question", "answer", "context"]
            frame["ckey"] = ""
        else:
            frame.columns = _CACHE_COLUMNS
    else:
        return cache

    for row in frame.itertuples(index=False):
        cache[str(row.question)] = (str(row.answer), str(row.context), str(row.ckey))
    return cache


def append_answer_cache(path: Path, rows: list[dict[str, str]]) -> None:
    """Append new answers, migrating a legacy header in place when needed."""
    if not rows:
        return
    frame = pd.DataFrame(rows, columns=_CACHE_COLUMNS)
    exists = path.exists()
    legacy = False
    if exists:
        try:
            header = path.read_text(encoding="utf-8").splitlines()[0]
            legacy = "ckey" not in [part.strip() for part in header.split(",")]
        except Exception:
            legacy = False

    path.parent.mkdir(parents=True, exist_ok=True)
    if exists and legacy:
        try:
            old = pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            old = pd.DataFrame(columns=_CACHE_COLUMNS)
        if old.shape[1] == 3:
            old.columns = ["question", "answer", "context"]
            old["ckey"] = ""
        pd.concat([old, frame], ignore_index=True).to_csv(path, index=False, quoting=csv.QUOTE_ALL)
    else:
        frame.to_csv(path, mode="a", index=False, header=not exists, quoting=csv.QUOTE_ALL)
    print(f"[cache] appended {len(rows)} rows -> {path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser(default_loop: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--trim_ctx", type=int, default=DEFAULT_TRIM_CTX)
    parser.add_argument("--full", type=str2bool, nargs="?", const=True, default=DEFAULT_FULL)
    parser.add_argument("--per_chunk", type=int, default=DEFAULT_PER_CHUNK)
    parser.add_argument("--thr", type=float, default=DEFAULT_THR)
    parser.add_argument("--fetch_k", type=int, default=DEFAULT_FETCH_K)
    parser.add_argument("--mmr", action="store_true", default=False)
    parser.add_argument("--mmr_lambda", type=float, default=0.85)
    parser.add_argument("--rerank", type=str2bool, nargs="?", const=True, default=DEFAULT_RERANK)
    parser.add_argument("--ce_model", type=str, default=_settings.cross_encoder_model)
    parser.add_argument("--show_debug", action="store_true", default=DEFAULT_SHOW_DEBUG)
    parser.add_argument("--row_idx", type=int, default=None)
    parser.add_argument("--question_substr", type=str, default=None)
    parser.add_argument("--no_cache", action="store_true", default=DEFAULT_NO_CACHE)
    parser.add_argument("--loop", action="store_true", default=default_loop, help="enable the self-evaluation loop")
    parser.add_argument("--critic_thr", type=float, default=0.80, help="evidence hit ratio required to accept")
    parser.add_argument("--judge", action="store_true", default=False, help="also ask the LLM judge")
    parser.add_argument("--testset", type=Path, default=TESTSET_CSV)
    return parser


def select_rows(testsets: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if not {"question", "ground_truth"}.issubset(testsets.columns):
        raise SystemExit(
            f"{args.testset} must contain 'question' and 'ground_truth' columns, found: {list(testsets.columns)}"
        )
    if args.row_idx is not None:
        if not 0 <= args.row_idx < len(testsets):
            raise IndexError(f"--row_idx out of range (0..{len(testsets) - 1})")
        selected = testsets.iloc[[args.row_idx]].copy()
        print(f"[single] row_idx={args.row_idx} -> Q: {selected.iloc[0]['question']}")
        return selected
    if args.question_substr:
        matched = testsets[testsets["question"].astype(str).str.contains(args.question_substr, case=False, na=False)]
        if matched.empty:
            raise ValueError(f"--question_substr not matched: {args.question_substr}")
        selected = matched.iloc[[0]].copy()
        print(f"[single] question_substr={args.question_substr!r} -> Q: {selected.iloc[0]['question']}")
        return selected
    return testsets.head(args.limit).copy()


def debug_sample(question: str, chunks: Sequence[str], answer: str, topn: int = 3, each_len: int = 280) -> None:
    lines = ["\n" + "=" * 72, "[DEBUG] Question:", question.strip(), "\n[DEBUG] Top contexts:"]
    for index, chunk in enumerate(chunks[:topn], 1):
        lines.append(f"  [{index}] {(chunk or '')[:each_len].rstrip()}")
    lines += ["\n[DEBUG] Answer:", (answer or "").strip(), "=" * 72]
    text = "\n".join(lines)
    print(text)
    try:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        (EVAL_DIR / "last_debug.txt").write_text(text, encoding="utf-8")
    except Exception:
        pass


def main(argv: Sequence[str] | None = None, *, default_loop: bool = False) -> int:
    args = build_parser(default_loop=default_loop).parse_args(argv)

    testsets = pd.read_csv(args.testset)
    frame = select_rows(testsets, args)

    vectorstore = get_vectorstore()
    llm = get_llm(model=EVAL_MODEL, temperature=0.0, num_predict=320)
    ragas_embeddings = LangchainEmbeddingsWrapper(LcHFEmb(model_name=_settings.embedding_model))

    cache = {} if args.no_cache else load_answer_cache(ANSWERS_CSV)

    rows_out: list[dict[str, str]] = []
    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []
    reference_contexts: list[list[str]] = []

    for _, row in tqdm(frame.iterrows(), total=len(frame), desc="[answer]"):
        question = str(row["question"])
        chunks, joined = retrieve_context(
            vectorstore,
            question,
            k=args.k,
            trim=args.trim_ctx,
            per_chunk=args.per_chunk,
            use_mmr=args.mmr,
            mmr_lambda=args.mmr_lambda,
            fetch_k=args.fetch_k,
            score_threshold=args.thr,
            rerank=args.rerank,
            cross_encoder_name=args.ce_model,
        )

        key = cache_key(question, args, EVAL_MODEL)
        cached = cache.get(question)
        use_cache = bool(
            (not args.no_cache)
            and cached
            and cached[0]
            and not is_bad_answer(cached[0])
            and (cached[2] == "" or cached[2] == key)
        )

        if use_cache:
            answer = cached[0]
        else:
            answer = answer_question(llm, joined, question)
            if args.loop:
                answer, chunks, joined = improve_answer(llm, vectorstore, question, answer, chunks, joined, args)
            rows_out.append({"question": question, "answer": answer, "context": joined, "ckey": key})

        if args.show_debug:
            debug_sample(question, chunks, answer, topn=3, each_len=args.per_chunk)

        questions.append(question)
        answers.append(answer)
        contexts.append(chunks)
        ground_truths.append(str(row["ground_truth"]))

        raw_reference = row.get("ref_context", None)
        reference = "" if raw_reference is None or pd.isna(raw_reference) else str(raw_reference).strip()
        reference_contexts.append([reference] if reference else list(chunks))

    append_answer_cache(ANSWERS_CSV, rows_out)

    dataset = Dataset.from_dict(
        {
            "user_input": questions,
            "retrieved_contexts": contexts,
            "reference_contexts": reference_contexts,
            "response": answers,
            "reference": ground_truths,
        }
    )

    os.environ.setdefault("RAGAS_CONCURRENCY", "1")
    os.environ.setdefault("RAGAS_TIMEOUT", "1200")

    extra_scores: dict[str, float] = {}
    if args.full and LangchainLLM is not None and faithfulness is not None and answer_relevancy is not None:
        ragas_llm = LangchainLLM(llm)
        dataset_llm = Dataset.from_dict(
            {
                "user_input": questions,
                "retrieved_contexts": shrink_contexts(contexts, topn=3, each_len=320),
                "reference_contexts": shrink_contexts(reference_contexts, topn=3, each_len=320),
                "response": answers,
                "reference": ground_truths,
            }
        )
        extra_scores = {
            **(evaluate_llm_metric(dataset_llm, faithfulness, ragas_llm, ragas_embeddings) or {}),
            **(evaluate_llm_metric(dataset_llm, answer_relevancy, ragas_llm, ragas_embeddings) or {}),
        }

    metrics = [NonLLMContextPrecisionWithReference(), NonLLMContextRecall()]
    result = evaluate(dataset, metrics=metrics, embeddings=ragas_embeddings)
    export_per_row_scores(result, questions)

    scores: dict[str, float] = {}
    try:
        frame_scores = result.to_pandas()
        numbers = frame_scores.select_dtypes(include=["number"])
        aggregate = numbers.mean(numeric_only=True) if len(frame_scores) > 1 else numbers.iloc[0]
        scores = {str(key): float(value) for key, value in aggregate.items()}
    except Exception:
        if hasattr(result, "scores"):
            scores = {str(key): float(value) for key, value in result.scores.items()}

    raw_scores = nan_safe_scores({**extra_scores, **scores})
    print("\n" + format_scores(raw_scores, "RAGAS Scores (raw)"))
    normalized = normalize_ragas_scores(raw_scores)
    print(
        "\n"
        + format_scores(
            {
                k: normalized[k]
                for k in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
                if k in normalized
            },
            "RAGAS Scores (standardized)",
        )
    )

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not AB_RESULTS_CSV.exists()
    with AB_RESULTS_CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "ts",
                "embed_model",
                "retriever_k",
                "limit",
                "trim_ctx",
                "eval_model",
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
            ],
        )
        if is_new:
            writer.writeheader()
        writer.writerow(
            {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "embed_model": _settings.embedding_model,
                "retriever_k": args.k,
                "limit": len(frame),
                "trim_ctx": args.trim_ctx,
                "eval_model": EVAL_MODEL,
                "faithfulness": f"{normalized.get('faithfulness', 0.0):.4f}",
                "answer_relevancy": f"{normalized.get('answer_relevancy', 0.0):.4f}",
                "context_precision": f"{normalized.get('context_precision', 0.0):.4f}",
                "context_recall": f"{normalized.get('context_recall', 0.0):.4f}",
            }
        )
    print(f"[Saved] {AB_RESULTS_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
