# -*- coding: utf-8 -*-
"""
LangSmith-based evaluator (Ollama-only, strict JSON grading)

- Reads Q/A/context from data/eval/run_answers.csv (created by eval_ragas.py)
- Grades with an Ollama model via a strict-JSON prompt (no external dependency)
- Saves scores to experiments/langsmith_scores.csv
- Optionally logs each example to a LangSmith dataset (create if missing, else reuse)

Env vars:
  OLLAMA_BASE_URL (default: http://localhost:11434)
  OLLAMA_MODEL    (default: qwen2.5:7b)
  LANGSMITH_TRACING (true/false/1/0) + LANGSMITH_API_KEY
  LS_DATASET      (default: qld_rr_eval_demo)
  RUN_TAG         (default: manual)

Columns expected in data/eval/run_answers.csv:
  - question
  - answer
  - context
Extra columns are ignored; bad lines are skipped safely.
"""

from __future__ import annotations
import os
import json
from typing import Dict, List
import argparse
import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# --- Optional LangSmith (guarded by env) ---
try:
    from langsmith import client as ls_client
except Exception:
    ls_client = None  # allow code to run without langsmith package


# =========================
# Config & Prompt
# =========================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

LS_DATASET      = os.getenv("LS_DATASET", "qld_rr_eval_demo")
RUN_TAG         = os.getenv("RUN_TAG", "manual")

LANGSMITH_TRACING = str(os.getenv("LANGSMITH_TRACING", "")).lower() in ("1", "true", "yes")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
STRICT_RELEVANCE = True
EVAL_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system",
     "You are a STRICT JSON grader. Output ONE JSON object ONLY. "
     "No prose, no code fences, no pre/post text."),
    ("human",
     "Judge ONLY using the provided context.\n\n"
     "Question:\n{question}\n\n"
     "Context:\n{context}\n\n"
     "Answer:\n{answer}\n\n"
     "Rules:\n"
     "- correctness=true only if the final conclusion exactly matches the rule implied by the context.\n"
     "- faithfulness=true only if EVERY claim in the answer is directly supported by the context (verbatim or unambiguous paraphrase).\n"
     "- relevance must be one of: high, medium, low (measure how on-topic the answer is wrt the question+context).\n"
     "- If the context lacks the needed rule or contradicts the answer, set correctness=false and faithfulness=false.\n"
     "- Return JSON ONLY in this schema:\n"
     '{{\"correctness\":true|false,\"faithfulness\":true|false,\"relevance\":\"high\"|\"medium\"|\"low\",\"rationale\":\"<short reason>\"}}'
    )
])


# =========================
# Helpers
# =========================
def _clip(text: str, max_len: int = 2400) -> str:
    """Clip long context/answers to keep evaluation snappy."""
    t = (text or "")
    return t if len(t) <= max_len else (t[:max_len] + " …")

def _safe_bool(v, default=False) -> bool:
    try:
        return bool(v)
    except Exception:
        return default

def _normalize_grade(obj: Dict) -> Dict:
    """Ensure fields exist & are valid types; optionally enforce stricter coupling to relevance."""
    correctness = _safe_bool(obj.get("correctness"), False)
    faithfulness = _safe_bool(obj.get("faithfulness"), False)
    relevance = str(obj.get("relevance", "low")).lower().strip()
    if relevance not in ("high", "medium", "low"):
        relevance = "low"
    rationale = str(obj.get("rationale", "")).strip()[:600]

    if STRICT_RELEVANCE and relevance == "low":
        correctness = False
        faithfulness = False

    return {
        "correctness": correctness,
        "faithfulness": faithfulness,
        "relevance": relevance,
        "rationale": rationale,
    }

def _create_or_get_dataset(client, name: str):
    """Create dataset if not exists; else reuse."""
    try:
        return client.create_dataset(dataset_name=name)
    except Exception:
        return client.read_dataset(dataset_name=name)


def _load_rows(csv_path: str) -> List[Dict]:
    """Load question/context/answer triads from CSV, skipping bad lines."""
    if not os.path.exists(csv_path):
        # Fallback single sample if file not present
        return [{
            "question": "At a T-intersection in Queensland, who has right of way?",
            "context": "(Put retrieved clauses here for your real run)",
            "answer": "The driver on the terminating road must give way to drivers on the continuing road.",
        }]

    # Robust read: allow python engine; skip bad lines; ignore extra columns
    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
    cols = [c for c in df.columns]
    need = {"question", "answer", "context"}
    if not need.issubset(cols):
        # Try alternate capitalizations or typos if needed
        raise ValueError(f"[read_csv] Missing required columns {need} in {csv_path}. Got: {cols}")

    rows = []
    for _, r in df.iterrows():
        q = str(r.get("question", "") or "")
        a = str(r.get("answer", "") or "")
        c = str(r.get("context", "") or "")
        if not q.strip() or not a.strip():
            continue
        rows.append({"question": q, "context": c, "answer": a})
    return rows

def _get_latest_ragas_for_question(q: str, ragas_csv: str = "experiments/ragas_row_scores.csv") -> dict:
    try:
        if not os.path.exists(ragas_csv):
            return {}
        import pandas as pd
        df = pd.read_csv(ragas_csv)
        if "question" not in df.columns:
            return {}
        m = df[df["question"] == q]
        if m.empty:
            return {}
        row = m.iloc[-1]
        out = {}
        if "Context_precision" in row:
            out["Context_precision"] = float(row["Context_precision"])
        if "Context_recall" in row:
            out["Context_recall"] = float(row["Context_recall"])
        return out
    except Exception:
        return {}

# =========================
# Core grading
# =========================
def grade_one(llm: ChatOllama, question: str, context: str, answer: str) -> Dict:
    """Call Ollama LLM to produce a strict JSON grade; return normalized dict."""
    msgs = EVAL_TEMPLATE.format_messages(
        question=question,
        context=_clip(context, 2800),
        answer=_clip(answer, 1600),
    )

    # 第一次尝试（有 format="json" 时通常能成功）
    resp = llm.invoke(msgs)
    try:
        raw = json.loads(resp.content)
        return _normalize_grade(raw)
    except Exception as e:
        # 第二次更强约束地重试
        retry_template = ChatPromptTemplate.from_messages([
            ("system", "Output one JSON object only. No extra text."),
            ("human",
             'Return EXACTLY a JSON with keys ["correctness","faithfulness","relevance","rationale"].\n'
             'Values: correctness (true/false), faithfulness (true/false), relevance ("high"|"medium"|"low"), '
             'rationale (short string). No trailing commas.\n\n'
             "Question:\n{question}\n\nContext:\n{context}\n\nAnswer:\n{answer}\n"
            ),
        ])
        retry_msgs = retry_template.format_messages(
            question=question,
            context=_clip(context, 2200),
            answer=_clip(answer, 1200),
        )
        resp2 = llm.invoke(retry_msgs)
        try:
            raw2 = json.loads(resp2.content)
            return _normalize_grade(raw2)
        except Exception as e2:
            fallback = {
                "correctness": False,
                "faithfulness": False,
                "relevance": "low",
                "rationale": f"parse_error_after_retry: {type(e2).__name__}"
            }
            return _normalize_grade(fallback)

# =========================
# Main
# =========================
def main():
    # --- CLI 过滤选项 ---
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="limit number of rows to grade")
    ap.add_argument("--row_idx", type=int, default=None, help="grade only the single row at this 0-based index")
    ap.add_argument("--question_substr", type=str, default=None, help="grade the first row whose question contains this substring (case-insensitive)")
    ap.add_argument("--tail", action="store_true", help="为 True 时取最后 N 条而不是最前 N 条")
    args = ap.parse_args()

    # 1) Load data
    rows = _load_rows("data/eval/run_answers.csv")

    # --- 应用过滤 ---
    if args.limit is not None and args.limit > 0:
        rows = rows[-args.limit:] if args.tail else rows[:args.limit]
    if args.row_idx is not None:
        if not (0 <= args.row_idx < len(rows)):
            raise IndexError(f"--row_idx out of range (0..{len(rows)-1})")
        rows = [rows[args.row_idx]]
        print(f"[filter] using row_idx={args.row_idx} → Q: {rows[0]['question'][:120]}")

    elif args.question_substr:
        tgt = args.question_substr.lower()
        pick = None
        for r in rows:
            if tgt in (r["question"] or "").lower():
                pick = r
                break
        if pick is None:
            raise ValueError(f"--question_substr not matched: {args.question_substr}")
        rows = [pick]
        print(f"[filter] using question_substr='{args.question_substr}' → Q: {rows[0]['question'][:120]}")

    elif args.limit is not None:
        rows = rows[:max(0, args.limit)]
        print(f"[filter] using limit={len(rows)}")

    # 2) Build LLM
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",
        temperature=0.0,
        format="json",
        seed=0,
    )
    # 3) Optional LangSmith client
    ls = None
    if LANGSMITH_TRACING and LANGSMITH_API_KEY and ls_client is not None:
        try:
            ls = ls_client.Client(api_key=LANGSMITH_API_KEY)
            dataset = _create_or_get_dataset(ls, LS_DATASET)
            print(f"[LangSmith] dataset ready: {LS_DATASET}")
        except Exception as e:
            print(f"[LangSmith] init failed: {e}")
            ls = None
    else:
        print("[LangSmith] Skipped (set LANGSMITH_TRACING=true and provide LANGSMITH_API_KEY)")

    # 4) Grade all
    graded = []
    for s in rows:
        g = grade_one(llm, s["question"], s["context"], s["answer"])
        rag = _get_latest_ragas_for_question(s["question"])

        final_rec = {
            "Question": s["question"],
            "Relevance": str(g["relevance"]).lower(),      # exactly the column name you want
            "Faithfulness": bool(g["faithfulness"]),
            "Context_precision": rag.get("Context_precision", None),
            "Context_recall": rag.get("Context_recall", None),
        }
        # 继续保留原来的 langsmith_scores.csv
        # 另存“最终四列”到统一文件
        os.makedirs("experiments", exist_ok=True)
        final_csv = "experiments/final_scores.csv"
        pd.DataFrame([final_rec]).to_csv(final_csv, mode="a", index=False, header=not os.path.exists(final_csv))
        print(f"[final] {final_rec}")

        rec = {**s, **g}
        graded.append(rec)

        # stream to console (optional)
        print("[scores]")
        print(json.dumps(rec, ensure_ascii=False, indent=2))

        # optionally write to LangSmith
        if ls is not None:
            try:
                ls.create_example(
                inputs={"question": s["question"], "context": s["context"]},
                outputs={"answer": s["answer"], "grade": g},
                dataset_id=dataset.id, 
                metadata={"source": "rag_eval", "run": RUN_TAG},
            )

            except Exception as e:
                print(f"[LangSmith] write example failed: {e}")

    # 5) Save local CSV
    os.makedirs("experiments", exist_ok=True)
    out_path = "experiments/langsmith_scores.csv"
    pd.DataFrame(graded).to_csv(out_path, index=False)
    print(f"[done] wrote {out_path}")

    # 6) Small summary
    try:
        df = pd.DataFrame(graded)
        corr = float(df["correctness"].mean()) if "correctness" in df else 0.0
        faith = float(df["faithfulness"].mean()) if "faithfulness" in df else 0.0
        rel = df["relevance"].value_counts(normalize=True).to_dict() if "relevance" in df else {}
        print(f"[summary] correctness={corr:.3f}, faithfulness={faith:.3f}, relevance_dist={rel}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
