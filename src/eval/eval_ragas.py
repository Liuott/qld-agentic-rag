# -*- coding: utf-8 -*-
"""
Stable RAGAS evaluation for Agentic RAG (no OpenAI dependency by default)
"""

from __future__ import annotations
import os, csv, math, argparse, time, random
from typing import List, Tuple
import pandas as pd
from pandas.errors import EmptyDataError
from tqdm import tqdm
from datasets import Dataset

from langchain_huggingface import HuggingFaceEmbeddings as LCHFEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ragas import evaluate

# ---- Non-LLM metrics (no LLM dependency) ----
try:
    from ragas.metrics import NonLLMContextPrecisionWithReference, NonLLMContextRecall
except Exception:
    from ragas.metrics import context_precision as NonLLMContextPrecisionWithReference
    from ragas.metrics import context_recall as NonLLMContextRecall

# ---- Optional LLM metrics (only used with --full) ----
try:
    from ragas.metrics import faithfulness, answer_relevancy
except Exception:
    faithfulness = None
    answer_relevancy = None

# ---- RAGAS-native HF embeddings (prevents defaulting to OpenAI) ----
from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmbeddings

# ---- Wrap LangChain LLM for RAGAS (only for --full) ----
try:
    from ragas.llms import LangchainLLMWrapper as LangchainLLM
except Exception:
    try:
        from ragas.integrations.langchain import LangchainLLM as LangchainLLM
    except Exception:
        LangchainLLM = None  # if missing, disallow --full

from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings as LcHFEmb

# ---- Optional Cross-Encoder (re-rank) ----
try:
    from sentence_transformers import CrossEncoder
    CE_AVAILABLE = True
except Exception:
    CE_AVAILABLE = False

from src.config import CHROMA_DIR, EMBEDDING_MODEL, OLLAMA_BASE_URL, OLLAMA_MODEL

# ---------------- Defaults ----------------
DEFAULT_LIMIT = int(os.getenv("LIMIT", "12"))
DEFAULT_TRIM_CTX = int(os.getenv("TRIM_CTX", "1200"))
DEFAULT_K = int(os.getenv("RETRIEVER_K", "6"))
DEFAULT_PER_CHUNK = int(os.getenv("PER_CHUNK", "360"))
DEFAULT_FETCH_K = int(os.getenv("FETCH_K", "80"))
DEFAULT_THR = float(os.getenv("THR", "0.20"))
DEFAULT_FULL = str(os.getenv("FULL", "true")).lower() in ("1","true","yes")
DEFAULT_SHOW_DEBUG = str(os.getenv("SHOW_DEBUG", "true")).lower() in ("1","true","yes")
DEFAULT_RERANK = str(os.getenv("RERANK", "true")).lower() in ("1","true","yes")
DEFAULT_NO_CACHE = str(os.getenv("NO_CACHE", "false")).lower() in ("1","true","yes")

EVAL_MODEL = os.getenv("EVAL_OLLAMA_MODEL", OLLAMA_MODEL)  # LLM for answering / --full scoring

RAG_PROMPT = ChatPromptTemplate.from_template("""
Answer strictly from the context. If something is not in the context, say you don't know.

Respond in two sections:
1) Evidence: quote the exact lines (verbatim) from the context that support your answer. Use bullet points and keep each quote short.
2) Answer: a concise synthesis. End with: "Source: <Act/MUTCD, clause #>" when the clause is explicitly present.

--------------------
{context}
--------------------
Question: {question}
""")

# ---------------- Utils ----------------
def str2bool(v) -> bool:
    if isinstance(v, bool): return v
    s = str(v).strip().lower()
    if s in ("1","true","yes","y","on"):  return True
    if s in ("0","false","no","n","off"): return False
    raise argparse.ArgumentTypeError(f"expected boolean, got: {v}")

def answer_question(llm, ctx_joined: str, q: str) -> str:
    chain = RAG_PROMPT | llm | StrOutputParser()
    return chain.invoke({"context": ctx_joined, "question": q})

def _cache_key(q: str, args) -> str:
    return "||".join([
        q.strip(),
        f"eval_model={EVAL_MODEL}",
        f"k={args.k}",
        f"trim={args.trim_ctx}",
        f"chunk={args.per_chunk}",
        f"thr={args.thr}",
        f"rerank={args.rerank}",
        f"mmr={args.mmr}",
    ])

def _bad_answer(a: str) -> bool:
    a = (a or "").strip()
    if len(a) < 50: return True
    if "Evidence:" not in a or "Answer:" not in a: return True
    return False

def nan_safe_scores(d: dict) -> dict:
    out = {}
    for k, v in (d or {}).items():
        try: fv = float(v)
        except Exception: fv = 0.0
        if math.isnan(fv) or math.isinf(fv): fv = 0.0
        out[k] = fv
    return out

def normalize_ragas_scores(scores: dict) -> dict:
    import math as _math
    clean = {}
    for k, v in (scores or {}).items():
        try: fv = float(v)
        except Exception: fv = 0.0
        if _math.isnan(fv) or _math.isinf(fv): fv = 0.0
        clean[k] = fv

    alias_table = {
        "context_precision": [
            "context_precision","context_precision_with_reference",
            "nonllm_context_precision_with_reference","non_llm_context_precision_with_reference",
            "retrieval_precision_with_reference",
        ],
        "context_recall": [
            "context_recall","context_recall_with_reference",
            "nonllm_context_recall","non_llm_context_recall",
            "retrieval_recall_with_reference",
        ],
        "faithfulness": ["faithfulness","llm_faithfulness"],
        "answer_relevancy": ["answer_relevancy","response_relevancy","llm_answer_relevancy"],
    }
    lower_map = {k.lower(): k for k in clean.keys()}
    normalized = dict(clean)
    for canon_key, aliases in alias_table.items():
        for a in aliases:
            if a.lower() in lower_map:
                normalized[canon_key] = clean[lower_map[a.lower()]]
                break

    def fuzzy_pick(pred):
        for orig_k in clean.keys():
            if pred(orig_k.lower()):
                return clean[orig_k]
        return None

    if "context_precision" not in normalized:
        v = fuzzy_pick(lambda lk: ("context" in lk) and ("precision" in lk))
        if v is not None: normalized["context_precision"] = v
    if "context_recall" not in normalized:
        v = fuzzy_pick(lambda lk: ("context" in lk) and ("recall" in lk))
        if v is not None: normalized["context_recall"] = v

    print("\n=== RAGAS Scores (raw) ===")
    for k in sorted(clean.keys()):
        print(f"{k}: {clean[k]:.4f}")

    print("\n=== RAGAS Scores (standardized) ===")
    for k in ("faithfulness","answer_relevancy","context_precision","context_recall"):
        if k in normalized:
            print(f"{k}: {normalized[k]:.4f}")
    return normalized

def _eval_one_metric(ds_llm, metric, ragas_llm, ragas_emb, max_retries=2, base_wait=8):
    from ragas import evaluate as ragas_evaluate
    metric_key = getattr(metric, "name", None) or metric.__class__.__name__
    attempt = 0
    while True:
        try:
            res = ragas_evaluate(ds_llm, metrics=[metric], llm=ragas_llm, embeddings=ragas_emb)
            try:
                dfm = res.to_pandas()
                num = dfm.select_dtypes(include=["number"])
                agg = num.mean(numeric_only=True) if len(dfm) > 1 else num.iloc[0]
                for k, v in agg.items():
                    if k.lower() == metric_key.lower():
                        return {metric_key: float(v)}
                for k, v in agg.items():
                    if isinstance(v, (int, float)):
                        return {metric_key: float(v)}
                return None
            except Exception:
                if hasattr(res, "scores") and metric_key in getattr(res, "scores", {}):
                    return {metric_key: float(res.scores.get(metric_key))}
                return None
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                print(f"[LLM metric] {metric_key} failed after retries: {e}")
                return None
            wait = base_wait * (2 ** (attempt - 1)) + random.uniform(0, 2)
            print(f"[LLM metric] {metric_key} retry {attempt}/{max_retries} after {wait:.1f}s ... ({e})")
            time.sleep(wait)

# ---------------- Retrieval (threshold / MMR / rerank) ----------------
def retrieve_context(
    vs, q: str, k: int, trim: int, per_chunk: int = 300,
    use_mmr: bool = False, mmr_lambda: float = 0.85,
    fetch_k: int = 50, score_threshold: float | None = 0.25,
    rerank: bool = False, cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> Tuple[List[str], str]:
    cand = vs.similarity_search_with_relevance_scores(q, k=max(fetch_k, k))
    if score_threshold is not None:
        cand = [(d, s) for (d, s) in cand if (s is not None and float(s) >= score_threshold)]
    if not cand:
        return [], ""
    docs_sorted = [d for (d, _) in sorted(cand, key=lambda x: x[1], reverse=True)]

    if rerank and CE_AVAILABLE and len(docs_sorted) > k:
        try:
            ce = CrossEncoder(cross_encoder_name)
            pairs = [(q, d.page_content) for d in docs_sorted]
            scores = ce.predict(pairs)
            ranked = sorted(zip(docs_sorted, scores), key=lambda x: x[1], reverse=True)
            docs_final = [d for d, _ in ranked[:k]]
        except Exception:
            docs_final = docs_sorted[:k]
    elif use_mmr and len(docs_sorted) > k:
        uniq, seen = [], set()
        for d in docs_sorted:
            key = (d.page_content[:120].strip(), d.metadata.get("source", ""))
            if key in seen: continue
            seen.add(key); uniq.append(d)
            if len(uniq) >= k: break
        docs_final = uniq
    else:
        docs_final = docs_sorted[:k]

    chunks = [d.page_content[:per_chunk] for d in docs_final]
    joined = "\n\n".join(chunks)[:trim]
    return chunks, joined

def _print_debug_sample(q: str, ctx_chunks: List[str], answer: str, topn: int = 3, each_len: int = 280):
    lines = []
    lines.append("\n" + "="*72)
    lines.append("[DEBUG] Question:")
    lines.append(q.strip())
    lines.append("\n[DEBUG] Top contexts:")
    for i, c in enumerate(ctx_chunks[:topn], 1):
        clip = (c or "")[:each_len].rstrip()
        lines.append(f"  [{i}] {clip}")
    lines.append("\n[DEBUG] Answer:")
    lines.append((answer or "").strip())
    lines.append("="*72 + "\n")
    out = "\n".join(lines)
    print(out)
    try:
        os.makedirs("data/eval", exist_ok=True)
        with open("data/eval/last_debug.txt", "w", encoding="utf-8") as f:
            f.write(out)
    except Exception:
        pass

# ---- shrink contexts for LLM metrics (reduce timeouts) ----
def _shrink_contexts(ctx_lists: List[List[str]], topn: int = 3, each_len: int = 200) -> List[List[str]]:
    out = []
    for lst in ctx_lists:
        small = [(c or "")[:each_len] for c in (lst or [])][:topn]
        out.append(small)
    return out

# --- robust per-row export for Context_precision / Context_recall ---
def _export_per_row_ragas(result, questions, out_csv="experiments/ragas_row_scores.csv"):
    import pandas as pd, os, numpy as np
    try:
        df = result.to_pandas()
    except Exception:
        if hasattr(result, "results") and isinstance(result.results, list):
            df = pd.DataFrame(result.results)
        else:
            print("[warn] ragas per-row export: cannot obtain dataframe"); return

    def pick_col(df, must_have, fuzzy_keys):
        cols = list(df.columns)
        lower_map = {c.lower(): c for c in cols}
        for name in must_have:
            if name.lower() in lower_map:
                return lower_map[name.lower()]
        for c in cols:
            lc = c.lower()
            if all(k in lc for k in fuzzy_keys):
                return c
        return None

    cprec_col = pick_col(
        df,
        must_have=[
            "context_precision", "context_precision_with_reference",
            "nonllm_context_precision_with_reference",
            "non_llm_context_precision_with_reference",
        ],
        fuzzy_keys=["context","precision"],
    )
    crecall_col = pick_col(
        df,
        must_have=[
            "context_recall", "context_recall_with_reference",
            "nonllm_context_recall", "non_llm_context_recall",
        ],
        fuzzy_keys=["context","recall"],
    )

    out = pd.DataFrame({"question": questions})
    out["Context_precision"] = pd.to_numeric(df[cprec_col], errors="coerce") if cprec_col else np.nan
    out["Context_recall"]    = pd.to_numeric(df[crecall_col], errors="coerce") if crecall_col else np.nan

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    # header detection must be robust to empty/broken files
    header_needed = True
    if os.path.exists(out_csv):
        try:
            existing_cols = list(pd.read_csv(out_csv, nrows=0).columns)
            header_needed = (existing_cols != list(out.columns))
        except Exception:
            header_needed = True
    out.to_csv(out_csv, mode="a", index=False, header=header_needed)
    print(f"[Saved] per-row RAGAS -> {out_csv} (prec_col={cprec_col}, rec_col={crecall_col})")

# ---------------- Main ----------------
def main():
    p = argparse.ArgumentParser()

    # Core evaluation knobs
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--trim_ctx", type=int, default=DEFAULT_TRIM_CTX)
    p.add_argument("--full", type=str2bool, nargs="?", const=True, default=DEFAULT_FULL)

    # Retrieval & filtering knobs
    p.add_argument("--per_chunk", type=int, default=DEFAULT_PER_CHUNK)
    p.add_argument("--thr", type=float, default=DEFAULT_THR)
    p.add_argument("--fetch_k", type=int, default=DEFAULT_FETCH_K)
    p.add_argument("--mmr", action="store_true", default=False)
    p.add_argument("--mmr_lambda", type=float, default=0.85)
    p.add_argument("--rerank", type=str2bool, nargs="?", const=True, default=DEFAULT_RERANK)
    p.add_argument("--ce_model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    p.add_argument("--show_debug", action="store_true", default=DEFAULT_SHOW_DEBUG)

    # Single-sample controls
    p.add_argument("--row_idx", type=int, default=None)
    p.add_argument("--question_substr", type=str, default=None)
    p.add_argument("--no_cache", action="store_true", default=DEFAULT_NO_CACHE)

    args = p.parse_args()

    # Load testset
    df_all = pd.read_csv("data/eval/testset.csv")
    assert {"question", "ground_truth"}.issubset(df_all.columns)

    # --- pick single row if requested ---
    if args.row_idx is not None:
        if not (0 <= args.row_idx < len(df_all)):
            raise IndexError(f"--row_idx out of range (0..{len(df_all)-1})")
        df = df_all.iloc[[args.row_idx]].copy()
        print(f"[single] use row_idx={args.row_idx} → Q: {df.iloc[0]['question']}")
    elif args.question_substr:
        m = df_all[df_all["question"].astype(str).str.contains(args.question_substr, case=False, na=False)]
        if m.empty:
            raise ValueError(f"--question_substr not matched: {args.question_substr}")
        df = m.iloc[[0]].copy()
        print(f"[single] use question_substr='{args.question_substr}' → Q: {df.iloc[0]['question']}")
    else:
        df = df_all.head(args.limit).copy()

    # Retrieval embeddings (LangChain)
    lc_embeddings = LCHFEmbeddings(model_name=EMBEDDING_MODEL)
    vs = Chroma(
        collection_name="qld_rr_docs",
        embedding_function=lc_embeddings,
        persist_directory=CHROMA_DIR,
    )

    # RAGAS-side embeddings (no OpenAI)
    ragas_emb = RagasHFEmbeddings(model=EMBEDDING_MODEL)

    # Answering LLM
    llm = ChatOllama(
        model=EVAL_MODEL,
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",
        temperature=0.0,
        num_predict=320,
    )
    ragas_emb_eval = LangchainEmbeddingsWrapper(LcHFEmb(model_name=EMBEDDING_MODEL))

    # Cache answers
    os.makedirs("data/eval", exist_ok=True)
    cache_csv = "data/eval/run_answers.csv"
    cache = {}
    if os.path.exists(cache_csv) and not args.no_cache:
        try:
            if os.path.getsize(cache_csv) == 0:
                raise EmptyDataError("empty run_answers.csv")
            tmp = pd.read_csv(cache_csv, engine="python", on_bad_lines="skip")
        except EmptyDataError:
            tmp = pd.DataFrame(columns=["question","answer","context","ckey"])
        except Exception:
            tmp = pd.DataFrame(columns=["question","answer","context","ckey"])

        cols = list(tmp.columns)
        if len(cols) >= 3:
            tmp = tmp.iloc[:, :min(4, len(cols))].copy()
            if len(tmp.columns) == 3:
                tmp.columns = ["question","answer","context"]
                tmp["ckey"] = ""
            else:
                tmp.columns = ["question","answer","context","ckey"]
        else:
            tmp = pd.DataFrame(columns=["question","answer","context","ckey"])

        for _, r in tmp.iterrows():
            q = str(r.get("question","")); a = str(r.get("answer",""))
            c = str(r.get("context",""));  k = str(r.get("ckey",""))
            cache[q] = (a, c, k)

    rows_out = []
    questions, answers, contexts, gts = [], [], [], []
    ref_ctx_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="[answer]"):
        q = str(row["question"])
        ctx_chunks, ctx_joined = retrieve_context(
            vs, q, k=args.k, trim=args.trim_ctx,
            per_chunk=args.per_chunk,
            use_mmr=args.mmr, mmr_lambda=args.mmr_lambda, fetch_k=args.fetch_k,
            score_threshold=args.thr,
            rerank=args.rerank, cross_encoder_name=args.ce_model,
        )

        ckey = _cache_key(q, args)
        cached = cache.get(q)  # (answer, context, cached_ckey)
        use_cache = (
            (not args.no_cache)
            and cached and cached[0]
            and (not _bad_answer(cached[0]))
            and (cached[2] == "" or cached[2] == ckey)
        )

        if use_cache:
            a = cached[0]
        else:
            a = answer_question(llm, ctx_joined, q)
            rows_out.append({"question": q, "answer": a, "context": ctx_joined, "ckey": ckey})

        if args.show_debug:
            _print_debug_sample(q, ctx_chunks, a, topn=3, each_len=args.per_chunk)

        questions.append(q)
        answers.append(a)
        contexts.append(ctx_chunks)   # List[str]
        gts.append(str(row["ground_truth"]))

        if "ref_context" in df.columns:
            raw = row.get("ref_context", None)
            rc = "" if pd.isna(raw) else str(raw)
            rc = rc.strip()
            ref_ctx_rows.append([rc] if rc else list(ctx_chunks))
        else:
            ref_ctx_rows.append(list(ctx_chunks))

    if rows_out:
        df_out = pd.DataFrame(rows_out, columns=["question","answer","context","ckey"])
        file_exists = os.path.exists(cache_csv)

        need_migrate = False
        if file_exists:
            try:
                with open(cache_csv, "r", encoding="utf-8") as f:
                    header_line = f.readline().strip()
                need_migrate = ("ckey" not in [h.strip() for h in header_line.split(",")])
            except Exception:
                need_migrate = False

        if file_exists and need_migrate:
            try:
                tmp = pd.read_csv(cache_csv, engine="python", on_bad_lines="skip")
            except Exception:
                tmp = pd.DataFrame(columns=["question","answer","context","ckey"])
            cols = list(tmp.columns)
            if len(cols) >= 3:
                tmp = tmp.iloc[:, :min(4, len(cols))].copy()
                if len(tmp.columns) == 3:
                    tmp.columns = ["question","answer","context"]; tmp["ckey"] = ""
                else:
                    tmp.columns = ["question","answer","context","ckey"]
            else:
                tmp = pd.DataFrame(columns=["question","answer","context","ckey"])
            all_df = pd.concat([tmp, df_out], ignore_index=True)
            all_df.to_csv(cache_csv, index=False, quoting=csv.QUOTE_ALL)
        else:
            df_out.to_csv(cache_csv, mode="a", index=False, header=not file_exists, quoting=csv.QUOTE_ALL)
        print(f"[cache] appended {len(rows_out)} rows -> {cache_csv}")

    # RAGAS schema（Non-LLM metrics）
    ds = Dataset.from_dict({
        "user_input": questions,
        "retrieved_contexts": contexts,      # List[List[str]]
        "reference_contexts": ref_ctx_rows,  # List[List[str]]
        "response": answers,
        "reference": gts,
    })

    # Conservative concurrency/timeout for ragas internals
    os.environ.setdefault("RAGAS_CONCURRENCY", "1")
    os.environ.setdefault("RAGAS_TIMEOUT", "1200")

    # Optional LLM metrics
    extra_llm_scores = {}
    if args.full and LangchainLLM is not None and faithfulness is not None and answer_relevancy is not None:
        ragas_llm = LangchainLLM(llm)
        ds_llm = Dataset.from_dict({
            "user_input": questions,
            "retrieved_contexts": _shrink_contexts(contexts, topn=3, each_len=320),
            "reference_contexts": _shrink_contexts(ref_ctx_rows, topn=3, each_len=320),
            "response": answers,
            "reference": gts,
        })
        s_faith = _eval_one_metric(ds_llm, faithfulness, ragas_llm, ragas_emb_eval,
                                   max_retries=3, base_wait=8) or {}
        s_rel   = _eval_one_metric(ds_llm, answer_relevancy, ragas_llm, ragas_emb_eval,
                                   max_retries=3, base_wait=8) or {}
        extra_llm_scores = {**s_faith, **s_rel}
        if args.show_debug:
            print("[DEBUG] LLM-metrics contexts (shrunken):")
            for i, lst in enumerate(_shrink_contexts(contexts, topn=3, each_len=320)[:1], 1):
                for j, c in enumerate(lst, 1):
                    print(f"  ({i}.{j}) {c}")

    # Non-LLM metrics（always）
    metrics = [NonLLMContextPrecisionWithReference(), NonLLMContextRecall()]
    result = evaluate(ds, metrics=metrics, embeddings=ragas_emb_eval)

    # --- write per-row Non-LLM metrics (Context_precision / Context_recall) ---
    _export_per_row_ragas(result, questions, out_csv="experiments/ragas_row_scores.csv")

    scores = {}
    try:
        df_scores = result.to_pandas()
        num = df_scores.select_dtypes(include=["number"])
        agg = num.mean(numeric_only=True) if len(df_scores) > 1 else num.iloc[0]
        scores = {k: float(v) for k, v in agg.items()}
    except Exception:
        if hasattr(result, "scores"):
            scores = {k: float(v) for k, v in result.scores.items()}

    # Merge -> sanitize -> normalize -> print
    scores = {**extra_llm_scores, **scores}
    scores = nan_safe_scores(scores)
    scores = normalize_ragas_scores(scores)

    # Persist (aggregated view)
    os.makedirs("experiments", exist_ok=True)
    csv_path = "experiments/ab_results.csv"
    is_new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ts","embed_model","retriever_k","limit","trim_ctx","eval_model",
                "faithfulness","answer_relevancy","context_precision","context_recall",
            ],
        )
        if is_new:
            writer.writeheader()
        writer.writerow({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "embed_model": EMBEDDING_MODEL,
            "retriever_k": args.k,
            "limit": len(df),
            "trim_ctx": args.trim_ctx,
            "eval_model": EVAL_MODEL,
            "faithfulness": f"{scores.get('faithfulness', 0.0):.4f}",
            "answer_relevancy": f"{scores.get('answer_relevancy', 0.0):.4f}",
            "context_precision": f"{scores.get('context_precision', 0.0):.4f}",
            "context_recall": f"{scores.get('context_recall', 0.0):.4f}",
        })
    print("[Saved] experiments/ab_results.csv updated.")

if __name__ == "__main__":
    main()
