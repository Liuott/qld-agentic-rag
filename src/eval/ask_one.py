# tools/ask_once.py
from __future__ import annotations
import argparse
from langchain_huggingface import HuggingFaceEmbeddings as LCHFEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

from src.eval.eval_ragas import retrieve_context, answer_question
from src.config import CHROMA_DIR, EMBEDDING_MODEL, OLLAMA_BASE_URL, OLLAMA_MODEL

def bootstrap():
    emb = LCHFEmbeddings(model_name=EMBEDDING_MODEL)
    vs  = Chroma(collection_name="qld_rr_docs",
                 embedding_function=emb,
                 persist_directory=CHROMA_DIR)
    llm = ChatOllama(model=OLLAMA_MODEL,
                     base_url=OLLAMA_BASE_URL,
                     api_key="ollama",
                     temperature=0.0)
    return vs, llm

def ask(q: str, *,
        k=10, trim=900, per_chunk=300, fetch_k=80,
        score_threshold=0.25, rerank=True, use_mmr=False):
    vs, llm = bootstrap()
    chunks, joined = retrieve_context(
        vs, q,
        k=k, trim=trim, per_chunk=per_chunk,
        fetch_k=fetch_k, score_threshold=score_threshold,
        rerank=rerank, use_mmr=use_mmr
    )
    ans = answer_question(llm, joined, q)
    return ans, chunks, joined

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", help="Question to ask")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--trim", type=int, default=2000)
    ap.add_argument("--per-chunk", type=int, dest="per_chunk", default=400)
    ap.add_argument("--fetch-k", type=int, dest="fetch_k", default=80)
    ap.add_argument("--thr", type=float, dest="thr", default=0.25)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--mmr", action="store_true")
    args = ap.parse_args()

    if not args.question:
        # 简单 REPL：一行一个问题
        print("Enter question (Ctrl+C to exit):")
        try:
            while True:
                q = input("> ").strip()
                if not q:
                    continue
                ans, _, _ = ask(q,
                                k=args.k, trim=args.trim, per_chunk=args.per_chunk,
                                fetch_k=args.fetch_k, score_threshold=args.thr,
                                rerank=not args.no_rerank, use_mmr=args.mmr)
                print("\nANSWER:\n", ans, "\n")
        except (EOFError, KeyboardInterrupt):
            return
    else:
        ans, chunks, joined = ask(args.question,
                                  k=args.k, trim=args.trim, per_chunk=args.per_chunk,
                                  fetch_k=args.fetch_k, score_threshold=args.thr,
                                  rerank=not args.no_rerank, use_mmr=args.mmr)
        print("ANSWER:\n", ans)
        print("\n--- CONTEXT ---\n", joined)
        print("\n--- CHUNKS ---")
        for i, c in enumerate(chunks, 1):
            print("[{}] {}...".format(i, c[:220].replace("\n", " ")))


if __name__ == "__main__":
    main()
