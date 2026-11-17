"""Ask a single question from the command line (debugging aid).

Before: the module imported ``retrieve_context`` / ``answer_question`` from
``src.eval.eval_ragas``, which pulled in ``ragas``, ``datasets`` and ``pandas``
-- several seconds of imports for a one-question CLI.  It now uses
:mod:`src.rag` directly.

    python -m src.eval.ask_one "What should I do when crossing the road?"
    python -m src.eval.ask_one                 # interactive REPL
"""

from __future__ import annotations

import argparse

from src.config import get_settings
from src.rag import answer_question, get_llm, get_vectorstore, retrieve_context


def ask(
    question: str,
    *,
    k: int = 10,
    trim: int = 900,
    per_chunk: int = 300,
    fetch_k: int = 80,
    score_threshold: float = 0.25,
    rerank: bool = True,
    use_mmr: bool = False,
) -> tuple[str, list[str], str]:
    """Return ``(answer, chunks, joined_context)`` for one question."""
    chunks, joined = retrieve_context(
        get_vectorstore(),
        question,
        k=k,
        trim=trim,
        per_chunk=per_chunk,
        fetch_k=fetch_k,
        score_threshold=score_threshold,
        rerank=rerank,
        use_mmr=use_mmr,
    )
    answer = answer_question(get_llm(temperature=0.0), joined, question)
    return answer, chunks, joined


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask one question against the local corpus.")
    parser.add_argument("question", nargs="?", help="Question to ask; omit for a REPL")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--trim", type=int, default=2000)
    parser.add_argument("--per-chunk", type=int, dest="per_chunk", default=400)
    parser.add_argument("--fetch-k", type=int, dest="fetch_k", default=80)
    parser.add_argument("--thr", type=float, dest="thr", default=0.25)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--mmr", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    options = {
        "k": args.k,
        "trim": args.trim,
        "per_chunk": args.per_chunk,
        "fetch_k": args.fetch_k,
        "score_threshold": args.thr,
        "rerank": not args.no_rerank,
        "use_mmr": args.mmr,
    }

    if not args.question:
        print(f"Enter question (model={settings.ollama_model}, Ctrl+C to exit):")
        try:
            while True:
                question = input("> ").strip()
                if not question:
                    continue
                answer, _, _ = ask(question, **options)
                print("\nANSWER:\n", answer, "\n")
        except (EOFError, KeyboardInterrupt):
            return 0
        return 0

    answer, chunks, joined = ask(args.question, **options)
    print("ANSWER:\n", answer)
    print("\n--- CONTEXT ---\n", joined)
    print("\n--- CHUNKS ---")
    for index, chunk in enumerate(chunks, 1):
        print(f"[{index}] {chunk[:220].replace(chr(10), ' ')}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
