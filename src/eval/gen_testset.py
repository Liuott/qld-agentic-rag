"""Generate a RAGAS test set (question / ground truth / reference context).

Fixes vs. the original
----------------------
* paths come from :mod:`src.config` (no more ``data/eval/testset.csv`` relative
  to whatever directory you happened to launch from);
* the vector store and the model are obtained from the shared factories, so the
  embedding model is not re-loaded twice;
* a row produced by the "plain + parser" fallback now consistently keeps
  ``ref_context`` (the old code dropped it, which silently degraded
  context-recall for those rows).

    python -m src.eval.gen_testset --n 30
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from tqdm import tqdm

from src.config import get_settings
from src.rag import get_llm, get_vectorstore


class QA(BaseModel):
    question: str = Field(..., description="User question")
    ground_truth: str = Field(..., description="Short factual answer from snippet")


parser = JsonOutputParser(pydantic_object=QA)

PROMPT = ChatPromptTemplate.from_template(
    """You must return ONLY a JSON object with keys "question" and "ground_truth".
Use ONLY the snippet to create ONE concise QA pair.
{format_instructions}

Snippet:
{snippet}
"""
)

SNIPPET_TRIM = 500
MAX_TOKENS = 128


def coerce_json_to_pair(text: str) -> dict[str, str] | None:
    """Best-effort extraction of a QA pair from a messy model response."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    candidate = cleaned[start : end + 1]
    for loader in (json.loads, ast.literal_eval):
        try:
            obj = loader(candidate)
        except Exception:
            continue
        question = str(obj.get("question", "")).strip()
        answer = str(obj.get("ground_truth", "")).strip()
        if question and answer:
            return {"question": question, "ground_truth": answer}
    return None


def sample_snippets(k: int = 30) -> list[str]:
    """Draw ``k`` random chunk texts from the vector store."""
    documents = get_vectorstore().get().get("documents", [])
    if not documents:
        raise RuntimeError("Vector store is empty. Run `python -m src.ingest` first.")
    return random.sample(documents, min(k, len(documents)))


def generate_pairs(snippets: Sequence[str]) -> list[dict[str, str]]:
    model = get_llm(temperature=0.1, num_predict=MAX_TOKENS)
    model_json = get_llm(temperature=0.1, num_predict=MAX_TOKENS, json_mode=True)
    format_instructions = parser.get_format_instructions()

    rows: list[dict[str, str]] = []
    for index, snippet in enumerate(tqdm(snippets, desc="[gen]"), 1):
        trimmed = snippet[:SNIPPET_TRIM]
        payload: dict[str, Any] = {"snippet": trimmed, "format_instructions": format_instructions}

        # 1) JSON-mode call
        try:
            raw = (PROMPT | model_json).invoke(payload).content
            obj = json.loads(raw)
            question = str(obj.get("question", "")).strip()
            answer = str(obj.get("ground_truth", "")).strip()
            if question and answer:
                rows.append({"question": question, "ground_truth": answer, "ref_context": trimmed})
                continue
        except Exception:
            pass

        # 2) structured output parser
        try:
            parsed = (PROMPT | model | parser).invoke(payload)
            question = str(parsed["question"]).strip()
            answer = str(parsed["ground_truth"]).strip()
            if question and answer:
                rows.append({"question": question, "ground_truth": answer, "ref_context": trimmed})
                continue
        except Exception:
            pass

        # 3) last resort: regex/literal-eval extraction
        try:
            raw = (PROMPT | model).invoke(payload).content
            pair = coerce_json_to_pair(raw)
            if pair:
                pair["ref_context"] = trimmed
                rows.append(pair)
            elif index <= 3:
                print("[debug raw]", raw[:300].replace("\n", " "))
        except Exception:
            pass
    return rows


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description="Generate a RAGAS test set from the corpus.")
    cli.add_argument("--n", type=int, default=30, help="number of snippets to sample")
    cli.add_argument("--out", type=Path, default=None, help="output CSV path")
    args = cli.parse_args(argv)

    out = args.out or (get_settings().eval_dir / "testset.csv")
    snippets = sample_snippets(args.n)
    rows = generate_pairs(snippets)
    if not rows:
        raise SystemExit(
            "No QA pairs generated. Try a smaller --n or a stronger OLLAMA_MODEL "
            "(the snippet may be too long for a 1.5B model)."
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[gen] Saved {len(rows)} rows to {out} (ok={len(rows)}/{len(snippets)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
