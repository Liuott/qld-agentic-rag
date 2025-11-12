import json, random, re, ast
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from src.config import CHROMA_DIR, EMBEDDING_MODEL, OLLAMA_BASE_URL, OLLAMA_MODEL

# ---------- Structured output schema ----------
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

def coerce_json_to_pair(text: str) -> Optional[dict]:
    txt = text.strip()
    txt = re.sub(r"^```(?:json)?|```$", "", txt, flags=re.MULTILINE).strip()
    i, j = txt.find("{"), txt.rfind("}")
    if i >= 0 and j > i:
        chunk = txt[i:j+1]
        for loader in (json.loads, ast.literal_eval):
            try:
                obj = loader(chunk)
                q = str(obj.get("question","")).strip()
                a = str(obj.get("ground_truth","")).strip()
                if q and a:
                    return {"question": q, "ground_truth": a}
            except Exception:
                pass
    return None

OUT_CSV = Path("data/eval/testset.csv")

def sample_snippets(k=30) -> List[str]:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vs = Chroma(collection_name="qld_rr_docs", embedding_function=embeddings, persist_directory=CHROMA_DIR)
    raw = vs.get()
    docs = raw.get("documents", [])
    if not docs:
        raise RuntimeError("Vector store empty. Run ingest.py first.")
    k = min(k, len(docs))
    return random.sample(docs, k)

TRIM = 500    
MAX_TOK = 128  

def generate_pairs(snippets: List[str]):

    llm_json = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",
        temperature=0.1,
        num_predict=MAX_TOK,
        format="json",         
    )
    llm_plain = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",
        temperature=0.1,
        num_predict=MAX_TOK,
    )

    rows = []
    total = len(snippets)
    fmt = {"format_instructions": parser.get_format_instructions()}

    for i, s in enumerate(tqdm(snippets, desc="[gen]"), 1):
        s = s[:TRIM]

        try:
            raw = (PROMPT | llm_json).invoke({"snippet": s, **fmt}).content
            obj = json.loads(raw)
            q = obj.get("question","").strip()
            a = obj.get("ground_truth","").strip()
            if q and a:
                rows.append({"question": q, "ground_truth": a, "ref_context": s})
                continue
        except Exception:
            pass

        try:
            out = (PROMPT | llm_plain | parser).invoke({"snippet": s, **fmt})
            rows.append({"question": out["question"].strip(), "ground_truth": out["ground_truth"].strip()})
            continue
        except Exception:
            pass

        try:
            raw2 = (PROMPT | llm_plain).invoke({"snippet": s, **fmt}).content
            pair = coerce_json_to_pair(raw2)
            if pair:
                rows.append(pair)
            else:

                if i <= 3:
                    print("[debug raw]", raw2[:300].replace("\n", " "))
        except Exception:
            pass

    return rows

def main(n: int = 30):
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    snippets = sample_snippets(n)
    rows = generate_pairs(snippets)
    if not rows:
        raise RuntimeError("No pairs generated. Try smaller n, smaller TRIM, and ensure OLLAMA_MODEL is qwen2.5:1.5b.")
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"[gen] Saved {len(rows)} rows to {OUT_CSV} (ok={len(rows)}/{len(snippets)})")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30)
    args = p.parse_args()
    main(args.n)
