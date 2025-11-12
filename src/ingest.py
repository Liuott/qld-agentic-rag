import os
from pathlib import Path
from typing import List


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader


from src.config import CHROMA_DIR, EMBEDDING_MODEL


DATA_DIR = Path("data/sources")




def load_sources() -> List:
    docs = []
    for p in DATA_DIR.glob("*.pdf"):
        loader = PyPDFLoader(str(p))
        docs.extend(loader.load())
    for p in DATA_DIR.glob("*.md"):
        loader = TextLoader(str(p), encoding="utf-8")
        docs.extend(loader.load())
    return docs




def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)


    print("[Ingest] Loading sources from:", DATA_DIR)
    raw_docs = load_sources()
    assert raw_docs, "No source files found in data/sources/."


    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    splits = splitter.split_documents(raw_docs)
    print(f"[Ingest] Split into {len(splits)} chunks")


    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


    vectorstore = Chroma(
        collection_name="qld_rr_docs",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


    BATCH = 64
    total = len(splits)
    for i in range(0, total, BATCH):
        batch = splits[i:i+BATCH]
        vectorstore.add_documents(batch)
        print(f"[Ingest] Embedded {min(i+BATCH, total)}/{total}", flush=True)
    print(f"[Ingest] Done. Persist at {CHROMA_DIR}")

    #ids = vectorstore.add_documents(splits)
    #print(f"[Ingest] Added {len(ids)} chunks. Persist at {CHROMA_DIR}")


if __name__ == "__main__":
    main()