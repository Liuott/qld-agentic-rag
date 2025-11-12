# serve_api.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.eval.eval_ragas import retrieve_context, answer_question
from langchain_huggingface import HuggingFaceEmbeddings as LCHFEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from src.config import CHROMA_DIR, EMBEDDING_MODEL, OLLAMA_BASE_URL, OLLAMA_MODEL

app = FastAPI()

# 允许从你的 WebUI 域/端口访问（偷懒可用 "*"）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://35.189.26.182:3000", "http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def init_services():
    # 运行时再初始化，避免导入时因为依赖/网络失败导致 uvicorn 直接退出
    app.state.emb = LCHFEmbeddings(model_name=EMBEDDING_MODEL)
    app.state.vs  = Chroma(
        collection_name="qld_rr_docs",
        embedding_function=app.state.emb,
        persist_directory=CHROMA_DIR
    )
    app.state.llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,  # 建议设为 http://127.0.0.1:11434
        api_key="ollama",
        temperature=0.0
    )

class AskIn(BaseModel):
    question: str
    k: int = 10
    trim_ctx: int = 900
    per_chunk: int = 300
    fetch_k: int = 80
    thr: float | None = 0.25
    rerank: bool = False
    mmr: bool = False

class AskOut(BaseModel):
    answer: str
    context: str
    chunks: list[str]

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/ask", response_model=AskOut)
def ask(payload: AskIn, request: Request):
    vs  = request.app.state.vs
    llm = request.app.state.llm
    chunks, joined = retrieve_context(
        vs, payload.question,
        k=payload.k, trim=payload.trim_ctx,
        per_chunk=payload.per_chunk,
        use_mmr=payload.mmr, fetch_k=payload.fetch_k,
        score_threshold=payload.thr, rerank=payload.rerank
    )
    ans = answer_question(llm, joined, payload.question)
    return AskOut(answer=ans, context=joined, chunks=chunks)
