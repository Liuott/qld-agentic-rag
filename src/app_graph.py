from typing import List, Literal, Sequence


from typing_extensions import Annotated, TypedDict


from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools.retriever import create_retriever_tool


from src.config import CHROMA_DIR, EMBEDDING_MODEL, OLLAMA_BASE_URL, OLLAMA_MODEL
from src.prompt import RAG_FALLBACK


# ===== Vector store / retriever =====
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(
    collection_name="qld_rr_docs",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_qld_rr",
    "Search Queensland Road Rules (2009) & Driving handout text clauses.",
)


# ===== LLM =====
llm_plain = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, api_key="ollama", temperature=0.2)


# Try to pull a standard RAG prompt; if unavailable, use local fallback
try:
    from langchain import hub
    RAG_PROMPT = hub.pull("rlm/rag-prompt")
except Exception:
    RAG_PROMPT = RAG_FALLBACK


# ===== Agent State =====
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], ...] # simple list, we don't need add_messages helper here




# ===== Nodes =====
def agent(state: AgentState):
    messages = state["messages"]
    model = llm_plain.bind_tools([retriever_tool])
    response = model.invoke(messages)
    return {"messages": [response]}




def evaluate_documents(state: AgentState) -> Literal["generate", "rewrite"]:
    # Simple heuristic: if last message is a tool-call result (documents), go generate; otherwise rewrite
    msgs = state["messages"]
    last = msgs[-1]
    content = getattr(last, "content", "")
    # If the retriever tool already returned text (docs), go generate; very lightweight gate
    return "generate" if (isinstance(content, str) and len(content) > 50) else "rewrite"




def rewrite(state: AgentState):
    q = state["messages"][0].content
    prompt = ChatPromptTemplate.from_template(
        "Rewrite the question to be clearer and answerable with QLD road rules.\nOriginal: {q}"
    )
    msg = prompt.invoke({"q": q})
    response = llm_plain.invoke([msg])
    return {"messages": [response]}




def generate(state: AgentState):
    msgs = state["messages"]
    question = msgs[0].content
    docs = msgs[-1].content # retriever tool output (stringified docs)


    chain = RAG_PROMPT | llm_plain | StrOutputParser()
    answer = chain.invoke({"context": docs, "question": question})
    return {"messages": [answer]}




workflow = StateGraph(AgentState)
workflow.add_node("agent", agent)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node("rewrite", rewrite)
workflow.add_node("generate", generate)


workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition, {"tools": "retrieve", END: END})
workflow.add_conditional_edges("retrieve", evaluate_documents)
workflow.add_edge("rewrite", "agent")
workflow.add_edge("generate", END)


graph = workflow.compile()




# ===== CLI =====
if __name__ == "__main__":
    print("[Agentic RAG] Ready. Type your question (or 'exit').")
    while True:
        q = input("Q> ").strip()
        if not q or q.lower() in {"exit", "quit"}: break
        inputs = {"messages": [HumanMessage(content=q)]}
        result = graph.invoke(inputs)
        final_msg = result.get("messages", [None])[-1]
        print("A>", getattr(final_msg, "content", str(final_msg)))