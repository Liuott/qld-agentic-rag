"""Agentic RAG graph: ``agent -> retrieve -> grade -> (generate | rewrite)``.

What changed vs. the original implementation
--------------------------------------------
1. **No work at import time.**  The vector store and the embedding model used to
   be created as module-level side effects, so merely importing this module
   downloaded models and opened Chroma.
2. **The answer is generated from the documents that were actually retrieved.**
   The old ``generate`` node used ``state["messages"][-1].content`` as the
   context; after a ``rewrite`` step that message was the rewritten *question*,
   so the model was told to answer a question using the question as context.
3. **The loop is bounded.**  ``rewrite -> agent`` had no step counter, so a
   model that kept calling the retriever could loop forever.
4. **The retrieval gate checks the retrieval result**, not
   ``len(content) > 50`` -- an empty answer or a short document no longer flips
   the branch unpredictably.
5. **Documents keep their metadata**, which is what makes real citations
   (``file.pdf p.12``) possible.

Try it::

    python -m src.app_graph                       # interactive
    python -m src.app_graph --question "..."      # one shot
"""

from __future__ import annotations

import argparse
import operator
import sys
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.config import get_settings
from src.prompt import RAG_FALLBACK
from src.rag import get_llm, get_vectorstore, retrieve_documents, vectorstore_is_empty

TOOL_NAME = "retrieve_qld_rr"
TOOL_DESCRIPTION = (
    "Search the Queensland Road Rules (2009) act text and the Queensland driving "
    "handbook for clauses about a specific driving situation. "
    "Input: a short, self-contained search query."
)


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
class AgentState(TypedDict):
    """Graph state.

    ``messages`` uses the ``add_messages`` reducer (append + id de-duplication),
    ``steps`` uses integer addition, everything else is overwritten.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    documents: list[Document]
    steps: Annotated[int, operator.add]
    answer: str


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def format_documents(documents: Sequence[Document], per_chunk: int = 1200) -> str:
    """Render retrieved documents as numbered, source-tagged context."""
    if not documents:
        return ""
    blocks: list[str] = []
    for index, doc in enumerate(documents, 1):
        metadata = doc.metadata or {}
        source = Path(str(metadata.get("source", "unknown"))).name
        page = metadata.get("page")
        location = f"{source} p.{page}" if isinstance(page, int) else source
        blocks.append(f"[{index}] ({location})\n{(doc.page_content or '').strip()[:per_chunk]}")
    return "\n\n".join(blocks)


@lru_cache(maxsize=1)
def build_retriever_tool() -> StructuredTool:
    """Tool exposed to the model so it can *ask* for a retrieval."""

    def _search(query: str) -> str:
        documents = retrieve_documents(get_vectorstore(), query)
        return format_documents(documents) or "No document matched this query."

    return StructuredTool.from_function(
        func=_search,
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
    )


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def agent(state: AgentState) -> dict[str, Any]:
    """Decide whether to retrieve more evidence or answer right away."""
    settings = get_settings()
    llm = get_llm(temperature=0.2)
    has_evidence = bool(state.get("documents"))
    exhausted = state.get("steps", 0) >= settings.max_agent_steps

    model = llm if (exhausted and has_evidence) else llm.bind_tools([build_retriever_tool()])
    response = model.invoke(state["messages"])

    if exhausted and has_evidence and getattr(response, "tool_calls", None):
        # Out of budget: keep the text, drop the tool call so we fall through to generate.
        response = AIMessage(content=response.content or "", id=getattr(response, "id", None))
    return {"messages": [response]}


def retrieve(state: AgentState) -> dict[str, Any]:
    """Run the retrieval for the pending tool call(s) and store the documents."""
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    settings = get_settings()

    documents: list[Document] = []
    tool_messages: list[BaseMessage] = []
    for call in tool_calls:
        query = str((call.get("args") or {}).get("query", "")).strip() or state["question"]
        found = retrieve_documents(get_vectorstore(), query, k=settings.retriever_k)
        documents.extend(found)
        tool_messages.append(
            ToolMessage(
                content=format_documents(found) or "No document matched this query.",
                tool_call_id=call.get("id", ""),
                name=TOOL_NAME,
            )
        )

    # Keep the best `k` unique documents across several tool calls.
    if len(documents) > settings.retriever_k:
        documents = documents[: settings.retriever_k]

    return {"messages": tool_messages, "documents": documents, "steps": 1}


def grade(state: AgentState) -> Literal["generate", "rewrite", "give_up"]:
    """Route on the retrieval outcome, not on string length."""
    settings = get_settings()
    if state.get("documents"):
        return "generate"
    if state.get("steps", 0) < settings.max_agent_steps:
        return "rewrite"
    return "give_up"


def rewrite(state: AgentState) -> dict[str, Any]:
    """Ask the model for a retrieval-friendly reformulation of the question."""
    settings = get_settings()
    if state.get("steps", 0) >= settings.max_agent_steps:
        return {"messages": [], "steps": 1}

    llm = get_llm(temperature=0.0)
    prompt = (
        "Rewrite the question so it is easier to match against Queensland Road "
        "Rules clauses. Use concrete vocabulary (give way, roundabout, park, "
        "speed limit, learner, seatbelt ...).\n"
        f"Original question: {state['question']}\n"
        "Return only the rewritten question."
    )
    rewritten = _rewrite_chain(llm).invoke({"prompt": prompt}).strip() or state["question"]
    return {
        "messages": [HumanMessage(content=f"Use this refined search query: {rewritten}")],
        "steps": 1,
    }


def generate(state: AgentState) -> dict[str, Any]:
    """Produce the final answer from the documents held in the state."""
    context = format_documents(state.get("documents") or [])
    question = state["question"]

    if not context:
        message = (
            "I don't know. No clause in the indexed Queensland Road Rules corpus "
            "matched this question -- please add a more specific scenario."
        )
        return {"messages": [AIMessage(content=message)], "answer": message}

    llm = get_llm(temperature=0.2)
    chain = RAG_FALLBACK | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question}).strip()
    return {"messages": [AIMessage(content=answer)], "answer": answer}


def give_up(state: AgentState) -> dict[str, Any]:
    """No evidence and no budget left: answer honestly instead of looping."""
    return generate(state)


@lru_cache(maxsize=2)
def _rewrite_chain(llm: Any):
    return ChatPromptTemplate.from_messages([("human", "{prompt}")]) | llm


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
def route_after_agent(state: AgentState) -> Literal["retrieve", "generate"]:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "retrieve"
    return "generate"


def build_graph(checkpointer: Any = None):
    """Compile the agent graph (cheap: nothing is loaded until it runs)."""
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("generate", generate)
    workflow.add_node("give_up", give_up)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", route_after_agent, {"retrieve": "retrieve", "generate": "generate"})
    workflow.add_conditional_edges(
        "retrieve", grade, {"generate": "generate", "rewrite": "rewrite", "give_up": "give_up"}
    )
    workflow.add_edge("rewrite", "agent")
    workflow.add_edge("generate", END)
    workflow.add_edge("give_up", END)
    return workflow.compile(checkpointer=checkpointer)


def ask(question: str, graph: Any | None = None) -> dict[str, Any]:
    """Answer one question and return ``{"answer", "documents"}``."""
    settings = get_settings()
    compiled = graph or build_graph()
    result = compiled.invoke(
        {"messages": [HumanMessage(content=question)], "question": question, "documents": [], "steps": 0},
        config={"recursion_limit": 4 * settings.max_agent_steps + 6},
    )
    return {"answer": result.get("answer", ""), "documents": result.get("documents", [])}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agentic RAG over the Queensland Road Rules.")
    parser.add_argument("--question", "-q", help="Answer a single question and exit.")
    parser.add_argument("--show-sources", action="store_true", help="Print the retrieved sources.")
    args = parser.parse_args(argv)

    if vectorstore_is_empty(get_vectorstore()):
        print(
            "The vector store is empty. Run `python -m src.ingest` first.",
            file=sys.stderr,
        )
        return 2

    if args.question:
        result = ask(args.question)
        print(result["answer"])
        if args.show_sources:
            for index, doc in enumerate(result["documents"], 1):
                source = Path(str(doc.metadata.get("source", "?"))).name
                print(f"  [{index}] {source} p.{doc.metadata.get('page', '?')}")
        return 0

    print("[Agentic RAG] Ready. Type your question (or 'exit').")
    graph = build_graph()
    while True:
        try:
            question = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in {"exit", "quit"}:
            break
        result = ask(question, graph)
        print("A>", result["answer"])
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
