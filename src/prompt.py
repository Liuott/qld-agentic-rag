"""Prompt templates used across the project.

Three flavours exist on purpose:

* :data:`RAG_ANSWER_PROMPT` -- the "Evidence / Answer" template.  Every
  evaluation script and the HTTP API use it, so scores stay comparable with the
  ``experiments/*.csv`` history.
* :data:`RAG_FALLBACK` -- a shorter, single-paragraph template for the
  interactive agent (fast on small local models).
* :data:`EVAL_TEMPLATE` / :data:`GEN_QA_TEMPLATE` -- grading / test-set
  generation helpers.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

#: Serving + evaluation template (kept byte-identical to the historical one so
#: cached answers in ``data/eval/run_answers.csv`` remain valid).
RAG_ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """
Answer strictly from the context. If something is not in the context, say you don't know.

Respond in two sections:
1) Evidence: quote the exact lines (verbatim) from the context that support your answer. Use bullet points and keep each quote short.
2) Answer: a concise synthesis. End with: "Source: <Act/MUTCD, clause #>" when the clause is explicitly present.

--------------------
{context}
--------------------
Question: {question}
"""
)

#: Compact template used by the interactive agent (``src.app_graph``).
RAG_FALLBACK = ChatPromptTemplate.from_template(
    """
    You are a careful assistant for Queensland road rules. Answer strictly based on the provided context.
    - Cite the specific clause number and source (Road Rules 2009 / Driving Handout text clauses) when possible.
    - If the answer cannot be supported by the context, say you don't know and ask for a more specific query.
    - You MUST include a source line at the end in the format: "Source: <Act/MUTCD, clause #>".
    - If no clause in the context supports the answer, reply: "I don't know. Please provide a more specific clause or scenario."

    --------------------
    Context:
    {context}
    --------------------
    Question: {question}
    """
)

EVAL_TEMPLATE = ChatPromptTemplate.from_template(
    """
    You are grading a QA for Queensland road rules.
    Role: strict examiner. Only judge based on the given context.


    Question: {question}
    Retrieved Context: {context}
    Candidate Answer: {answer}


    Please output a JSON with fields:
    - correctness: one of ["yes", "no"]
    - relevance: one of ["high", "medium", "low"]
    - faithfulness: one of ["yes", "no"]
    - rationale: short explanation in English
    """
)

GEN_QA_TEMPLATE = ChatPromptTemplate.from_template(
    """
    You are creating exam-style QA pairs for Queensland road rules.
    Use only the snippet below to propose ONE useful user question and its ground-truth short answer.


    Snippet:
    {snippet}


    Output JSON with fields: question, ground_truth.
    """
)

__all__ = ["EVAL_TEMPLATE", "GEN_QA_TEMPLATE", "RAG_ANSWER_PROMPT", "RAG_FALLBACK"]
