"""Answer generation: prompt -> LLM -> plain text.

``answer_question`` keeps the original call signature ``(llm, context, question)``
so the existing evaluation scripts keep working unchanged, but the chain is now
built once per (prompt, llm) pair instead of on every call.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_ollama import ChatOllama

from src.config import get_settings
from src.prompt import RAG_ANSWER_PROMPT


@lru_cache(maxsize=8)
def get_llm(
    model: str | None = None,
    temperature: float = 0.0,
    num_predict: int | None = None,
    json_mode: bool = False,
    seed: int | None = None,
) -> ChatOllama:
    """Return a cached ``ChatOllama`` client (one per parameter combination)."""
    settings = get_settings()
    kwargs: dict = {
        "model": model or settings.ollama_model,
        "base_url": settings.ollama_base_url,
        "api_key": "ollama",
        "temperature": temperature,
    }
    if num_predict is not None:
        kwargs["num_predict"] = num_predict
    if json_mode:
        kwargs["format"] = "json"
    if seed is not None:
        kwargs["seed"] = seed
    return ChatOllama(**kwargs)


@lru_cache(maxsize=8)
def _build_chain(prompt: BasePromptTemplate, llm: BaseLanguageModel):
    return prompt | llm | StrOutputParser()


def answer_question(
    llm: BaseLanguageModel,
    ctx_joined: str,
    question: str,
    prompt: BasePromptTemplate | None = None,
) -> str:
    """Answer ``question`` from ``ctx_joined`` using ``llm``."""
    chain = _build_chain(prompt or RAG_ANSWER_PROMPT, llm)
    return chain.invoke({"context": ctx_joined, "question": question})


__all__ = ["answer_question", "get_llm"]
