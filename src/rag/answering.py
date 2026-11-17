"""Answer generation: prompt -> LLM -> plain text.

``answer_question`` keeps the original call signature ``(llm, context, question)``
so the existing evaluation scripts keep working unchanged.
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
    """Return a cached ``ChatOllama`` client (one per parameter combination).

    Building the client re-reads a config and constructs a pydantic model, which
    is cheap but not free; more importantly, one client per configuration keeps
    the HTTP session pool warm across questions.
    """
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


def answer_question(
    llm: BaseLanguageModel,
    ctx_joined: str,
    question: str,
    prompt: BasePromptTemplate | None = None,
) -> str:
    """Answer ``question`` from ``ctx_joined`` using ``llm``.

    The LCEL chain is composed per call on purpose: composition is pure object
    wiring (microseconds, dwarfed by the model call), whereas caching it would
    require the prompt and LLM objects to be hashable -- pydantic models in the
    LangChain stack are not guaranteed to be, and an unhashable argument would
    raise ``TypeError`` at call time.
    """
    chain = (prompt or RAG_ANSWER_PROMPT) | llm | StrOutputParser()
    return chain.invoke({"context": ctx_joined, "question": question})


__all__ = ["answer_question", "get_llm"]
