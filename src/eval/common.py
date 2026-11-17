"""Pure helpers shared by the evaluation scripts.

``eval_ragas.py`` and ``eval_newv.py`` were 95 % identical: the second one was a
re-formatted copy of the first with the self-evaluation loop added.  All the
duplicated helpers now live here, are free of side effects, and are unit tested
in ``tests/test_eval_common.py``.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterable, Sequence
from typing import Any

# --------------------------------------------------------------------------- #
# CLI / parsing
# --------------------------------------------------------------------------- #
_TRUE = {"1", "true", "yes", "y", "on"}
_FALSE = {"0", "false", "no", "n", "off"}


def str2bool(value: Any) -> bool:
    """``argparse`` type for boolean flags that accept an explicit value."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got: {value}")


# --------------------------------------------------------------------------- #
# Answer quality / caching
# --------------------------------------------------------------------------- #
def is_bad_answer(answer: str | None, min_length: int = 50) -> bool:
    """True when a cached answer should be regenerated.

    Keeps the historic rule: too short, or missing the mandatory
    ``Evidence`` / ``Answer`` sections.
    """
    text = (answer or "").strip()
    return len(text) < min_length or "Evidence" not in text or "Answer" not in text


def cache_key(question: str, args: argparse.Namespace, eval_model: str) -> str:
    """Stable cache key for one (question, retrieval settings) pair.

    The first eight fields are byte-identical to the historical key, so answers
    cached by older runs keep hitting.  ``loop``/``judge`` are appended only for
    loop runs, which never existed before.
    """
    fields = [
        question.strip(),
        f"eval_model={eval_model}",
        f"k={getattr(args, 'k', '')}",
        f"trim={getattr(args, 'trim_ctx', '')}",
        f"chunk={getattr(args, 'per_chunk', '')}",
        f"thr={getattr(args, 'thr', '')}",
        f"rerank={getattr(args, 'rerank', '')}",
        f"mmr={getattr(args, 'mmr', '')}",
    ]
    if getattr(args, "loop", False) or getattr(args, "judge", False):
        fields.append(f"loop={getattr(args, 'loop', False)}")
        fields.append(f"judge={getattr(args, 'judge', False)}")
    return "||".join(fields)


# --------------------------------------------------------------------------- #
# Score handling
# --------------------------------------------------------------------------- #
def nan_safe_scores(scores: dict[str, Any] | None) -> dict[str, float]:
    """Coerce every value to a finite float (NaN / inf / junk -> 0.0)."""
    clean: dict[str, float] = {}
    for key, value in (scores or {}).items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
        if math.isnan(number) or math.isinf(number):
            number = 0.0
        clean[key] = number
    return clean


#: Mapping of canonical metric name -> known aliases used across ragas versions.
METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "context_precision": (
        "context_precision",
        "context_precision_with_reference",
        "nonllm_context_precision_with_reference",
        "non_llm_context_precision_with_reference",
        "retrieval_precision_with_reference",
    ),
    "context_recall": (
        "context_recall",
        "context_recall_with_reference",
        "nonllm_context_recall",
        "non_llm_context_recall",
        "retrieval_recall_with_reference",
    ),
    "faithfulness": ("faithfulness", "llm_faithfulness"),
    "answer_relevancy": ("answer_relevancy", "response_relevancy", "llm_answer_relevancy"),
}


def normalize_ragas_scores(scores: dict[str, Any] | None) -> dict[str, float]:
    """Map ragas' version-dependent metric names onto stable canonical names."""
    clean = nan_safe_scores(scores)
    normalized: dict[str, float] = dict(clean)
    lower_map = {key.lower(): key for key in clean}

    for canonical, aliases in METRIC_ALIASES.items():
        for alias in aliases:
            if alias.lower() in lower_map:
                normalized[canonical] = clean[lower_map[alias.lower()]]
                break

    for canonical, needle in (("context_precision", "precision"), ("context_recall", "recall")):
        if canonical in normalized:
            continue
        for original in clean:
            lowered = original.lower()
            if "context" in lowered and needle in lowered:
                normalized[canonical] = clean[original]
                break
    return normalized


def format_scores(scores: dict[str, float], title: str) -> str:
    """Human readable score block."""
    lines = [f"=== {title} ==="]
    lines += [f"{key}: {value:.4f}" for key, value in sorted(scores.items())]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Context shaping / evidence heuristics
# --------------------------------------------------------------------------- #
def shrink_contexts(context_lists: Sequence[Sequence[str]], topn: int = 3, each_len: int = 200) -> list[list[str]]:
    """Truncate contexts before feeding them to LLM-based metrics."""
    return [[(chunk or "")[:each_len] for chunk in (chunks or [])][:topn] for chunks in context_lists]


def _section_label(line: str) -> str:
    """Strip numbering / bullet markers so ``2) Answer:`` reads as ``answer``."""
    import re

    return re.sub(r"^[\s\d.)\-•*#]+", "", line).strip().lower()


def extract_evidence_bullets(answer: str | None) -> list[str]:
    """Pull the bullet list that follows an ``Evidence`` heading."""
    bullets: list[str] = []
    collecting = False
    for line in (answer or "").splitlines():
        stripped = line.strip()
        label = _section_label(stripped)
        if not collecting and ("evidence" in label or "证据" in label):
            collecting = True
            continue
        if not collecting:
            continue
        if label.startswith(("answer", "结论")):
            break
        if stripped.startswith(("- ", "• ")):
            bullets.append(stripped[2:].strip())
    return [bullet for bullet in bullets if bullet]


def evidence_hit_ratio(context_joined: str | None, answer: str | None) -> float:
    """Fraction of the answer's evidence bullets that appear verbatim in context."""
    lowered = (context_joined or "").lower()
    bullets = extract_evidence_bullets(answer)
    if not bullets:
        return 0.0
    hits = 0
    for bullet in bullets:
        key = bullet[:60].lower()
        if key and key in lowered:
            hits += 1
    return hits / max(1, len(bullets))


def mean_of(scores: Iterable[float]) -> float:
    values = [float(value) for value in scores]
    return sum(values) / len(values) if values else 0.0


__all__ = [
    "METRIC_ALIASES",
    "cache_key",
    "evidence_hit_ratio",
    "extract_evidence_bullets",
    "format_scores",
    "is_bad_answer",
    "mean_of",
    "nan_safe_scores",
    "normalize_ragas_scores",
    "shrink_contexts",
    "str2bool",
]
