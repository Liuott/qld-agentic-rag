"""Central configuration.

Everything the project needs to know about *where things are* and *which model
to use* lives here, so that no module has to hard-code a path or an env lookup.

Design notes
------------
* Paths are resolved against the **repository root**, not the current working
  directory.  The old code used ``"./storage/chroma_db"`` and
  ``"data/eval/testset.csv"``, which broke whenever a script was launched from
  a different folder (``loop_run.py`` had to ``cd`` to the root to compensate).
* Values are parsed and validated once, with a clear error message when a
  variable is malformed (``THR=abc`` used to blow up with a bare ValueError
  deep inside an eval loop).
* Module level constants (``CHROMA_DIR``, ``OLLAMA_MODEL`` ...) are kept for
  backwards compatibility with the existing scripts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:  # python-dotenv is optional at import time (e.g. inside CI)
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - dependency is declared in pyproject

    def load_dotenv(*_args, **_kwargs) -> bool:  # type: ignore[misc]
        return False


# ``src/config.py`` -> ``src`` -> repo root
REPO_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(REPO_ROOT / ".env")


class ConfigError(ValueError):
    """Raised when an environment variable cannot be parsed."""


def _str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


_TRUE = {"1", "true", "yes", "y", "on"}
_FALSE = {"0", "false", "no", "n", "off"}


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ConfigError(f"{name} must be a boolean, got {raw!r}")


def _path(name: str, default: str) -> Path:
    """Resolve a path variable relative to the repository root."""
    raw = _str(name, default)
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _csv(name: str, default: str) -> tuple[str, ...]:
    raw = _str(name, default)
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the runtime configuration."""

    # storage / models
    chroma_dir: Path
    collection_name: str
    embedding_model: str
    ollama_base_url: str
    ollama_model: str
    eval_ollama_model: str

    # retrieval defaults (shared by CLI, API and eval)
    retriever_k: int
    fetch_k: int
    per_chunk: int
    trim_ctx: int
    score_threshold: float
    rerank: bool
    cross_encoder_model: str

    # agent
    max_agent_steps: int

    # observability
    langsmith_tracing: bool
    langsmith_api_key: str
    langsmith_project: str

    # serving
    api_host: str
    api_port: int
    cors_origins: tuple[str, ...]
    serve_api_key: str

    # data
    data_dir: Path
    sources_dir: Path
    eval_dir: Path
    experiments_dir: Path

    @classmethod
    def from_env(cls) -> Settings:
        data_dir = REPO_ROOT / "data"
        return cls(
            chroma_dir=_path("CHROMA_DIR", "storage/chroma_db"),
            collection_name=_str("CHROMA_COLLECTION", "qld_rr_docs"),
            embedding_model=_str("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
            ollama_base_url=_str("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
            ollama_model=_str("OLLAMA_MODEL", "qwen2.5:3b"),
            eval_ollama_model=_str("EVAL_OLLAMA_MODEL", _str("OLLAMA_MODEL", "qwen2.5:3b")),
            retriever_k=_int("RETRIEVER_K", 6),
            fetch_k=_int("FETCH_K", 80),
            per_chunk=_int("PER_CHUNK", 360),
            trim_ctx=_int("TRIM_CTX", 1200),
            score_threshold=_float("THR", 0.20),
            rerank=_bool("RERANK", True),
            cross_encoder_model=_str("CE_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            max_agent_steps=_int("MAX_AGENT_STEPS", 4),
            langsmith_tracing=_bool("LANGSMITH_TRACING", False),
            langsmith_api_key=_str("LANGSMITH_API_KEY", ""),
            langsmith_project=_str("LANGSMITH_PROJECT", "qld-agentic-rag"),
            api_host=_str("API_HOST", "127.0.0.1"),
            api_port=_int("API_PORT", 8008),
            cors_origins=_csv(
                "CORS_ORIGINS",
                "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173",
            ),
            serve_api_key=_str("SERVE_API_KEY", ""),
            data_dir=data_dir,
            sources_dir=data_dir / "sources",
            eval_dir=data_dir / "eval",
            experiments_dir=REPO_ROOT / "experiments",
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor -- import this instead of reading ``os.environ``."""
    return Settings.from_env()


# --------------------------------------------------------------------------- #
# Backwards compatible module level constants.
# Existing evaluation scripts do ``from src.config import CHROMA_DIR, ...``.
# --------------------------------------------------------------------------- #
_settings = get_settings()

CHROMA_DIR = str(_settings.chroma_dir)
EMBEDDING_MODEL = _settings.embedding_model
OLLAMA_BASE_URL = _settings.ollama_base_url
OLLAMA_MODEL = _settings.ollama_model
EVAL_OLLAMA_MODEL = _settings.eval_ollama_model
COLLECTION_NAME = _settings.collection_name

LANGSMITH_TRACING = _settings.langsmith_tracing
LANGSMITH_API_KEY = _settings.langsmith_api_key
LANGSMITH_PROJECT = _settings.langsmith_project

DATA_DIR = _settings.data_dir
SOURCES_DIR = _settings.sources_dir
EVAL_DIR = _settings.eval_dir
EXPERIMENTS_DIR = _settings.experiments_dir

__all__ = [
    "CHROMA_DIR",
    "COLLECTION_NAME",
    "DATA_DIR",
    "EMBEDDING_MODEL",
    "EVAL_DIR",
    "EVAL_OLLAMA_MODEL",
    "EXPERIMENTS_DIR",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "LANGSMITH_TRACING",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "REPO_ROOT",
    "SOURCES_DIR",
    "ConfigError",
    "Settings",
    "get_settings",
]
