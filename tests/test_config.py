"""Tests for the configuration layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from src import config
from src.config import ConfigError, Settings, get_settings

CONFIG_VARS = [
    "CHROMA_DIR",
    "CHROMA_COLLECTION",
    "EMBEDDING_MODEL",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "EVAL_OLLAMA_MODEL",
    "RETRIEVER_K",
    "FETCH_K",
    "PER_CHUNK",
    "TRIM_CTX",
    "THR",
    "RERANK",
    "CE_MODEL",
    "MAX_AGENT_STEPS",
    "LANGSMITH_TRACING",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "API_HOST",
    "API_PORT",
    "CORS_ORIGINS",
    "SERVE_API_KEY",
]


@pytest.fixture
def clean_env(monkeypatch):
    """Remove every project variable so the built-in defaults apply."""
    for name in CONFIG_VARS:
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield monkeypatch
    get_settings.cache_clear()


class TestDefaults:
    def test_sane_defaults_without_env(self, clean_env):
        settings = Settings.from_env()
        assert settings.collection_name == "qld_rr_docs"
        assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
        assert settings.ollama_base_url == "http://localhost:11434"
        assert settings.retriever_k == 6
        assert settings.score_threshold == pytest.approx(0.20)
        assert settings.rerank is True
        assert settings.max_agent_steps == 4
        assert settings.api_port == 8008
        assert settings.langsmith_tracing is False
        assert settings.serve_api_key == ""

    def test_paths_are_absolute_under_repo_root(self, clean_env):
        settings = Settings.from_env()
        assert settings.chroma_dir == (config.REPO_ROOT / "storage/chroma_db").resolve()
        assert settings.sources_dir == config.REPO_ROOT / "data" / "sources"
        assert settings.eval_dir == config.REPO_ROOT / "data" / "eval"

    def test_data_directories_exist_in_the_repository(self, clean_env):
        assert (config.REPO_ROOT / "src").is_dir()

    def test_cors_defaults_are_split_on_commas(self, clean_env):
        settings = Settings.from_env()
        assert "http://localhost:3000" in settings.cors_origins
        assert all("," not in origin for origin in settings.cors_origins)


class TestEnvOverrides:
    def test_reads_values_from_environment(self, clean_env):
        clean_env.setenv("OLLAMA_MODEL", "qwen2.5:7b")
        clean_env.setenv("RETRIEVER_K", "9")
        clean_env.setenv("THR", "0.42")
        clean_env.setenv("RERANK", "false")
        settings = Settings.from_env()
        assert settings.ollama_model == "qwen2.5:7b"
        assert settings.retriever_k == 9
        assert settings.score_threshold == pytest.approx(0.42)
        assert settings.rerank is False

    def test_eval_model_falls_back_to_serving_model(self, clean_env):
        clean_env.setenv("OLLAMA_MODEL", "llama3.2:3b")
        assert Settings.from_env().eval_ollama_model == "llama3.2:3b"

    def test_eval_model_can_be_overridden(self, clean_env):
        clean_env.setenv("OLLAMA_MODEL", "llama3.2:3b")
        clean_env.setenv("EVAL_OLLAMA_MODEL", "qwen2.5:7b")
        assert Settings.from_env().eval_ollama_model == "qwen2.5:7b"

    def test_trailing_slash_removed_from_base_url(self, clean_env):
        clean_env.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/")
        assert Settings.from_env().ollama_base_url == "http://127.0.0.1:11434"

    def test_absolute_chroma_dir_is_kept(self, clean_env, tmp_path):
        target = tmp_path / "index"
        clean_env.setenv("CHROMA_DIR", str(target))
        assert Settings.from_env().chroma_dir == target

    def test_relative_chroma_dir_is_resolved_against_repo_root(self, clean_env):
        clean_env.setenv("CHROMA_DIR", "my/index")
        assert Settings.from_env().chroma_dir == (config.REPO_ROOT / "my/index").resolve()

    def test_empty_value_falls_back_to_default(self, clean_env):
        clean_env.setenv("OLLAMA_MODEL", "   ")
        assert Settings.from_env().ollama_model == "qwen2.5:3b"

    def test_boolean_accepts_common_spellings(self, clean_env):
        clean_env.setenv("LANGSMITH_TRACING", "YES")
        assert Settings.from_env().langsmith_tracing is True


class TestValidation:
    def test_invalid_integer_raises_config_error(self, clean_env):
        clean_env.setenv("RETRIEVER_K", "six")
        with pytest.raises(ConfigError, match="RETRIEVER_K"):
            Settings.from_env()

    def test_invalid_float_raises_config_error(self, clean_env):
        clean_env.setenv("THR", "high")
        with pytest.raises(ConfigError, match="THR"):
            Settings.from_env()

    def test_invalid_boolean_raises_config_error(self, clean_env):
        clean_env.setenv("RERANK", "sometimes")
        with pytest.raises(ConfigError, match="RERANK"):
            Settings.from_env()

    def test_settings_are_immutable(self, clean_env):
        settings = Settings.from_env()
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            settings.retriever_k = 99  # type: ignore[misc]


class TestCaching:
    def test_get_settings_is_cached(self, clean_env):
        assert get_settings() is get_settings()

    def test_cache_clear_picks_up_new_values(self, clean_env):
        clean_env.setenv("OLLAMA_MODEL", "first")
        assert get_settings().ollama_model == "first"
        clean_env.setenv("OLLAMA_MODEL", "second")
        get_settings.cache_clear()
        assert get_settings().ollama_model == "second"

    def test_module_constants_match_settings(self, clean_env):
        get_settings.cache_clear()
        assert config.OLLAMA_BASE_URL == get_settings().ollama_base_url
        assert Path(config.CHROMA_DIR) == get_settings().chroma_dir
