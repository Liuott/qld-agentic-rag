"""Tests for the shared evaluation helpers."""

from __future__ import annotations

import argparse

import pytest

from src.eval.common import (
    cache_key,
    evidence_hit_ratio,
    extract_evidence_bullets,
    format_scores,
    is_bad_answer,
    nan_safe_scores,
    normalize_ragas_scores,
    shrink_contexts,
    str2bool,
)


def make_args(**overrides) -> argparse.Namespace:
    base = {
        "k": 6,
        "trim_ctx": 1200,
        "per_chunk": 360,
        "thr": 0.2,
        "rerank": True,
        "mmr": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


class TestStr2Bool:
    @pytest.mark.parametrize("value", ["1", "true", "Yes", "y", "ON", True])
    def test_truthy(self, value):
        assert str2bool(value) is True

    @pytest.mark.parametrize("value", ["0", "false", "No", "n", "off", False])
    def test_falsy(self, value):
        assert str2bool(value) is False

    def test_rejects_garbage(self):
        with pytest.raises(argparse.ArgumentTypeError):
            str2bool("maybe")


class TestIsBadAnswer:
    def test_rejects_short_answer(self):
        assert is_bad_answer("too short") is True

    def test_rejects_missing_sections(self):
        assert is_bad_answer("x" * 200) is True

    def test_accepts_well_formed_answer(self):
        answer = "1) Evidence:\n- the rule says X\n2) Answer:\n" + "y" * 80
        assert is_bad_answer(answer) is False

    def test_handles_none_and_empty(self):
        assert is_bad_answer(None) is True
        assert is_bad_answer("") is True


class TestCacheKey:
    def test_legacy_key_layout_is_preserved(self):
        key = cache_key("  What is the speed limit?  ", make_args(), "qwen2.5:3b")
        assert key == (
            "What is the speed limit?||eval_model=qwen2.5:3b||k=6||trim=1200||chunk=360"
            "||thr=0.2||rerank=True||mmr=False"
        )

    def test_loop_flag_extends_the_key(self):
        key = cache_key("q", make_args(loop=True), "m")
        assert key.endswith("||loop=True||judge=False")

    def test_different_settings_produce_different_keys(self):
        assert cache_key("q", make_args(), "m") != cache_key("q", make_args(k=8), "m")


class TestScoreHandling:
    def test_nan_safe_replaces_non_finite(self):
        cleaned = nan_safe_scores({"a": float("nan"), "b": float("inf"), "c": "0.5", "d": None})
        assert cleaned == {"a": 0.0, "b": 0.0, "c": 0.5, "d": 0.0}

    def test_normalize_maps_known_aliases(self):
        normalized = normalize_ragas_scores(
            {
                "non_llm_context_precision_with_reference": 0.7,
                "non_llm_context_recall": 0.4,
                "llm_faithfulness": 0.9,
            }
        )
        assert normalized["context_precision"] == 0.7
        assert normalized["context_recall"] == 0.4
        assert normalized["faithfulness"] == 0.9

    def test_normalize_falls_back_to_fuzzy_match(self):
        normalized = normalize_ragas_scores({"my_context_precision_score": 0.3})
        assert normalized["context_precision"] == 0.3

    def test_normalize_keeps_unknown_metrics(self):
        assert normalize_ragas_scores({"custom": 0.2})["custom"] == 0.2

    def test_format_scores_is_sorted(self):
        text = format_scores({"b": 0.2, "a": 0.5}, "Raw")
        assert text.splitlines()[0] == "=== Raw ==="
        assert text.index("a:") < text.index("b:")


class TestContextHelpers:
    def test_shrink_contexts_limits_topn_and_length(self):
        shrunk = shrink_contexts([["x" * 500, "y" * 500, "z" * 500, "w" * 500]], topn=2, each_len=10)
        assert shrunk == [["x" * 10, "y" * 10]]

    def test_shrink_handles_none_entries(self):
        assert shrink_contexts([[None, "abc"]], topn=5, each_len=5) == [["", "abc"]]

    def test_extract_bullets(self):
        answer = "1) Evidence:\n- first rule\n- second rule\n2) Answer:\n- not a bullet"
        assert extract_evidence_bullets(answer) == ["first rule", "second rule"]

    def test_extract_bullets_without_section(self):
        assert extract_evidence_bullets("- orphan bullet") == []

    def test_hit_ratio(self):
        context = "The driver must give way to pedestrians at a crossing."
        answer = (
            "1) Evidence:\n"
            "- The driver must give way to pedestrians at a crossing.\n"
            "- Something completely different that is absent.\n"
            "2) Answer:\n yes"
        )
        assert evidence_hit_ratio(context, answer) == 0.5

    def test_hit_ratio_without_bullets(self):
        assert evidence_hit_ratio("ctx", "no evidence section") == 0.0
