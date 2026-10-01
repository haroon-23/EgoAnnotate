"""Output-contract layer for the VLM dual backend (Phase D):
parse_json_strict — strict extraction, one repair re-prompt, then None
(the stage's own regex fallbacks take it from there). Model-free.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vlm_backend import build_repair_prompt, parse_json_strict


# ---------------------------------------------------------------------------
# Strict pass
# ---------------------------------------------------------------------------


def test_clean_json_list():
    assert parse_json_strict('[{"a": 1}, {"b": 2}]') == [{"a": 1}, {"b": 2}]


def test_clean_json_object():
    assert parse_json_strict('{"a": 1}') == {"a": 1}


def test_markdown_wrapped_json():
    text = '```json\n[{"a": 1}]\n```'
    assert parse_json_strict(text) == [{"a": 1}]


def test_bare_fence_wrapped_json():
    assert parse_json_strict('```\n{"a": 1}\n```') == {"a": 1}


def test_prose_around_json():
    text = 'Here you go:\n[{"a": 1}, {"b": 2}]\nHope that helps!'
    assert parse_json_strict(text) == [{"a": 1}, {"b": 2}]


def test_truncated_json_is_not_salvageable():
    assert parse_json_strict('[{"name": "cup"') is None


def test_garbage_returns_none_without_repair():
    assert parse_json_strict("no json here at all") is None


def test_non_string_input_returns_none():
    assert parse_json_strict(None) is None
    assert parse_json_strict("") is None


# ---------------------------------------------------------------------------
# Repair pass
# ---------------------------------------------------------------------------


def test_repair_fixes_garbage():
    calls = []

    def repair(prompt):
        calls.append(prompt)
        return '{"a": 9}'

    assert parse_json_strict("total garbage", repair_fn=repair) == {"a": 9}
    assert len(calls) == 1
    assert "ONLY the JSON" in calls[0]
    assert "total garbage" in calls[0]  # original attempt is included for context


def test_repair_still_garbage_returns_none():
    assert parse_json_strict("garbage", repair_fn=lambda p: "still garbage") is None


def test_repair_not_called_when_strict_succeeds():
    calls = []
    out = parse_json_strict('[{"a": 1}]', repair_fn=lambda p: calls.append(p) or '{"x": 0}')
    assert out == [{"a": 1}]
    assert calls == []


def test_repair_called_exactly_once():
    calls = []
    parse_json_strict("garbage", repair_fn=lambda p: calls.append(p) or "garbage")
    assert len(calls) == 1


def test_repair_exception_returns_none():
    def boom(p):
        raise RuntimeError("repair exploded")

    assert parse_json_strict("garbage", repair_fn=boom) is None


def test_repair_result_also_uses_bracket_extraction():
    fixed = parse_json_strict("garbage", repair_fn=lambda p: "Sure: ```json\n[1, 2]\n```")
    assert fixed == [1, 2]


# ---------------------------------------------------------------------------
# Repair prompt builder
# ---------------------------------------------------------------------------


def test_build_repair_prompt_contents():
    prompt = build_repair_prompt("not json", "expected a JSON array")
    assert "ONLY the JSON" in prompt
    assert "not json" in prompt
    assert "expected a JSON array" in prompt
