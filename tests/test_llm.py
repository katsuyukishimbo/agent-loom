"""Tests for the unified LLM client (fake mode only — real mode requires keys)."""

from __future__ import annotations

import json

import pytest

from agent_loom.llm import _extract_json, complete


async def test_fake_planner_returns_contract_shape() -> None:
    resp = await complete(
        model="claude-opus-4-7",
        system="ignored",
        user="Write a Python function fib(n).",
        role="planner",
    )
    assert resp.provider == "fake"
    assert "goal" in resp.parsed
    assert "acceptance_criteria" in resp.parsed
    assert resp.cost_usd == 0.0


async def test_fake_generator_returns_artifact_shape() -> None:
    resp = await complete(
        model="claude-haiku-4-5",
        system="ignored",
        user='{"goal": "fib"}',
        role="generator",
    )
    assert resp.parsed["kind"] == "text"
    assert "def fib" in resp.parsed["content"]


async def test_fake_verifier_returns_judgement_shape() -> None:
    resp = await complete(
        model="claude-sonnet-4-6",
        system="ignored",
        user='{"contract": {}, "artifact": {}}',
        role="verifier",
    )
    assert resp.parsed["passed"] is True
    assert 0.0 <= resp.parsed["score"] <= 1.0


def test_extract_json_strips_prose() -> None:
    text = 'Here you go:\n```json\n{"a": 1}\n```\nThanks.'
    assert _extract_json(text) == {"a": 1}


def test_extract_json_raises_on_missing_object() -> None:
    with pytest.raises(ValueError):
        _extract_json("no braces here")


async def test_fake_response_is_json_serialisable() -> None:
    """The fake text field must be valid JSON for callers that re-parse it."""
    resp = await complete(
        model="claude-haiku-4-5",
        system="x",
        user="y",
        role="generator",
    )
    assert json.loads(resp.text) == resp.parsed
