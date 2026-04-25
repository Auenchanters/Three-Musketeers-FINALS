"""Tests for the LLM-output action parsers.

Both ``inference.parse_action`` (sync OpenAI client path) and
``web.agents._parse_action_json`` (async UI path) must tolerate the same
three failure modes we observed during SFT debugging on Qwen2.5-Instruct
and DeepSeek-R1-Distill variants:

  1. Markdown code fences (``json ... `` or `` ... ``).
  2. ``<think>...</think>`` reasoning blocks (DeepSeek R1).
  3. The model emitting ``"action"`` instead of the schema's ``"action_type"``.

These tests pin the contract so a future refactor can't silently regress
the live UI back to the "submits unknown" failure mode.
"""

from __future__ import annotations

import pytest

from inference import parse_action
from web.agents import _parse_action_json


CASES = [
    # (label, raw, expected action_type, expected key:value pair)
    (
        "plain_json",
        '{"action_type": "query_logs", "service": "data", "keyword": "error"}',
        "query_logs",
        ("service", "data"),
    ),
    (
        "json_fenced",
        '```json\n{"action_type": "query_logs", "service": "data"}\n```',
        "query_logs",
        ("service", "data"),
    ),
    (
        "bare_fence",
        '```\n{"action_type": "diff_commit", "commit_hash": "abc"}\n```',
        "diff_commit",
        ("commit_hash", "abc"),
    ),
    (
        "action_alias",
        '{"action": "submit", "final_cause": "commit-1", "final_chain": []}',
        "submit",
        ("final_cause", "commit-1"),
    ),
    (
        "prose_around_json",
        'Sure! Here is my next move:\n{"action_type": "fetch_trace", "trace_id": "trace-001"}\nLet me know what you think.',
        "fetch_trace",
        ("trace_id", "trace-001"),
    ),
]


@pytest.mark.parametrize("label,raw,expected_type,expected_kv", CASES)
def test_inference_parse_action_handles_failure_modes(
    label: str, raw: str, expected_type: str, expected_kv: tuple[str, str]
) -> None:
    obj = parse_action(raw)
    assert obj["action_type"] == expected_type, label
    k, v = expected_kv
    assert obj[k] == v, label


@pytest.mark.parametrize("label,raw,expected_type,expected_kv", CASES)
def test_web_agents_parse_action_handles_failure_modes(
    label: str, raw: str, expected_type: str, expected_kv: tuple[str, str]
) -> None:
    obj = _parse_action_json(raw)
    assert obj["action_type"] == expected_type, label
    k, v = expected_kv
    assert obj[k] == v, label


def test_web_agents_parser_strips_deepseek_think_tags() -> None:
    raw = (
        "<think>The user wants me to investigate the data service. "
        'I will start with logs.</think>\n'
        '{"action_type": "query_logs", "service": "data", "keyword": "error"}'
    )
    obj = _parse_action_json(raw)
    assert obj["action_type"] == "query_logs"
    assert obj["service"] == "data"


def test_inference_parse_action_falls_back_on_garbage() -> None:
    # The sync inference path is meant to be tolerant — random prose with no
    # JSON object at all returns the documented "Parse failure fallback"
    # action so the agent doesn't crash an investigation mid-run.
    obj = parse_action("hello, I think the answer is 42")
    assert obj["action_type"] == "query_logs"
    assert obj.get("reason", "").startswith("Parse failure")


def test_web_agents_parser_raises_on_garbage() -> None:
    # The async web path's contract is to raise so the runner can mark the
    # step as a parse failure (and the UI can render a fallback "thought").
    import pytest as _pt

    with _pt.raises(ValueError):
        _parse_action_json("hello, I think the answer is 42")


def test_action_construction_rejects_bogus_action_type() -> None:
    # Critical: the LLM can hallucinate an action_type that isn't in the
    # ActionType enum (e.g. "investigate", "search"). Action() should raise
    # so the runner's try/except kicks in and submits an empty fallback —
    # never let one bad turn crash the SSE stream.
    import pytest as _pt
    from models.action import Action

    with _pt.raises(Exception):
        Action(action_type="investigate", service="data")


def test_action_construction_accepts_partial_query_logs() -> None:
    # Inverse: a *valid* action_type with missing optional params (e.g.
    # query_logs without keyword) must still construct — the env will
    # reject it semantically, not at the dataclass boundary.
    from models.action import Action, ActionType

    a = Action(action_type=ActionType.QUERY_LOGS, service="data")
    assert a.action_type == ActionType.QUERY_LOGS
    assert a.service == "data"
    assert a.keyword is None
