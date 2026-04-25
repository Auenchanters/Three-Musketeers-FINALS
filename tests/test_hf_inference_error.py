"""Tests for HFInferenceError formatting and status->hint mapping.

The frontend reads `hint` straight off the SSE error event, so the mapping
needs to stay readable and actionable. These tests pin the user-visible
hint for each known status code we map.
"""

from __future__ import annotations

import pytest

from web.agents import HFInferenceError


def _hint(status: int, body: str = "") -> str:
    return HFInferenceError(status=status, body=body, repo="x/y")._hint()


class TestHFInferenceErrorHint:
    def test_400_points_at_inference_providers(self) -> None:
        h = _hint(400, body='{"error":"unknown model"}')
        assert "Inference Providers" in h
        assert "different model" in h.lower()

    def test_401_points_at_token_scope(self) -> None:
        h = _hint(401, body='{"error":"Invalid username or password."}')
        assert "huggingface.co/settings/tokens" in h
        assert "Inference Providers" in h

    def test_401_via_body_substring(self) -> None:
        h = _hint(500, body='{"error":"Invalid username or password."}')
        assert "huggingface.co/settings/tokens" in h

    def test_403_says_subscription(self) -> None:
        h = _hint(403)
        assert "subscription" in h.lower() or "rejected" in h.lower()

    def test_404_says_not_deployed(self) -> None:
        h = _hint(404)
        assert "not currently deployed" in h.lower()

    def test_429_says_rate_limit(self) -> None:
        h = _hint(429)
        assert "rate-limited" in h.lower() or "rate limit" in h.lower()

    def test_503_says_cold_start(self) -> None:
        h = _hint(503)
        assert "cold-starting" in h.lower() or "30s" in h

    def test_unknown_status_has_fallback(self) -> None:
        h = _hint(418)
        assert "different model" in h.lower() or "Oracle" in h


class TestHFInferenceErrorMessage:
    def test_message_includes_status_repo_and_body_snippet(self) -> None:
        exc = HFInferenceError(
            status=400,
            body='{"error":"This model is not available on any provider"}',
            repo="Qwen/Qwen2.5-1.5B-Instruct",
        )
        msg = str(exc)
        assert "400" in msg
        assert "Qwen/Qwen2.5-1.5B-Instruct" in msg
        assert "not available on any provider" in msg

    def test_empty_body_renders_placeholder(self) -> None:
        exc = HFInferenceError(status=502, body="", repo="x/y")
        assert "(no body)" in str(exc)

    def test_body_truncated_to_200_chars(self) -> None:
        long_body = "A" * 5000
        exc = HFInferenceError(status=400, body=long_body, repo="x/y")
        # 200 A's in the snippet, full body kept on the exception
        assert "A" * 200 in str(exc)
        assert "A" * 201 not in str(exc)
        assert exc.body == long_body

    def test_newlines_collapsed_in_message(self) -> None:
        exc = HFInferenceError(
            status=400,
            body='{\n  "error": "bad"\n}',
            repo="x/y",
        )
        # No literal newlines in the formatted message — the SSE error event
        # gets stuffed into a CSS overlay where preserved newlines look broken.
        msg_first_line = str(exc).split(" - ")[0]
        assert "\n" not in msg_first_line


@pytest.mark.parametrize(
    ("status", "body", "must_contain"),
    [
        (400, "", "different model"),
        (401, "", "settings/tokens"),
        (404, "", "not currently deployed"),
        (429, "rate limit exceeded", "rate-limited"),
        (503, "currently loading", "cold-starting"),
    ],
)
def test_hint_parametrized(status: int, body: str, must_contain: str) -> None:
    assert must_contain.lower() in _hint(status, body).lower()
