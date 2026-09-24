"""Test doubles for litellm. Replies go through litellm's own `mock_response`."""
from __future__ import annotations

from collections.abc import Callable

import litellm

_REAL_COMPLETION = litellm.completion

Reply = str | Exception


class FakeLLM:
    """Replaces `litellm.completion` and answers through litellm's own
    `mock_response`, so replies and errors are real litellm objects and no
    request leaves the process. Every call's keyword arguments are recorded.
    """

    def __init__(self, respond: Callable[[dict], Reply]) -> None:
        self.respond = respond
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return _REAL_COMPLETION(**kwargs, mock_response=self.respond(kwargs))


def rate_limit_error() -> Exception:
    """A transient litellm error (retryable)."""
    return litellm.RateLimitError(message="rate limited", llm_provider="openai", model="test")


def auth_error(message: str = "invalid API key") -> Exception:
    """A permanent litellm error (not retryable)."""
    return litellm.AuthenticationError(message=message, llm_provider="openai", model="test")
