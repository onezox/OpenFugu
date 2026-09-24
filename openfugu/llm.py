# OpenFugu — Apache-2.0. Part of an independent, open reimplementation of
# the Fugu orchestrator. NOT affiliated with Sakana AI. See NOTICE.
"""The single place where OpenFugu talks to litellm.

Endpoint configuration comes only from FUGU_* environment variables. When no
API key is configured, litellm's own provider variables apply, and
`check_endpoint` verifies them at startup. `complete` is the one call path:
every request has a timeout, transient failures are retried with exponential
backoff, and every other failure surfaces as `LLMCallError`.
"""
from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass, field

import litellm

logger = logging.getLogger(__name__)

# litellm prints "Provider List" / "Give Feedback" banners to stdout on errors.
litellm.suppress_debug_info = True

RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    litellm.Timeout,
    litellm.APIConnectionError,
    litellm.RateLimitError,
    litellm.ServiceUnavailableError,
    litellm.InternalServerError,
    litellm.BadGatewayError,
)
BACKOFF_BASE_S = 1.0
BACKOFF_MAX_S = 8.0


class ConfigError(Exception):
    """Configuration is missing or invalid; raised at startup."""


class LLMCallError(Exception):
    """A model call failed.

    `retryable` is True when the failure was transient (timeout, connection,
    rate limit, 5xx) and trying again later may succeed.
    """

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class Endpoint:
    """A litellm model id and the credentials used to call it."""

    model: str
    api_key: str | None = field(default=None, repr=False)
    api_base: str | None = None


def _env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def _first_env(*names: str) -> str | None:
    return next((value for value in map(_env, names) if value), None)


def router_endpoint() -> Endpoint:
    """Router endpoint: FUGU_ROUTER_* settings, each falling back to FUGU_CONDUCTOR_*.

    The key and base URL fall back further to FUGU_API_KEY and FUGU_BASE_URL.
    """
    model = _first_env("FUGU_ROUTER_MODEL", "FUGU_CONDUCTOR_MODEL")
    if not model:
        raise ConfigError("no router model: set FUGU_ROUTER_MODEL or FUGU_CONDUCTOR_MODEL")
    return Endpoint(
        model=model,
        api_key=_first_env("FUGU_ROUTER_API_KEY", "FUGU_CONDUCTOR_API_KEY", "FUGU_API_KEY"),
        api_base=_first_env("FUGU_ROUTER_BASE_URL", "FUGU_CONDUCTOR_BASE_URL", "FUGU_BASE_URL"),
    )


def check_endpoint(endpoint: Endpoint, purpose: str) -> None:
    """Fail fast if litellm cannot resolve the provider or find credentials.

    Only checks that configuration is present; it makes no network call.
    """
    try:
        litellm.get_llm_provider(endpoint.model, api_base=endpoint.api_base)
    except Exception as exc:
        raise ConfigError(
            f"{purpose} model {endpoint.model!r}: litellm cannot resolve its provider "
            f"(use a 'provider/model' id such as 'openai/my-model')") from exc
    if endpoint.api_key:
        return
    missing = litellm.validate_environment(endpoint.model).get("missing_keys") or []
    if missing:
        raise ConfigError(
            f"{purpose} model {endpoint.model!r}: no API key configured; set the FUGU_* "
            f"key variable or {', '.join(missing)}")


def backoff_delay(attempt: int) -> float:
    """Seconds to wait after failed attempt number `attempt` (1-based), with jitter."""
    delay = min(BACKOFF_MAX_S, BACKOFF_BASE_S * 2 ** (attempt - 1))
    return delay * random.uniform(0.5, 1.0)


def _describe(exc: Exception, endpoint: Endpoint) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if endpoint.api_key:
        text = text.replace(endpoint.api_key, "***")
    return text


def complete(endpoint: Endpoint, messages: list[dict], *, max_tokens: int,
             temperature: float, timeout: float, attempts: int) -> str:
    """Return the assistant text for `messages`.

    Transient failures are retried up to `attempts` calls in total, sleeping
    `backoff_delay` between them. Non-transient failures (authentication, bad
    request, unknown model, ...) are raised immediately. Every failure is
    raised as `LLMCallError`, with the API key redacted from its message.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    kwargs = {"model": endpoint.model, "messages": messages, "max_tokens": max_tokens,
              "temperature": temperature, "timeout": timeout}
    if endpoint.api_key:
        kwargs["api_key"] = endpoint.api_key
    if endpoint.api_base:
        kwargs["api_base"] = endpoint.api_base

    attempt = 1
    while True:
        try:
            response = litellm.completion(**kwargs)
        except RETRYABLE_ERRORS as exc:
            detail = _describe(exc, endpoint)
            if attempt == attempts:
                raise LLMCallError(f"{endpoint.model}: {detail} (after {attempts} attempts)",
                                   retryable=True) from exc
            delay = backoff_delay(attempt)
            logger.warning("%s: attempt %d/%d failed (%s); retrying in %.1fs",
                           endpoint.model, attempt, attempts, detail, delay)
            time.sleep(delay)
            attempt += 1
        except Exception as exc:
            raise LLMCallError(f"{endpoint.model}: {_describe(exc, endpoint)}",
                               retryable=False) from exc
        else:
            choices = response.choices
            return (choices[0].message.content if choices else None) or ""
