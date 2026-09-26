# OpenFugu — Apache-2.0. Part of an independent, open reimplementation of
# the Fugu orchestrator. NOT affiliated with Sakana AI. See NOTICE.
"""The single place where OpenFugu talks to litellm.

Endpoint configuration comes only from FUGU_* environment variables. When no
API key is configured, litellm's own provider variables apply, and
`check_endpoint` verifies them at startup. `complete` is the one call path:
every request has a timeout, transient failures are retried with exponential
backoff, and every other failure surfaces as `LLMCallError`. `WorkerPool` is
the worker pool shared by the TRINITY coordinator and the Conductor.
"""
from __future__ import annotations

import logging
import os
import random
import time
from collections.abc import Sequence
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

WORKER_ATTEMPTS = 3
WORKER_TIMEOUT_S = 120.0
WORKER_MAX_TOKENS = 1024
WORKER_TEMPERATURE = 0.2


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


def _endpoint(purpose: str, prefixes: tuple[str, ...]) -> Endpoint:
    """Build an endpoint from {prefix}_MODEL, _API_KEY and _BASE_URL, trying the
    prefixes in order; the key and base URL finally fall back to the worker
    settings FUGU_API_KEY and FUGU_BASE_URL."""
    model_vars = [f"{prefix}_MODEL" for prefix in prefixes]
    model = _first_env(*model_vars)
    if not model:
        raise ConfigError(f"no {purpose} model: set {' or '.join(model_vars)}")
    return Endpoint(
        model=model,
        api_key=_first_env(*(f"{prefix}_API_KEY" for prefix in prefixes), "FUGU_API_KEY"),
        api_base=_first_env(*(f"{prefix}_BASE_URL" for prefix in prefixes), "FUGU_BASE_URL"),
    )


def router_endpoint() -> Endpoint:
    """Router endpoint: FUGU_ROUTER_* settings, each falling back to FUGU_CONDUCTOR_*.

    The key and base URL fall back further to FUGU_API_KEY and FUGU_BASE_URL.
    """
    return _endpoint("router", ("FUGU_ROUTER", "FUGU_CONDUCTOR"))


def conductor_endpoint() -> Endpoint:
    """Conductor endpoint: FUGU_CONDUCTOR_* settings.

    The key and base URL fall back to FUGU_API_KEY and FUGU_BASE_URL.
    """
    return _endpoint("Conductor", ("FUGU_CONDUCTOR",))


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


def configure_logging(level: int) -> None:
    """Set up logging for a CLI entry point at `level`.

    litellm, the OpenAI SDK and httpx log every request at INFO; they are kept
    at WARNING so the application's own messages stay readable.
    """
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for name in ("LiteLLM", "openai", "httpx"):
        logging.getLogger(name).setLevel(logging.WARNING)


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
    raised as `LLMCallError`, with the API key redacted from its message. The
    original exception is not chained (`from None`): provider error text can
    echo credentials, and a logged traceback would print it unredacted.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    # max_retries=0 turns off the OpenAI SDK's own retries, so this loop is the
    # only retry layer and `attempts` is the real number of calls.
    kwargs = {"model": endpoint.model, "messages": messages, "max_tokens": max_tokens,
              "temperature": temperature, "timeout": timeout, "max_retries": 0}
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
                if attempts > 1:
                    detail += f" (after {attempts} attempts)"
                raise LLMCallError(f"{endpoint.model}: {detail}", retryable=True) from None
            delay = backoff_delay(attempt)
            logger.warning("%s: attempt %d/%d failed (%s); retrying in %.1fs",
                           endpoint.model, attempt, attempts, detail, delay)
            time.sleep(delay)
            attempt += 1
        except Exception as exc:
            raise LLMCallError(f"{endpoint.model}: {_describe(exc, endpoint)}",
                               retryable=False) from None
        else:
            choices = response.choices
            return (choices[0].message.content if choices else None) or ""


def parse_slot_models(csv: str) -> list[str]:
    """Split a --slot-models value into litellm model ids; entry i serves agent i."""
    models = [model.strip() for model in csv.split(",")]
    if not all(models):
        raise ConfigError(f"--slot-models {csv!r}: expected comma-separated litellm model "
                          f"ids with no empty entries")
    return models


class WorkerPool:
    """The worker pool: agent i is served by slot model i through litellm.

    Calling the pool runs one worker turn, `(label, messages, agent_id) -> reply`,
    where `label` (a role name or a subtask) is only used for logging. Every
    call goes through `complete`, so a failure raises `LLMCallError`. The pool
    size is exactly the number of slot models; an agent id outside it is an
    error, never wrapped around.
    """

    def __init__(self, slot_models: Sequence[str], api_key: str | None = None,
                 api_base: str | None = None) -> None:
        if not slot_models:
            raise ValueError("the worker pool is empty")
        self.endpoints = [Endpoint(model, api_key, api_base) for model in slot_models]

    @classmethod
    def from_env(cls, slot_models: Sequence[str]) -> WorkerPool:
        """A pool using FUGU_API_KEY and FUGU_BASE_URL when set; otherwise
        litellm's own provider variables apply."""
        return cls(slot_models, _env("FUGU_API_KEY"), _env("FUGU_BASE_URL"))

    def __len__(self) -> int:
        return len(self.endpoints)

    def check(self) -> None:
        """Fail fast if any slot's provider or credentials cannot be resolved."""
        for agent_id, endpoint in enumerate(self.endpoints):
            check_endpoint(endpoint, f"worker {agent_id}")

    def __call__(self, label: str, messages: list[dict], agent_id: int) -> str:
        """Run agent `agent_id` on `messages` and return its reply."""
        if not 0 <= agent_id < len(self.endpoints):
            raise ValueError(f"agent_id {agent_id} is outside the pool of "
                             f"{len(self.endpoints)} workers")
        logger.debug("worker %d (%s): %.60s", agent_id, self.endpoints[agent_id].model, label)
        return complete(self.endpoints[agent_id], messages, max_tokens=WORKER_MAX_TOKENS,
                        temperature=WORKER_TEMPERATURE, timeout=WORKER_TIMEOUT_S,
                        attempts=WORKER_ATTEMPTS)
