"""Shared test setup: offline, no real credentials, scripted litellm replies."""
from __future__ import annotations

import os

# litellm downloads its model cost map at import time unless told to use the
# bundled copy. Tests must not touch the network, so set this before any import
# of litellm.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

from collections.abc import Callable, Iterable

import litellm
import pytest

from fakes import FakeLLM, Reply


@pytest.fixture
def fake_llm(monkeypatch) -> Callable[[Callable[[dict], Reply] | Iterable[Reply]], FakeLLM]:
    """Install a FakeLLM. Pass a function of the call kwargs, or a list of
    replies to hand out in order (an Exception in the list is raised)."""
    def install(respond: Callable[[dict], Reply] | Iterable[Reply]) -> FakeLLM:
        if not callable(respond):
            replies = iter(respond)
            respond = lambda kwargs: next(replies)
        fake = FakeLLM(respond)
        monkeypatch.setattr(litellm, "completion", fake)
        return fake
    return install


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch) -> None:
    """Hide any FUGU_* settings of the machine running the tests."""
    for name in list(os.environ):
        if name.startswith("FUGU_"):
            monkeypatch.delenv(name)


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch) -> None:
    """Retries back off with time.sleep; skip the waiting in tests."""
    monkeypatch.setattr("openfugu.llm.time.sleep", lambda seconds: None)
