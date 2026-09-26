"""openfugu.llm: endpoint configuration, startup checks, the retrying call path
and the shared worker pool."""
import pytest

from fakes import auth_error, rate_limit_error
from openfugu.llm import (ConfigError, Endpoint, LLMCallError, WorkerPool, check_endpoint,
                          complete, conductor_endpoint, parse_slot_models, router_endpoint)

MESSAGES = [{"role": "user", "content": "hi"}]
ENDPOINT = Endpoint("openai/test-model", api_key="test-key")


def set_env(monkeypatch, **values: str) -> None:
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_router_endpoint_falls_back_to_conductor_then_worker_settings(monkeypatch):
    set_env(monkeypatch, FUGU_CONDUCTOR_MODEL="openai/conductor",
            FUGU_CONDUCTOR_API_KEY="conductor-key", FUGU_BASE_URL="http://workers.test/v1")
    assert router_endpoint() == Endpoint("openai/conductor", "conductor-key",
                                         "http://workers.test/v1")


def test_router_settings_take_precedence(monkeypatch):
    set_env(monkeypatch,
            FUGU_CONDUCTOR_MODEL="openai/conductor", FUGU_CONDUCTOR_API_KEY="conductor-key",
            FUGU_CONDUCTOR_BASE_URL="http://conductor.test/v1",
            FUGU_ROUTER_MODEL="openai/router", FUGU_ROUTER_API_KEY="router-key",
            FUGU_ROUTER_BASE_URL="http://router.test/v1")
    assert router_endpoint() == Endpoint("openai/router", "router-key", "http://router.test/v1")


def test_router_key_falls_back_to_worker_key(monkeypatch):
    set_env(monkeypatch, FUGU_ROUTER_MODEL="openai/router", FUGU_API_KEY="worker-key")
    assert router_endpoint() == Endpoint("openai/router", "worker-key", None)


def test_blank_variables_count_as_unset(monkeypatch):
    set_env(monkeypatch, FUGU_ROUTER_MODEL="  ", FUGU_CONDUCTOR_MODEL="openai/conductor")
    assert router_endpoint().model == "openai/conductor"


def test_missing_router_model_is_a_config_error():
    with pytest.raises(ConfigError, match="FUGU_ROUTER_MODEL or FUGU_CONDUCTOR_MODEL"):
        router_endpoint()


def test_conductor_endpoint_falls_back_to_worker_settings(monkeypatch):
    set_env(monkeypatch, FUGU_CONDUCTOR_MODEL="openai/conductor", FUGU_API_KEY="worker-key",
            FUGU_BASE_URL="http://workers.test/v1")
    assert conductor_endpoint() == Endpoint("openai/conductor", "worker-key",
                                            "http://workers.test/v1")


def test_conductor_settings_take_precedence_and_ignore_router_settings(monkeypatch):
    set_env(monkeypatch, FUGU_ROUTER_MODEL="openai/router", FUGU_ROUTER_API_KEY="router-key",
            FUGU_CONDUCTOR_MODEL="openai/conductor", FUGU_CONDUCTOR_API_KEY="conductor-key",
            FUGU_CONDUCTOR_BASE_URL="http://conductor.test/v1", FUGU_API_KEY="worker-key")
    assert conductor_endpoint() == Endpoint("openai/conductor", "conductor-key",
                                            "http://conductor.test/v1")


def test_missing_conductor_model_is_a_config_error(monkeypatch):
    set_env(monkeypatch, FUGU_ROUTER_MODEL="openai/router")
    with pytest.raises(ConfigError, match="no Conductor model: set FUGU_CONDUCTOR_MODEL$"):
        conductor_endpoint()


def test_endpoint_repr_hides_the_key():
    assert "test-key" not in repr(ENDPOINT)


def test_check_endpoint_rejects_a_model_without_provider():
    with pytest.raises(ConfigError, match="cannot resolve its provider"):
        check_endpoint(Endpoint("my-finetune", api_key="k"), "router")


def test_check_endpoint_reports_missing_provider_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        check_endpoint(Endpoint("openai/test-model"), "router")


def test_check_endpoint_accepts_a_configured_key():
    check_endpoint(ENDPOINT, "router")


def test_complete_returns_text_and_passes_call_settings(fake_llm):
    fake = fake_llm(["hello"])
    endpoint = Endpoint("openai/test-model", "test-key", "http://host.test/v1")
    text = complete(endpoint, MESSAGES, max_tokens=5, temperature=0.0, timeout=7.0, attempts=3)
    assert text == "hello"
    call = fake.calls[0]
    assert (call["model"], call["api_key"], call["api_base"]) == (
        "openai/test-model", "test-key", "http://host.test/v1")
    assert (call["max_tokens"], call["temperature"], call["timeout"]) == (5, 0.0, 7.0)
    assert call["max_retries"] == 0      # the OpenAI SDK must not retry behind our back


def test_complete_retries_transient_errors(fake_llm):
    fake = fake_llm([rate_limit_error(), rate_limit_error(), "ok"])
    assert complete(ENDPOINT, MESSAGES, max_tokens=5, temperature=0.0, timeout=5,
                    attempts=3) == "ok"
    assert len(fake.calls) == 3


def test_complete_gives_up_after_the_last_attempt(fake_llm):
    fake = fake_llm([rate_limit_error()] * 3)
    with pytest.raises(LLMCallError, match="after 3 attempts") as info:
        complete(ENDPOINT, MESSAGES, max_tokens=5, temperature=0.0, timeout=5, attempts=3)
    assert info.value.retryable
    assert len(fake.calls) == 3


def test_complete_does_not_retry_permanent_errors_and_redacts_the_key(fake_llm):
    fake = fake_llm([auth_error("Incorrect API key provided: test-key")])
    with pytest.raises(LLMCallError) as info:
        complete(ENDPOINT, MESSAGES, max_tokens=5, temperature=0.0, timeout=5, attempts=3)
    assert not info.value.retryable
    assert len(fake.calls) == 1
    assert "AuthenticationError" in str(info.value)
    assert "test-key" not in str(info.value)
    # the unredacted provider exception must not ride along into logged tracebacks
    assert info.value.__cause__ is None and info.value.__suppress_context__


# ---- worker pool ---------------------------------------------------------------
TWO_SLOTS = ["openai/w0", "openai/w1"]


def test_worker_pool_uses_fugu_worker_settings(fake_llm, monkeypatch):
    set_env(monkeypatch, FUGU_API_KEY="worker-key", FUGU_BASE_URL="http://workers.test/v1",
            FUGU_WORKER_MODEL="openai/ignored")
    fake = fake_llm(["hi"])
    assert WorkerPool.from_env(TWO_SLOTS)("Worker", MESSAGES, 1) == "hi"
    call = fake.calls[0]
    assert (call["model"], call["api_key"], call["api_base"]) == (
        "openai/w1", "worker-key", "http://workers.test/v1")
    assert (call["temperature"], call["timeout"], call["max_tokens"]) == (0.2, 120.0, 1024)


def test_worker_pool_does_not_read_openai_variables_itself(fake_llm, monkeypatch):
    set_env(monkeypatch, OPENAI_API_KEY="sk-openai", OPENAI_BASE_URL="http://openai.test/v1")
    fake = fake_llm(["hi"])
    WorkerPool.from_env(TWO_SLOTS)("Worker", MESSAGES, 0)
    assert "api_key" not in fake.calls[0]
    assert "api_base" not in fake.calls[0]


@pytest.mark.parametrize("agent_id", [-1, 2, 7])
def test_worker_pool_rejects_out_of_range_ids_without_wrapping(fake_llm, agent_id):
    fake = fake_llm([])
    with pytest.raises(ValueError, match=f"agent_id {agent_id} is outside the pool of 2 workers"):
        WorkerPool(TWO_SLOTS, "test-key")("Worker", MESSAGES, agent_id)
    assert fake.calls == []


def test_worker_failure_raises_llm_call_error_after_retrying_transient_errors(fake_llm):
    fake = fake_llm([rate_limit_error(), auth_error()])
    with pytest.raises(LLMCallError) as info:
        WorkerPool(TWO_SLOTS, "test-key")("Worker", MESSAGES, 0)
    assert not info.value.retryable
    assert len(fake.calls) == 2


def test_worker_pool_check_names_the_slot_without_a_key(monkeypatch):
    set_env(monkeypatch, OPENAI_API_KEY="sk-openai")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    pool = WorkerPool(["openai/w0", "anthropic/claude-test"])
    with pytest.raises(ConfigError, match="worker 1 model 'anthropic/claude-test'.*ANTHROPIC_API_KEY"):
        pool.check()


def test_worker_pool_check_accepts_a_fugu_key_for_every_slot(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    WorkerPool(["openai/w0", "anthropic/claude-test"], "test-key").check()


def test_worker_pool_check_rejects_a_model_without_provider():
    with pytest.raises(ConfigError, match="worker 0 model 'my-finetune'"):
        WorkerPool(["my-finetune"], "test-key").check()


def test_empty_worker_pool_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        WorkerPool([])


@pytest.mark.parametrize("csv, models", [
    ("openai/w0", ["openai/w0"]),
    (" openai/w0 , anthropic/claude-test ", ["openai/w0", "anthropic/claude-test"]),
])
def test_parse_slot_models(csv, models):
    assert parse_slot_models(csv) == models


@pytest.mark.parametrize("csv", ["", " ", "openai/w0,", "openai/w0,,openai/w1"])
def test_parse_slot_models_rejects_empty_entries(csv):
    with pytest.raises(ConfigError, match="no empty entries"):
        parse_slot_models(csv)
