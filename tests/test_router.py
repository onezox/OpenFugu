"""ApiRouter: parsing router replies, validating them, retrying and falling back."""
import logging

import pytest

from fakes import auth_error, rate_limit_error
from openfugu.llm import Endpoint
from openfugu.mini import ApiRouter, RouteError, parse_route, router_messages, router_state

POOL = ["openai/alpha", "openai/beta", "openai/gamma"]
ALL = [0, 1, 2]
ENDPOINT = Endpoint("openai/router", api_key="test-key")


def turn0_messages(allowed=ALL):
    return router_messages(POOL, allowed, "What is 2+2?", router_state(0, None, None))


@pytest.mark.parametrize("reply", [
    '{"agent_id": 1, "role": "solver"}',
    '```json\n{"agent_id": 1, "role": "solver"}\n```',
    'Agent 1 is strongest here.\n{"agent_id": 1, "role": "solver"}\nThat is my choice.',
    '<think>perhaps {"agent_id": 0, "role": "thinker"}</think>{"agent_id": 1, "role": "solver"}',
    'First idea {"agent_id": 0, "role": "thinker"}, final: {"agent_id": 1, "role": "solver"}',
    '{"agent_id": 1, "role": " Solver ", "reason": "fast"}',
    'Pick {agent 1} to solve: {"agent_id": 1, "role": "solver"}',
], ids=["plain", "fenced", "prose", "think-block", "last-wins", "case-and-extra-keys",
        "brace-in-prose"])
def test_parse_route_accepts(reply):
    assert parse_route(reply, ALL) == (1, "solver")


@pytest.mark.parametrize("reply, allowed, reason", [
    ('{"agent_id": 3, "role": "solver"}', ALL, "agent_id 3 is not in the list"),
    ('{"agent_id": -1, "role": "solver"}', ALL, "agent_id -1 is not in the list"),
    ('{"agent_id": 1, "role": "solver"}', [0, 2], "agent_id 1 is not in the list"),
    ('{"agent_id": "1", "role": "solver"}', ALL, "agent_id must be an integer"),
    ('{"agent_id": 1.0, "role": "solver"}', ALL, "agent_id must be an integer"),
    ('{"agent_id": true, "role": "solver"}', ALL, "agent_id must be an integer"),
    ('{"agent_id": 1, "role": "planner"}', ALL, "role must be one of"),
    ('{"agent_id": 1, "role": ["solver"]}', ALL, "role must be one of"),
    ('{"agent_id": 1}', ALL, "no JSON object"),
    ("Agent one should solve it.", ALL, "no JSON object"),
    ("", ALL, "no JSON object"),
], ids=["out-of-range", "negative", "masked", "string-id", "float-id", "bool-id",
        "unknown-role", "non-string-role", "missing-key", "prose-only", "empty"])
def test_parse_route_rejects(reply, allowed, reason):
    with pytest.raises(RouteError, match=reason):
        parse_route(reply, allowed)


def test_route_returns_the_models_decision(fake_llm):
    fake = fake_llm(['{"agent_id": 2, "role": "verifier"}'])
    decision = ApiRouter(ENDPOINT, POOL).route(turn0_messages())
    assert decision == {"agent_id": 2, "role_name": "Verifier"}
    call = fake.calls[0]
    assert call["messages"] == turn0_messages()
    assert (call["model"], call["temperature"], call["timeout"]) == ("openai/router", 0.0, 30.0)


def test_invalid_reply_is_retried_with_a_correction(fake_llm):
    fake = fake_llm(['{"agent_id": 7, "role": "solver"}', '{"agent_id": 0, "role": "solver"}'])
    decision = ApiRouter(ENDPOINT, POOL).route(turn0_messages())
    assert decision == {"agent_id": 0, "role_name": "Worker"}
    retry = fake.calls[1]["messages"]
    assert retry[:2] == turn0_messages()
    assert retry[2] == {"role": "assistant", "content": '{"agent_id": 7, "role": "solver"}'}
    assert retry[3] == {"role": "user", "content": (
        "Your reply was invalid: agent_id 7 is not in the list. Reply with only the JSON "
        'object {"agent_id": <id>, "role": "<solver|thinker|verifier>"} using an agent_id '
        "from the list.")}


def test_masked_agent_every_time_falls_back_to_first_allowed_agent(fake_llm, caplog):
    fake = fake_llm(lambda kwargs: '{"agent_id": 0, "role": "thinker"}')
    with caplog.at_level(logging.WARNING, logger="openfugu.mini"):
        decision = ApiRouter(ENDPOINT, POOL).route(turn0_messages([1, 2]),
                                                   agent_mask=[False, True, True])
    assert decision == {"agent_id": 1, "role_name": "Worker"}
    assert len(fake.calls) == 3
    assert "falling back to agent 1 as solver" in caplog.text


def test_out_of_range_every_time_falls_back(fake_llm):
    fake = fake_llm(lambda kwargs: '{"agent_id": 9, "role": "solver"}')
    assert ApiRouter(ENDPOINT, POOL).route(turn0_messages()) == {"agent_id": 0,
                                                                  "role_name": "Worker"}
    assert len(fake.calls) == 3


def test_garbage_output_falls_back_with_a_warning(fake_llm, caplog):
    fake = fake_llm(lambda kwargs: "I cannot decide who should go next.")
    with caplog.at_level(logging.WARNING, logger="openfugu.mini"):
        decision = ApiRouter(ENDPOINT, POOL).route(turn0_messages())
    assert decision == {"agent_id": 0, "role_name": "Worker"}
    assert len(fake.calls) == 3
    assert "no JSON object" in caplog.text
    assert "falling back to agent 0 as solver" in caplog.text


def test_transient_call_failure_is_retried(fake_llm):
    fake = fake_llm([rate_limit_error(), '{"agent_id": 1, "role": "thinker"}'])
    assert ApiRouter(ENDPOINT, POOL).route(turn0_messages()) == {"agent_id": 1,
                                                                  "role_name": "Thinker"}
    assert len(fake.calls) == 2


def test_repeated_transient_failures_fall_back(fake_llm):
    fake = fake_llm(lambda kwargs: rate_limit_error())
    assert ApiRouter(ENDPOINT, POOL).route(turn0_messages()) == {"agent_id": 0,
                                                                  "role_name": "Worker"}
    assert len(fake.calls) == 3


def test_permanent_call_failure_falls_back_without_retrying(fake_llm, caplog):
    fake = fake_llm([auth_error()])
    with caplog.at_level(logging.WARNING, logger="openfugu.mini"):
        decision = ApiRouter(ENDPOINT, POOL).route(turn0_messages())
    assert decision == {"agent_id": 0, "role_name": "Worker"}
    assert len(fake.calls) == 1
    assert "AuthenticationError" in caplog.text


def test_empty_pool_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        ApiRouter(ENDPOINT, [])
