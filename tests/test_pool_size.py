"""The pool size is len(--slot-models) everywhere: router roster and validation,
Conductor validation, and worker dispatch. 8 exceeds the old hard-coded 7."""
import json
import logging

import pytest

from openfugu.llm import Endpoint, WorkerPool
from openfugu.mini import ApiRouter, Coordinator
from openfugu.ultra import WorkflowError, main, parse_workflow, validate_workflow

SIZES = [1, 3, 8]
ROUTER = Endpoint("openai/router", api_key="test-key")
MESSAGES = [{"role": "user", "content": "hi"}]


def slots(size):
    return [f"openai/w{i}" for i in range(size)]


def decision(agent_id, role="solver"):
    return json.dumps({"agent_id": agent_id, "role": role})


@pytest.mark.parametrize("size", SIZES)
def test_router_roster_lists_every_slot_and_dispatch_reaches_the_last(fake_llm, size):
    last = size - 1
    fake = fake_llm(lambda kwargs: decision(last) if kwargs["model"] == ROUTER.model
                    else "done")
    Coordinator(ApiRouter(ROUTER, slots(size)), WorkerPool(slots(size), "test-key"),
                max_turns=1).run("Q")
    router_call, worker_call = fake.calls
    roster = "\n".join(f"{i}: openai/w{i}" for i in range(size))
    assert f"Agents (agent_id: model):\n{roster}\n\nRoles:" in router_call["messages"][0]["content"]
    assert worker_call["model"] == f"openai/w{last}"


@pytest.mark.parametrize("size", SIZES)
def test_router_rejects_an_id_one_past_the_pool_instead_of_wrapping(fake_llm, caplog, size):
    fake = fake_llm(lambda kwargs: decision(size + 1) if kwargs["model"] == ROUTER.model
                    else "done")
    with caplog.at_level(logging.WARNING, logger="openfugu.mini"):
        Coordinator(ApiRouter(ROUTER, slots(size)), WorkerPool(slots(size), "test-key"),
                    max_turns=1).run("Q")
    assert [call["model"] for call in fake.calls] == [ROUTER.model] * 3 + ["openai/w0"]
    assert f"agent_id {size + 1} is not in the list" in caplog.text
    assert "falling back to agent 0 as solver" in caplog.text


@pytest.mark.parametrize("size", SIZES)
def test_conductor_validation_uses_the_pool_size(size):
    text = f'model_id: [{size - 1}]\nsubtasks: ["x"]\naccess_list: [[]]'
    assert validate_workflow(*parse_workflow(text), pool_size=size).model_ids == [size - 1]
    with pytest.raises(WorkflowError, match=f"must be a worker index from 0 to {size - 1}"):
        validate_workflow(*parse_workflow(text.replace(f"[{size - 1}]", f"[{size}]")),
                          pool_size=size)


@pytest.mark.parametrize("size", SIZES)
def test_worker_dispatch_uses_the_pool_size(fake_llm, size):
    fake = fake_llm(lambda kwargs: kwargs["model"])
    pool = WorkerPool(slots(size), "test-key")
    assert len(pool) == size
    assert pool("Worker", MESSAGES, size - 1) == f"openai/w{size - 1}"
    with pytest.raises(ValueError, match=f"outside the pool of {size} workers"):
        pool("Worker", MESSAGES, size)
    assert len(fake.calls) == 1


@pytest.mark.parametrize("size", SIZES)
def test_ultra_cli_runs_the_last_worker_of_any_pool_size(fake_llm, monkeypatch, capsys, size):
    monkeypatch.setenv("FUGU_CONDUCTOR_MODEL", "openai/conductor")
    monkeypatch.setenv("FUGU_API_KEY", "test-key")
    workflow = f'model_id: [{size - 1}]\nsubtasks: ["Solve it"]\naccess_list: [[]]'
    fake = fake_llm(lambda kwargs: workflow if kwargs["model"] == "openai/conductor"
                    else f"reply from {kwargs['model']}")
    assert main(["--query", "Q", "--slot-models", ",".join(slots(size))]) == 0
    assert f"  {size - 1}: openai/w{size - 1}\n\n" in fake.calls[0]["messages"][0]["content"]
    assert fake.calls[1]["model"] == f"openai/w{size - 1}"
    assert f"reply from openai/w{size - 1}" in capsys.readouterr().out
