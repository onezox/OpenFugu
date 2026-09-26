"""serve.py: the real HTTP handler end to end, error handling and startup checks.

Servers bind 127.0.0.1 on a free port; every model call is answered offline by
litellm's mock_response.
"""
import http.client
import json
import logging
import re
import threading
import time

import pytest

from fakes import auth_error
from openfugu.llm import Endpoint, WorkerPool
from openfugu.mini import ApiRouter
from openfugu.serve import FuguServer, main

ROUTER_MODEL = "openai/router"
POOL = ["openai/w0", "openai/w1"]
SOLVER_REPLY = "<think>2+2 is 4</think>The answer is 4."


@pytest.fixture
def fugu_server():
    """Start FuguServer on 127.0.0.1 with a real router and worker pool."""
    running = []

    def start(max_turns=5):
        server = FuguServer(("127.0.0.1", 0), ApiRouter(Endpoint(ROUTER_MODEL, "test-key"), POOL),
                            WorkerPool(POOL, "test-key"), max_turns)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05},
                                  daemon=True)
        thread.start()
        running.append((server, thread))
        return server.server_address

    yield start
    for server, thread in running:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def request(address, method, path, body=None):
    """Send one request; return (status, raw response body)."""
    connection = http.client.HTTPConnection(*address, timeout=10)
    try:
        headers = {"Content-Type": "application/json"} if body is not None else {}
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def chat(address, payload):
    return request(address, "POST", "/v1/chat/completions", json.dumps(payload).encode())


def solve_then_accept(kwargs):
    """Router: agent 0 solves, then agent 1 verifies. Workers answer by prompt."""
    if kwargs["model"] == ROUTER_MODEL:
        solved = "<reference_thought_0>" in kwargs["messages"][1]["content"]
        return json.dumps({"agent_id": 1, "role": "verifier"} if solved
                          else {"agent_id": 0, "role": "solver"})
    if kwargs["messages"][-1]["content"].startswith("Please carefully review"):
        return "ACCEPT - the answer is correct"
    return SOLVER_REPLY


def test_request_end_to_end_router_worker_verifier_accept(fake_llm, fugu_server):
    fake = fake_llm(solve_then_accept)
    before = int(time.time())
    status, raw = chat(fugu_server(), {"messages": [{"role": "user", "content": "What is 2+2?"}]})

    assert status == 200
    body = json.loads(raw)
    assert re.fullmatch(r"chatcmpl-[0-9a-f]{24}", body["id"])
    assert before <= body["created"] <= int(time.time())
    expected = {
        "id": body["id"],
        "object": "chat.completion",
        "created": body["created"],
        "model": "fugu",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": SOLVER_REPLY},
            "finish_reason": "stop",
        }],
        "usage": {"fugu_turns": 2},
    }
    assert raw == json.dumps(expected).encode()     # same keys, order and separators as before
    assert [call["model"] for call in fake.calls] == [ROUTER_MODEL, "openai/w0",
                                                      ROUTER_MODEL, "openai/w1"]
    assert SOLVER_REPLY in fake.calls[3]["messages"][-1]["content"]


def test_request_model_name_is_echoed(fake_llm, fugu_server):
    fake_llm(solve_then_accept)
    status, raw = chat(fugu_server(), {"model": "my-client-model",
                                       "messages": [{"role": "user", "content": "Q"}]})
    assert status == 200
    assert json.loads(raw)["model"] == "my-client-model"


def test_worker_failure_returns_a_generic_500_and_logs_the_details(fake_llm, fugu_server,
                                                                   caplog):
    def respond(kwargs):
        if kwargs["model"] == ROUTER_MODEL:
            return json.dumps({"agent_id": 0, "role": "solver"})
        return auth_error("Incorrect API key provided: test-key")
    fake_llm(respond)
    with caplog.at_level(logging.ERROR, logger="openfugu.serve"):
        status, raw = chat(fugu_server(), {"messages": [{"role": "user", "content": "Q"}]})
    assert status == 500
    assert raw == b'{"error": "internal server error"}'
    assert "POST /v1/chat/completions failed" in caplog.text
    assert "openai/w0: AuthenticationError" in caplog.text
    assert "test-key" not in caplog.text


def test_unexpected_error_returns_the_same_generic_500_with_a_traceback(
        fake_llm, fugu_server, monkeypatch, caplog):
    def broken_run(self, query):
        raise RuntimeError("bug in the coordinator")
    monkeypatch.setattr("openfugu.mini.Coordinator.run", broken_run)
    fake_llm([])
    with caplog.at_level(logging.ERROR, logger="openfugu.serve"):
        status, raw = chat(fugu_server(), {"messages": [{"role": "user", "content": "Q"}]})
    assert (status, raw) == (500, b'{"error": "internal server error"}')
    assert "Traceback" in caplog.text and "bug in the coordinator" in caplog.text


def test_empty_messages_is_still_a_400(fake_llm, fugu_server):
    fake = fake_llm([])
    status, raw = chat(fugu_server(), {"messages": []})
    assert (status, raw) == (400, b'{"error": "messages required"}')
    assert fake.calls == []


def test_get_endpoints_are_unchanged(fugu_server):
    address = fugu_server()
    assert request(address, "GET", "/health") == (200, b'{"status": "ok", "model": "fugu"}')
    assert request(address, "GET", "/v1/models") == (200, (
        b'{"object": "list", "data": [{"id": "fugu", "object": "model", "owned_by": "openfugu"}]}'))
    assert request(address, "GET", "/nope") == (404, b'{"error": "not found"}')


# ---- startup checks (these fail before the server binds) --------------------------
def test_main_fails_fast_without_a_router_model(monkeypatch, caplog):
    monkeypatch.setenv("FUGU_API_KEY", "test-key")
    assert main(["--slot-models", "openai/w0"]) == 1
    assert ("configuration error: no router model: set FUGU_ROUTER_MODEL or "
            "FUGU_CONDUCTOR_MODEL") in caplog.text


def test_main_fails_fast_on_a_worker_without_credentials(monkeypatch, caplog):
    monkeypatch.setenv("FUGU_CONDUCTOR_MODEL", ROUTER_MODEL)
    monkeypatch.setenv("FUGU_CONDUCTOR_API_KEY", "router-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["--slot-models", "anthropic/claude-test"]) == 1
    assert "worker 0 model 'anthropic/claude-test': no API key" in caplog.text


def test_main_rejects_empty_slot_entries(monkeypatch, caplog):
    monkeypatch.setenv("FUGU_API_KEY", "test-key")
    assert main(["--slot-models", "openai/w0,,openai/w1"]) == 1
    assert "configuration error: --slot-models" in caplog.text


@pytest.mark.parametrize("argv", [
    [],
    ["--slot-models", "openai/w0", "--port", "70000"],
    ["--slot-models", "openai/w0", "--max-turns", "0"],
    ["--slot-models", "openai/w0", "--model", "qwen"],
], ids=["no-slot-models", "bad-port", "zero-turns", "removed-model-flag"])
def test_main_rejects_bad_or_removed_options(argv):
    with pytest.raises(SystemExit) as info:
        main(argv)
    assert info.value.code == 2
