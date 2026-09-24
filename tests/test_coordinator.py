"""Coordinator: what the router sees each turn, including the <state> line."""
import json

import pytest

from openfugu.llm import Endpoint
from openfugu.mini import ApiRouter, Coordinator

POOL = ["openai/alpha", "openai/beta"]
ENDPOINT = Endpoint("openai/router", api_key="test-key")
SOLVER_REPLY = "<think>2+2 is 4</think>The answer is 4."


class ScriptedWorker:
    """Test double for the worker pool: verifier verdicts come from a script."""

    def __init__(self, verdicts=()):
        self.verdicts = iter(verdicts)
        self.calls = []

    def __call__(self, role_name, messages, agent_id):
        self.calls.append((role_name, agent_id))
        if role_name == "Verifier":
            return next(self.verdicts)
        if role_name == "Thinker":
            return "<suggestion>double-check the arithmetic</suggestion>"
        return SOLVER_REPLY


def script_router(fake_llm, decisions):
    return fake_llm([json.dumps({"agent_id": a, "role": r}) for a, r in decisions])


def router_inputs(fake):
    return [call["messages"][1]["content"] for call in fake.calls]


def test_state_line_changes_after_every_turn(fake_llm):
    fake = script_router(fake_llm, [(0, "solver"), (1, "verifier"), (1, "thinker"),
                                    (0, "solver"), (1, "verifier")])
    worker = ScriptedWorker(["REJECT - the proof is missing", "ACCEPT - correct"])
    result = Coordinator(ApiRouter(ENDPOINT, POOL), worker).run("What is 2+2?")

    assert result.terminated_by == "verifier_accept"
    assert result.final == SOLVER_REPLY
    thought0 = "\n<reference_thought_0>2+2 is 4</reference_thought_0>"
    thought1 = "\n<reference_thought_1>2+2 is 4</reference_thought_1>"
    assert router_inputs(fake) == [
        "What is 2+2?\n<state>turn=0 last_role=none last_verdict=none</state>",
        f"What is 2+2?{thought0}\n<state>turn=1 last_role=solver last_verdict=none</state>",
        f"What is 2+2?{thought0}\n<state>turn=2 last_role=verifier last_verdict=REJECT</state>",
        f"What is 2+2?{thought0}\n<state>turn=3 last_role=thinker last_verdict=REJECT</state>",
        f"What is 2+2?{thought0}{thought1}\n<state>turn=4 last_role=solver last_verdict=none</state>",
    ]
    assert worker.calls == [("Worker", 0), ("Verifier", 1), ("Thinker", 1), ("Worker", 0),
                            ("Verifier", 1)]


def test_router_system_prompt_is_exact(fake_llm):
    fake = script_router(fake_llm, [(0, "solver")])
    Coordinator(ApiRouter(ENDPOINT, POOL), ScriptedWorker(), max_turns=1).run("Q")
    assert fake.calls[0]["messages"][0] == {"role": "system", "content": (
        "You are a message dispatcher that coordinates a pool of agents to solve a problem. "
        "Read the problem, the reference thoughts from earlier solver turns and the <state> "
        "line, then decide which agent acts next and in which role.\n\n"
        "Agents (agent_id: model):\n"
        "0: openai/alpha\n"
        "1: openai/beta\n\n"
        "Roles:\n"
        "- solver: writes a complete answer to the problem.\n"
        "- thinker: analyses the problem and the current answer, then suggests how to "
        "improve it.\n"
        "- verifier: reviews the current answer and replies ACCEPT or REJECT. An ACCEPT "
        "ends the run.\n\n"
        "The <state> line gives the turn number (starting at 0), the role that acted on the "
        "previous turn, and the verifier's verdict on the current answer (none if it has not "
        "been reviewed since it was written).\n\n"
        "Reply with only this JSON object and nothing else:\n"
        '{"agent_id": <an agent_id from the list above>, "role": "<solver|thinker|verifier>"}')}


def test_roster_lists_only_the_agents_the_mask_offers(fake_llm):
    fake = script_router(fake_llm, [(1, "solver")])
    Coordinator(ApiRouter(ENDPOINT, POOL), ScriptedWorker(), max_turns=1,
                agent_mask=[False, True]).run("Q")
    assert "Agents (agent_id: model):\n1: openai/beta\n\nRoles:" in (
        fake.calls[0]["messages"][0]["content"])


def test_cold_verifier_runs_as_solver_and_state_reports_solver(fake_llm):
    fake = script_router(fake_llm, [(0, "verifier"), (0, "verifier")])
    worker = ScriptedWorker(["ACCEPT"])
    result = Coordinator(ApiRouter(ENDPOINT, POOL), worker).run("Q")
    assert worker.calls == [("Worker", 0), ("Verifier", 0)]
    assert router_inputs(fake)[1].endswith(
        "<state>turn=1 last_role=solver last_verdict=none</state>")
    assert result.terminated_by == "verifier_accept"


@pytest.mark.parametrize("mask, message", [
    ([False, False], "allows no agents"),
    ([True], "has 1 entries but the pool has 2 agents"),
])
def test_invalid_agent_mask_fails_before_any_call(fake_llm, mask, message):
    fake = fake_llm([])
    with pytest.raises(ValueError, match=message):
        Coordinator(ApiRouter(ENDPOINT, POOL), ScriptedWorker(), agent_mask=mask)
    assert fake.calls == []
