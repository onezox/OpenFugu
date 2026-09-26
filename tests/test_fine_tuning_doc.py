"""docs/FINE_TUNING_FORMAT.md must show exactly what the code sends and accepts.

Every block that follows a `<!-- check: NAME -->` marker in the document is
rebuilt here by the real code (the Coordinator and ApiRouter over litellm's
mock_response, conductor_prompt and ConductorExecutor) from the example
scenario below. A prompt change fails this test until the document shows the
new text.
"""
import json
import re
from pathlib import Path

import pytest

from openfugu import mini, ultra
from openfugu.llm import Endpoint
from openfugu.mini import ROUTER_CORRECTION_PROMPT, ApiRouter, Coordinator, RouteError, parse_route
from openfugu.ultra import (LABELS, ConductorExecutor, conductor_prompt, parse_workflow,
                            validate_workflow)

DOC = Path(__file__).resolve().parents[1] / "docs" / "FINE_TUNING_FORMAT.md"
POOL = ["openai/model-a", "anthropic/model-b", "gemini/model-c"]
ROUTER = Endpoint("openai/fugu-router", api_key="test-key")

# One coordination run: the router's label and the worker's reply for each turn.
EPISODE_QUERY = "How many times does the letter r appear in the word strawberry?"
EPISODE = [
    ((0, "solver"), "<think>strawberry has one r in 'straw' and one in 'berry'.</think>"
                    "The letter r appears 2 times."),
    ((2, "verifier"), "REJECT - 'berry' has two r's, so the count is wrong."),
    ((2, "thinker"), "<suggestion>Spell the word letter by letter and count every r."
                     "</suggestion>\n\n<suggested_role>solver</suggested_role>"),
    ((1, "solver"), "<think>s-t-r-a-w-b-e-r-r-y: the r's are letters 3, 8 and 9.</think>"
                    "The letter r appears 3 times."),
    ((2, "verifier"), "ACCEPT - the letters are counted correctly."),
]
# An invalid first reply on turn i of the episode; the retry answers with that turn's label.
INVALID_REPLIES = [
    '{"agent_id": 3, "role": "solver"}',
    '{"agent_id": "2", "role": "verifier"}',
    '{"agent_id": 2, "role": "critic"}',
    "Agent 1 should write a new answer.",
]
# (query, model_id, subtasks, access_list) of each Conductor example.
CONDUCTOR_EXAMPLES = [
    ("Write a Python function that checks whether a string is a palindrome, and test it.",
     [1, 0, 2],
     ["Write a Python function is_palindrome(text) that returns True when text reads the "
      "same forwards and backwards, ignoring case, spaces and punctuation.",
      "Write pytest tests for the is_palindrome(text) function above, covering an empty "
      "string, mixed case and punctuation.",
      "Review the is_palindrome function and the tests above. Fix any bug and return the "
      "final function and tests as one Python file."],
     [[], [0], [0, 1]]),
    ("What is the capital of Australia?",
     [0],
     ["What is the capital of Australia? Answer in one sentence."],
     [[]]),
]


def label(agent_id: int, role: str) -> str:
    return json.dumps({"agent_id": agent_id, "role": role})


def jsonl_line(messages: list[dict], target: str) -> str:
    return json.dumps({"messages": [*messages, {"role": "assistant", "content": target}]},
                      ensure_ascii=False)


def conductor_completion(model_ids: list, subtasks: list, access_list: list) -> str:
    return (f"model_id: {json.dumps(model_ids)}\nsubtasks: {json.dumps(subtasks)}\n"
            f"access_list: {json.dumps(access_list)}")


def episode_router_calls(fake_llm) -> list[list[dict]]:
    """Run the episode through the Coordinator; return the messages of every router call."""
    fake = fake_llm([label(*decision) for decision, _ in EPISODE])
    replies = iter(reply for _, reply in EPISODE)
    result = Coordinator(ApiRouter(ROUTER, POOL),
                         lambda role, messages, agent_id: next(replies)).run(EPISODE_QUERY)
    assert result.terminated_by == "verifier_accept"
    assert len(fake.calls) == len(EPISODE)          # every label was accepted first time
    return [call["messages"] for call in fake.calls]


def retry_line(fake_llm, messages: list[dict], invalid: str, target: str) -> str:
    """Route once with an invalid first reply; return the retry call plus its target."""
    fake = fake_llm([invalid, target])
    ApiRouter(ROUTER, POOL).route(messages)
    return jsonl_line(fake.calls[1]["messages"], target)


def worker_prompt_of_last_step(model_ids: list, subtasks: list, access_list: list) -> str:
    """The user message the last step's worker receives when step j outputs "(output of step j)"."""
    prompts = []

    def worker(subtask, messages, agent_id):
        prompts.append(messages[0]["content"])
        return f"(output of step {len(prompts) - 1})"
    ConductorExecutor(worker).execute(validate_workflow(model_ids, subtasks, access_list,
                                                        pool_size=len(POOL)))
    return prompts[-1]


def expected_blocks(fake_llm) -> dict[str, str]:
    """Every checked block of the document, rendered by the code."""
    calls = episode_router_calls(fake_llm)
    labels = [label(*decision) for decision, _ in EPISODE]
    return {
        "router-system-prompt": calls[0][0]["content"],
        "router-user-message": calls[-1][1]["content"],
        "router-jsonl": "\n".join(map(jsonl_line, calls, labels)),
        "router-correction-prompt": ROUTER_CORRECTION_PROMPT,
        "router-retry-jsonl": "\n".join(
            retry_line(fake_llm, messages, invalid, target)
            for messages, invalid, target in zip(calls, INVALID_REPLIES, labels)),
        "conductor-system-prompt": conductor_prompt(CONDUCTOR_EXAMPLES[0][0], POOL)[0]["content"],
        "conductor-jsonl": "\n".join(
            jsonl_line(conductor_prompt(query, POOL), conductor_completion(*lists))
            for query, *lists in CONDUCTOR_EXAMPLES),
        "conductor-worker-prompt": worker_prompt_of_last_step(*CONDUCTOR_EXAMPLES[0][1:]),
    }


def call_settings() -> list[list[str]]:
    """Temperature, max output tokens, timeout and attempts of each model call."""
    return [["Router", str(mini.ROUTER_TEMPERATURE), str(mini.ROUTER_MAX_TOKENS),
             f"{mini.ROUTER_TIMEOUT_S:g} s", str(mini.ROUTER_ATTEMPTS)],
            ["Conductor", str(ultra.CONDUCTOR_TEMPERATURE), str(ultra.CONDUCTOR_MAX_TOKENS),
             f"{ultra.CONDUCTOR_TIMEOUT_S:g} s", str(ultra.CONDUCTOR_ATTEMPTS)]]


def route_rejections() -> list[tuple[str, str]]:
    """(reply, reason) for each invalid reply, as the router's parser reports it."""
    rows = []
    for reply in INVALID_REPLIES:
        with pytest.raises(RouteError) as info:
            parse_route(reply, range(len(POOL)))
        rows.append((reply, str(info.value)))
    return rows


# ---- reading the document -------------------------------------------------------
_MARKER = re.compile(r"^<!-- check: ([a-z-]+) -->\n", re.M)
_FENCE = re.compile(r"```[a-z]*\n(.*?)\n```", re.S)


def doc_sections() -> dict[str, str]:
    """The text after each check marker: a fenced block's content or a table's rows."""
    text = DOC.read_text(encoding="utf-8")
    sections = {}
    for marker in _MARKER.finditer(text):
        rest = text[marker.end():]
        fence = _FENCE.match(rest)
        if fence:
            sections[marker.group(1)] = fence.group(1)
        else:
            rows = re.match(r"(?:\|.*\n)+", rest)
            sections[marker.group(1)] = rows.group(0) if rows else ""
    return sections


def table_rows(section: str) -> list[list[str]]:
    """Cells of a markdown table's body rows, with code spans unwrapped."""
    lines = section.strip().splitlines()[2:]         # skip the header and |---| rows
    return [[cell.strip().strip("`") for cell in line.strip("|").split("|")] for line in lines]


BLOCKS = ["router-system-prompt", "router-user-message", "router-jsonl",
          "router-correction-prompt", "router-retry-jsonl", "conductor-system-prompt",
          "conductor-jsonl", "conductor-worker-prompt"]
TABLES = ["call-settings", "route-rejections", "conductor-labels"]


def test_document_has_exactly_the_checked_sections():
    assert sorted(doc_sections()) == sorted(BLOCKS + TABLES)


@pytest.mark.parametrize("name", BLOCKS)
def test_block_matches_the_code(fake_llm, name):
    expected = expected_blocks(fake_llm)[name]
    assert doc_sections()[name] == expected, (
        f"docs/FINE_TUNING_FORMAT.md block {name!r} is out of date; the code renders:\n"
        f"{expected}")


def test_call_settings_match_the_code():
    assert table_rows(doc_sections()["call-settings"]) == call_settings()


def test_route_rejection_reasons_match_the_parser():
    assert [tuple(row) for row in table_rows(doc_sections()["route-rejections"])] == (
        route_rejections())


def test_conductor_label_spellings_match_the_parser():
    rows = table_rows(doc_sections()["conductor-labels"])
    documented = {name: tuple(alias.strip().strip("`") for alias in aliases.split(","))
                  for name, aliases in rows}
    assert documented == LABELS


@pytest.mark.parametrize("query, model_ids, subtasks, access_list", CONDUCTOR_EXAMPLES)
def test_conductor_targets_are_valid_workflows(query, model_ids, subtasks, access_list):
    completion = conductor_completion(model_ids, subtasks, access_list)
    workflow = validate_workflow(*parse_workflow(completion), pool_size=len(POOL))
    assert (workflow.model_ids, workflow.subtasks) == (model_ids, subtasks)
