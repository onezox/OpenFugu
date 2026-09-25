"""Conductor (ultra.py): parsing, validation, execution and the prompt."""
import pytest

from openfugu.ultra import (ConductorExecutor, WorkflowError, conductor_prompt,
                            parse_workflow, validate_workflow)

SLOTS = "openai/w0,openai/w1,openai/w2"

# The canned workflow of the former `ultra.py --self-test`.
CANNED = (
    "Plan: derive then implement then verify.\n"
    "model_id: [2, 0, 1]\n"
    'subtasks: ["Devise an algorithm for the task", '
    '"Implement it in Python using the devised algorithm", '
    '"Verify the implementation is correct"]\n'
    "access_list: [[], [0], [0, 1]]\n"
)


def workflow_text(model_id="[0]", subtasks='["Solve it"]', access="[[]]"):
    return f"model_id: {model_id}\nsubtasks: {subtasks}\naccess_list: {access}\n"


def parse_and_validate(text, pool_size=3):
    return validate_workflow(*parse_workflow(text), pool_size=pool_size)


class RecordingWorker:
    """Test double for the worker pool: records each step's prompt."""

    def __init__(self):
        self.calls = []

    def __call__(self, subtask, messages, agent_id):
        self.calls.append((agent_id, messages[0]["content"]))
        return f"output of step {len(self.calls) - 1}"


# ---- parsing -------------------------------------------------------------------
def test_canned_workflow_parses_validates_and_executes():
    workflow = parse_and_validate(CANNED)
    assert workflow.model_ids == [2, 0, 1]
    assert workflow.access_list == [[], [0], [0, 1]]
    assert workflow.sees == [[], [0], [0, 1]]

    worker = RecordingWorker()
    result = ConductorExecutor(worker).execute(workflow)
    assert [step.agent_id for step in result.steps] == [2, 0, 1]
    assert result.final == "output of step 2"
    assert worker.calls[0] == (2, "Your subtask: Devise an algorithm for the task")
    assert worker.calls[2] == (1, (
        "USER QUESTION context:\n"
        "\n<Subtask assigned to Agent 2>Devise an algorithm for the task"
        "</Subtask assigned to Agent 2>"
        "\n<Agent 2 response>output of step 0</Agent 2 response>"
        "\n<Subtask assigned to Agent 0>Implement it in Python using the devised algorithm"
        "</Subtask assigned to Agent 0>"
        "\n<Agent 0 response>output of step 1</Agent 0 response>"
        "\n\nYour subtask: Verify the implementation is correct"))


def test_repo_bug_all_inside_a_list_sees_every_earlier_step():
    """results/conductor_e2e_run.txt ran [[], [], ['all']] with step 2 seeing nothing."""
    workflow = parse_and_validate(workflow_text("[0, 0, 0]", '["a", "b", "c"]',
                                                "[[], [], ['all']]"))
    assert workflow.sees == [[], [], [0, 1]]


@pytest.mark.parametrize("entry", ['["all"]', '"all"', '"ALL"', "['All']", '[" all "]'])
def test_all_in_any_form_and_case_sees_every_earlier_step(entry):
    workflow = parse_and_validate(workflow_text("[0, 1, 2]", '["a", "b", "c"]',
                                                f"[{entry}, [0], {entry}]"))
    assert workflow.sees == [[], [0], [0, 1]]


def test_repo_bug_labels_in_reasoning_are_ignored():
    """The old parser took the first 'model id:' anywhere and parsed ["difficulty"]."""
    text = ("The model id: depends on [difficulty], so the subtasks: [plan, code] follow.\n"
            "model_id: depends on [difficulty] as well\n"
            "access: [nothing yet]\n"
            + workflow_text("[1]", '["Write the code"]', "[[]]"))
    workflow = parse_and_validate(text)
    assert (workflow.model_ids, workflow.subtasks, workflow.sees) == ([1], ["Write the code"],
                                                                      [[]])


def test_a_label_only_in_prose_counts_as_missing():
    with pytest.raises(WorkflowError, match="^model_id: missing"):
        parse_workflow('The model id: depends on [difficulty].\nsubtasks: ["a"]\n'
                       "access_list: [[]]")


def test_last_occurrence_of_each_label_wins():
    draft = workflow_text("[0, 1]", '["draft a", "draft b"]', "[[], [0]]")
    final = workflow_text("[2]", '["final"]', "[[]]")
    workflow = parse_and_validate(f"Draft:\n{draft}\nFinal answer:\n{final}")
    assert (workflow.model_ids, workflow.subtasks) == ([2], ["final"])


def test_label_spellings_separators_and_multiline_lists():
    text = ("```python\n"
            "Model_IDs = [0, 1]\n"
            "  subtask: [\n    'Draft it',\n    \"Check it, twice\",\n]\n"
            "access list =\n[[], 'all']\n"
            "```")
    workflow = parse_and_validate(text)
    assert workflow.model_ids == [0, 1]
    assert workflow.subtasks == ["Draft it", "Check it, twice"]
    assert workflow.sees == [[], [0]]


def test_curly_quoted_lists_are_normalised():
    workflow = parse_and_validate(workflow_text("[0, 1]", "[“Plan it”, “Don’t skip tests”]",
                                                "[[], “all”]"))
    assert workflow.subtasks == ["Plan it", "Don't skip tests"]
    assert workflow.sees == [[], [0]]


def test_curly_quotes_inside_straight_quotes_are_kept():
    workflow = parse_and_validate(workflow_text(subtasks='["Print “hello”"]'))
    assert workflow.subtasks == ["Print “hello”"]


def test_json_literals_are_accepted():
    _, _, access = parse_workflow(workflow_text("[0, 1]", '["a", "b"]', "[[], null]"))
    assert access == [[], None]


def test_unquoted_items_are_rejected_instead_of_split_on_commas():
    with pytest.raises(WorkflowError, match="^subtasks = .*not a complete Python or JSON list"):
        parse_workflow(workflow_text(subtasks="[plan it, then code it]"))


def test_an_unclosed_list_is_rejected():
    with pytest.raises(WorkflowError, match="^access_list = .*not a complete Python or JSON"):
        parse_workflow('model_id: [0]\nsubtasks: ["a"]\naccess_list: [[], [0')


@pytest.mark.parametrize("missing", ["model_id", "subtasks", "access_list"])
def test_a_missing_list_is_named(missing):
    lines = {"model_id": "model_id: [0]", "subtasks": 'subtasks: ["a"]',
             "access_list": "access_list: [[]]"}
    text = "\n".join(line for name, line in lines.items() if name != missing)
    with pytest.raises(WorkflowError, match=f"^{missing}: missing"):
        parse_workflow(text)


# ---- validation ----------------------------------------------------------------
@pytest.mark.parametrize("model_id, message", [
    ('["2"]', "model_id[0] = '2': must be an integer worker index"),
    ('["gpt-5"]', "model_id[0] = 'gpt-5': must be an integer worker index"),
    ("[1.0]", "model_id[0] = 1.0: must be an integer worker index"),
    ("[1.7]", "model_id[0] = 1.7: must be an integer worker index"),
    ("[True]", "model_id[0] = True: must be an integer worker index"),
    ("[-1]", "model_id[0] = -1: must be a worker index from 0 to 2"),
    ("[3]", "model_id[0] = 3: must be a worker index from 0 to 2"),
], ids=["numeric-string", "name", "float", "fraction", "bool", "negative", "out-of-range"])
def test_invalid_model_ids(model_id, message):
    with pytest.raises(WorkflowError) as info:
        parse_and_validate(workflow_text(model_id=model_id))
    assert str(info.value) == message


@pytest.mark.parametrize("subtasks, message", [
    ('[""]', "subtasks[0] = '': must be a non-empty string"),
    ('["   "]', "subtasks[0] = '   ': must be a non-empty string"),
    ("[42]", "subtasks[0] = 42: must be a non-empty string"),
    ('[["nested"]]', "subtasks[0] = ['nested']: must be a non-empty string"),
], ids=["empty", "blank", "number", "list"])
def test_invalid_subtasks(subtasks, message):
    with pytest.raises(WorkflowError) as info:
        parse_and_validate(workflow_text(subtasks=subtasks))
    assert str(info.value) == message


ACCESS_RULE = 'must be [], "all", ["all"] or a list of indices of earlier steps'


@pytest.mark.parametrize("access, message", [
    ("[[], [2], []]",
     "access_list[1] = [2]: step 1 can only see earlier steps (step 0), not step 2"),
    ("[[], [0], [2]]",
     "access_list[2] = [2]: step 2 can only see earlier steps (steps 0 to 1), not step 2"),
    ("[[0], [], []]",
     "access_list[0] = [0]: step 0 can only see earlier steps (none), not step 0"),
    ("[[], [-1], []]",
     "access_list[1] = [-1]: step 1 can only see earlier steps (step 0), not step -1"),
    ('[[], [], ["all", 0]]', f"access_list[2] = ['all', 0]: {ACCESS_RULE}"),
    ("[[], 0, []]", f"access_list[1] = 0: {ACCESS_RULE}"),
    ('[[], ["0"], []]', f"access_list[1] = ['0']: {ACCESS_RULE}"),
    ("[[], [0.0], []]", f"access_list[1] = [0.0]: {ACCESS_RULE}"),
    ("[[], None, []]", f"access_list[1] = None: {ACCESS_RULE}"),
    ('[[], "none", []]', f"access_list[1] = 'none': {ACCESS_RULE}"),
], ids=["forward", "self", "self-at-step-0", "negative", "all-mixed", "bare-int",
        "string-index", "float-index", "null", "other-string"])
def test_invalid_access_entries(access, message):
    with pytest.raises(WorkflowError) as info:
        parse_and_validate(workflow_text("[0, 1, 2]", '["a", "b", "c"]', access))
    assert str(info.value) == message


def test_duplicate_access_indices_are_merged():
    workflow = parse_and_validate(workflow_text("[0, 1, 2]", '["a", "b", "c"]',
                                                "[[], [0, 0], [1, 0, 1]]"))
    assert workflow.sees == [[], [0], [0, 1]]


def test_workflow_error_names_the_list_index_value_and_rule():
    with pytest.raises(WorkflowError) as info:
        parse_and_validate(workflow_text("[0, 1]", '["a", "b"]', "[[], [1]]"))
    error = info.value
    assert (error.field, error.index, error.value) == ("access_list", 1, [1])
    assert error.rule == "step 1 can only see earlier steps (step 0), not step 1"


def test_unequal_lengths_are_rejected():
    with pytest.raises(WorkflowError) as info:
        parse_and_validate(workflow_text("[0, 1]", '["a"]', "[[], [0]]"))
    assert str(info.value) == ("workflow: model_id, subtasks and access_list have 2, 1 and 2 "
                               "entries; all three lists must have the same length")


def test_six_steps_is_an_error_not_a_silent_cut():
    text = workflow_text(str([0] * 6), str(["step"] * 6), str([[]] * 6))
    with pytest.raises(WorkflowError) as info:
        parse_and_validate(text)
    assert str(info.value) == "workflow: has 6 steps; a workflow must have 1 to 5 steps"


def test_an_empty_workflow_is_rejected():
    with pytest.raises(WorkflowError, match="has 0 steps"):
        parse_and_validate(workflow_text("[]", "[]", "[]"))


def test_five_steps_are_allowed():
    workflow = parse_and_validate(workflow_text(str([0] * 5), str(["step"] * 5),
                                                str(["all"] * 5)))
    assert workflow.sees == [[], [0], [0, 1], [0, 1, 2], [0, 1, 2, 3]]


# ---- the Conductor prompt --------------------------------------------------------
def test_conductor_prompt_lists_the_real_pool_and_is_exact():
    system, user = conductor_prompt("Sum 1..10", SLOTS.split(","))
    assert user == {"role": "user", "content": "USER QUESTION: Sum 1..10"}
    assert system == {"role": "system", "content": (
        "You are a Conductor that orchestrates a pool of worker LLMs to solve a task. "
        "Design an agentic workflow as THREE equal-length Python lists:\n"
        "  model_id   = [int, ...]   # which worker (0-indexed) runs each step\n"
        "  subtasks   = [str, ...]   # the natural-language instruction for each step\n"
        "  access_list= [list, ...]  # for each step, the indices of EARLIER steps whose\n"
        "                            # outputs that step may see ([] = none, \"all\" = every "
        "earlier step)\n"
        "Rules: lists must be equal length (<=5 steps); every model_id must be one of the "
        "worker indices listed below; access_list may only reference strictly earlier steps "
        "(it is a DAG executed in order); workers never see the user question, only their "
        "own subtask and the outputs they may access, so every subtask must be "
        "self-contained; the LAST step's output is the final answer. Pick workers to match "
        "each subtask's demands.\n\n"
        "AVAILABLE LANGUAGE MODELS:\n"
        "  0: openai/w0\n"
        "  1: openai/w1\n"
        "  2: openai/w2\n\n"
        "Output the three lists explicitly as 'model_id: [...]', 'subtasks: [...]', "
        "'access_list: [...]', each on its own line. You may reason first, but the three "
        "lists must appear.")}
