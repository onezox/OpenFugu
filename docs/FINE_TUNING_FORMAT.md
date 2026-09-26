# Fine-tuning data format

This document specifies the training data for the two models that OpenFugu
asks for decisions:

- the **router** of `python -m openfugu.serve`, which picks the agent and the
  role for each turn;
- the **Conductor** of `python -m openfugu.ultra`, which writes a whole
  workflow in one reply.

The worker models are not fine-tuned; they only follow the prompts they are
given.

Every prompt and example below was produced by the code in `openfugu/`: a real
`Coordinator` run for the router messages, `ApiRouter`'s retry path for the
correction examples, and `conductor_prompt` and `ConductorExecutor` for the
Conductor. `tests/test_fine_tuning_doc.py` rebuilds each of them and fails when
this document no longer matches the code.

> [!WARNING]
> **Freeze the slot order.** An agent id is a position in `--slot-models`: id 0
> is the first entry, id 1 the second, and so on. A fine-tuned router or
> Conductor learns what each id is good at. After fine-tuning, never change the
> order of `--slot-models`: a reordered list silently sends every decision to a
> different model, and nothing detects it. Generate the training data with the
> exact `--slot-models` list you will run with, and treat any other change to
> the pool (adding, removing or replacing a model) as a reason to generate new
> data and fine-tune again.

## File format

Both datasets are UTF-8 JSONL files with one training example per line. Each
line is a JSON object with a single `messages` list: the exact messages the
runtime sends to the model, followed by the target reply as the last
`assistant` message. Every message has only `role` and `content`, and
`content` is always a string. This is the usual chat fine-tuning layout (the
one OpenAI's chat fine-tuning uses, for example).

The examples use this three-slot pool:

```text
--slot-models "openai/model-a,anthropic/model-b,gemini/model-c"
```

## Runtime settings

How each model is called at run time:

<!-- check: call-settings -->
| Model | Temperature | Max output tokens | Timeout per call | Attempts |
|---|---|---|---|---|
| Router | 0.0 | 256 | 30 s | 3 |
| Conductor | 0.2 | 2048 | 120 s | 3 |

For the router, the attempts are the calls per turn, including correction
retries. For the Conductor, they are the calls made when the API fails with a
transient error; an invalid workflow is never retried.

## Router

### Input

Every turn, the router receives two messages.

**System message.** A fixed prompt with a roster of the agents it may choose:
one line per agent, `<agent_id>: <model id>`, in slot order. For the example
pool:

<!-- check: router-system-prompt -->
```text
You are a message dispatcher that coordinates a pool of agents to solve a problem. Read the problem, the reference thoughts from earlier solver turns and the <state> line, then decide which agent acts next and in which role.

Agents (agent_id: model):
0: openai/model-a
1: anthropic/model-b
2: gemini/model-c

Roles:
- solver: writes a complete answer to the problem.
- thinker: analyses the problem and the current answer, then suggests how to improve it.
- verifier: reviews the current answer and replies ACCEPT or REJECT. An ACCEPT ends the run.

The <state> line gives the turn number (starting at 0), the role that acted on the previous turn, and the verifier's verdict on the current answer (none if it has not been reviewed since it was written).

Reply with only this JSON object and nothing else:
{"agent_id": <an agent_id from the list above>, "role": "<solver|thinker|verifier>"}
```

**User message.** The observation, a line break, and the state line.

- The observation starts with the user's query: the text of the last user
  message sent to the server. After each solver turn, the Coordinator appends
  that solver's thought on a new line as
  `<reference_thought_N>…</reference_thought_N>`, with N counting from 0. The
  thought is the text inside the first `<think>…</think>` block of the
  solver's reply or, when the reply has no such block, the whole reply without
  stray `<think>` or `</think>` tags. Surrounding whitespace is removed, and an
  empty thought adds nothing. Thinker and verifier replies never enter the
  observation.
- The state line is `<state>turn=T last_role=R last_verdict=V</state>`:
  - `T` is the turn number, from 0.
  - `R` is `none` on turn 0, and afterwards the role that ran on the previous
    turn: `solver`, `thinker` or `verifier`. This is the role that actually
    ran, which can differ from the router's choice (labelling rules 3 and 5).
  - `V` is `REJECT` when a verifier rejected the current answer and no solver
    has written a new one since, and `none` otherwise. `ACCEPT` never appears,
    because an accepting verifier ends the run.

The user message of turn 4 in the example episode below:

<!-- check: router-user-message -->
```text
How many times does the letter r appear in the word strawberry?
<reference_thought_0>strawberry has one r in 'straw' and one in 'berry'.</reference_thought_0>
<reference_thought_1>s-t-r-a-w-b-e-r-r-y: the r's are letters 3, 8 and 9.</reference_thought_1>
<state>turn=4 last_role=solver last_verdict=none</state>
```

### Target

One JSON object, written exactly as Python's `json.dumps` writes it, for
example `{"agent_id": 1, "role": "solver"}`.

### Example: one episode

The five router calls of one run, each with its label as the target:

<!-- check: router-jsonl -->
```jsonl
{"messages": [{"role": "system", "content": "You are a message dispatcher that coordinates a pool of agents to solve a problem. Read the problem, the reference thoughts from earlier solver turns and the <state> line, then decide which agent acts next and in which role.\n\nAgents (agent_id: model):\n0: openai/model-a\n1: anthropic/model-b\n2: gemini/model-c\n\nRoles:\n- solver: writes a complete answer to the problem.\n- thinker: analyses the problem and the current answer, then suggests how to improve it.\n- verifier: reviews the current answer and replies ACCEPT or REJECT. An ACCEPT ends the run.\n\nThe <state> line gives the turn number (starting at 0), the role that acted on the previous turn, and the verifier's verdict on the current answer (none if it has not been reviewed since it was written).\n\nReply with only this JSON object and nothing else:\n{\"agent_id\": <an agent_id from the list above>, \"role\": \"<solver|thinker|verifier>\"}"}, {"role": "user", "content": "How many times does the letter r appear in the word strawberry?\n<state>turn=0 last_role=none last_verdict=none</state>"}, {"role": "assistant", "content": "{\"agent_id\": 0, \"role\": \"solver\"}"}]}
{"messages": [{"role": "system", "content": "You are a message dispatcher that coordinates a pool of agents to solve a problem. Read the problem, the reference thoughts from earlier solver turns and the <state> line, then decide which agent acts next and in which role.\n\nAgents (agent_id: model):\n0: openai/model-a\n1: anthropic/model-b\n2: gemini/model-c\n\nRoles:\n- solver: writes a complete answer to the problem.\n- thinker: analyses the problem and the current answer, then suggests how to improve it.\n- verifier: reviews the current answer and replies ACCEPT or REJECT. An ACCEPT ends the run.\n\nThe <state> line gives the turn number (starting at 0), the role that acted on the previous turn, and the verifier's verdict on the current answer (none if it has not been reviewed since it was written).\n\nReply with only this JSON object and nothing else:\n{\"agent_id\": <an agent_id from the list above>, \"role\": \"<solver|thinker|verifier>\"}"}, {"role": "user", "content": "How many times does the letter r appear in the word strawberry?\n<reference_thought_0>strawberry has one r in 'straw' and one in 'berry'.</reference_thought_0>\n<state>turn=1 last_role=solver last_verdict=none</state>"}, {"role": "assistant", "content": "{\"agent_id\": 2, \"role\": \"verifier\"}"}]}
{"messages": [{"role": "system", "content": "You are a message dispatcher that coordinates a pool of agents to solve a problem. Read the problem, the reference thoughts from earlier solver turns and the <state> line, then decide which agent acts next and in which role.\n\nAgents (agent_id: model):\n0: openai/model-a\n1: anthropic/model-b\n2: gemini/model-c\n\nRoles:\n- solver: writes a complete answer to the problem.\n- thinker: analyses the problem and the current answer, then suggests how to improve it.\n- verifier: reviews the current answer and replies ACCEPT or REJECT. An ACCEPT ends the run.\n\nThe <state> line gives the turn number (starting at 0), the role that acted on the previous turn, and the verifier's verdict on the current answer (none if it has not been reviewed since it was written).\n\nReply with only this JSON object and nothing else:\n{\"agent_id\": <an agent_id from the list above>, \"role\": \"<solver|thinker|verifier>\"}"}, {"role": "user", "content": "How many times does the letter r appear in the word strawberry?\n<reference_thought_0>strawberry has one r in 'straw' and one in 'berry'.</reference_thought_0>\n<state>turn=2 last_role=verifier last_verdict=REJECT</state>"}, {"role": "assistant", "content": "{\"agent_id\": 2, \"role\": \"thinker\"}"}]}
{"messages": [{"role": "system", "content": "You are a message dispatcher that coordinates a pool of agents to solve a problem. Read the problem, the reference thoughts from earlier solver turns and the <state> line, then decide which agent acts next and in which role.\n\nAgents (agent_id: model):\n0: openai/model-a\n1: anthropic/model-b\n2: gemini/model-c\n\nRoles:\n- solver: writes a complete answer to the problem.\n- thinker: analyses the problem and the current answer, then suggests how to improve it.\n- verifier: reviews the current answer and replies ACCEPT or REJECT. An ACCEPT ends the run.\n\nThe <state> line gives the turn number (starting at 0), the role that acted on the previous turn, and the verifier's verdict on the current answer (none if it has not been reviewed since it was written).\n\nReply with only this JSON object and nothing else:\n{\"agent_id\": <an agent_id from the list above>, \"role\": \"<solver|thinker|verifier>\"}"}, {"role": "user", "content": "How many times does the letter r appear in the word strawberry?\n<reference_thought_0>strawberry has one r in 'straw' and one in 'berry'.</reference_thought_0>\n<state>turn=3 last_role=thinker last_verdict=REJECT</state>"}, {"role": "assistant", "content": "{\"agent_id\": 1, \"role\": \"solver\"}"}]}
{"messages": [{"role": "system", "content": "You are a message dispatcher that coordinates a pool of agents to solve a problem. Read the problem, the reference thoughts from earlier solver turns and the <state> line, then decide which agent acts next and in which role.\n\nAgents (agent_id: model):\n0: openai/model-a\n1: anthropic/model-b\n2: gemini/model-c\n\nRoles:\n- solver: writes a complete answer to the problem.\n- thinker: analyses the problem and the current answer, then suggests how to improve it.\n- verifier: reviews the current answer and replies ACCEPT or REJECT. An ACCEPT ends the run.\n\nThe <state> line gives the turn number (starting at 0), the role that acted on the previous turn, and the verifier's verdict on the current answer (none if it has not been reviewed since it was written).\n\nReply with only this JSON object and nothing else:\n{\"agent_id\": <an agent_id from the list above>, \"role\": \"<solver|thinker|verifier>\"}"}, {"role": "user", "content": "How many times does the letter r appear in the word strawberry?\n<reference_thought_0>strawberry has one r in 'straw' and one in 'berry'.</reference_thought_0>\n<reference_thought_1>s-t-r-a-w-b-e-r-r-y: the r's are letters 3, 8 and 9.</reference_thought_1>\n<state>turn=4 last_role=solver last_verdict=none</state>"}, {"role": "assistant", "content": "{\"agent_id\": 2, \"role\": \"verifier\"}"}]}
```

| Turn | Label | What ran |
|---|---|---|
| 0 | agent 0, solver | Answers 2. Its thought joins the observation. |
| 1 | agent 2, verifier | REJECT. |
| 2 | agent 2, thinker | Suggests spelling the word out, with `solver` as the next role. |
| 3 | agent 1, solver | Answers 3, with the thinker's suggestion added to its prompt. |
| 4 | agent 2, verifier | ACCEPT. The run ends and returns the turn 3 answer. |

### Labelling rules

Make one training example per router call: every turn of a run, not only the
first, each with the exact messages that turn produced.

The target is the JSON object alone, with no prose, code fence or `<think>`
block around it. The parser tolerates those (see Parser rules), but they cost
output tokens and are not what the model should learn.

The parser enforces:

1. `agent_id` is a JSON integer that appears in the roster of the example's
   system message. It is the slot position, not the line number in the roster.
   The two differ only when the `Coordinator` is given an agent mask (library
   use only: `python -m openfugu.serve` always offers every slot); a mask
   removes slots from the roster but keeps their ids.
2. `role` is `solver`, `thinker` or `verifier`, in lowercase.

How the Coordinator uses the decision:

3. Before the first solver turn, a verifier has no answer to review, so the
   Coordinator runs a verifier choice as a solver, and the next state line says
   `last_role=solver`. Label `solver` or `thinker` on turn 0 and after turns
   that were all thinkers.
4. After `last_verdict=REJECT`, the current answer is the one just rejected,
   and a verifier would review it again. Label `solver` for a new answer, or
   `thinker` for a suggestion that the next turn receives.
5. A thinker may end its reply with `<suggested_role>…</suggested_role>`. The
   Coordinator then runs the next turn in that role, with the agent the router
   chose; the router never sees the suggestion. Label the agent and role you
   want regardless, as turn 3 of the example does.
6. A verifier's ACCEPT ends the run and returns the latest solver answer.
   Without one, the run stops after the turn budget (`--max-turns`, default 5),
   which the router is not told, and returns the latest solver answer
   unreviewed. Label `verifier` as soon as the latest answer looks right.
7. Keep `--slot-models` identical between generating the data and serving
   (see the warning at the top).

### Correction retries

When a reply breaks a parser rule, the router is called again, within the
attempts in Runtime settings. The retry sends the original two messages, the
invalid reply as an `assistant` message, and a `user` message with this
correction prompt, where `{reason}` is the rejection reason:

<!-- check: router-correction-prompt -->
```text
Your reply was invalid: {reason}. Reply with only the JSON object {"agent_id": <id>, "role": "<solver|thinker|verifier>"} using an agent_id from the list.
```

Only the latest invalid reply is included; the conversation does not grow
with each attempt. Transient API errors (timeouts, rate limits, server errors)
are retried with the same messages after a short backoff and use up the same
attempts. When the attempts run out, or the API fails with a permanent error,
the router falls back to the first agent in the roster as solver and logs a
warning.

The parser gives one of these four reasons:

<!-- check: route-rejections -->
| Invalid reply | Reason |
|---|---|
| `{"agent_id": 3, "role": "solver"}` | `agent_id 3 is not in the list` |
| `{"agent_id": "2", "role": "verifier"}` | `agent_id must be an integer` |
| `{"agent_id": 2, "role": "critic"}` | `role must be one of solver, thinker, verifier` |
| `Agent 1 should write a new answer.` | `no JSON object with "agent_id" and "role" found` |

One retry example for each reason, on turns 0 to 3 of the episode:

<!-- check: router-retry-jsonl -->
```jsonl
{"messages": [{"role": "system", "content": "You are a message dispatcher that coordinates a pool of agents to solve a problem. Read the problem, the reference thoughts from earlier solver turns and the <state> line, then decide which agent acts next and in which role.\n\nAgents (agent_id: model):\n0: openai/model-a\n1: anthropic/model-b\n2: gemini/model-c\n\nRoles:\n- solver: writes a complete answer to the problem.\n- thinker: analyses the problem and the current answer, then suggests how to improve it.\n- verifier: reviews the current answer and replies ACCEPT or REJECT. An ACCEPT ends the run.\n\nThe <state> line gives the turn number (starting at 0), the role that acted on the previous turn, and the verifier's verdict on the current answer (none if it has not been reviewed since it was written).\n\nReply with only this JSON object and nothing else:\n{\"agent_id\": <an agent_id from the list above>, \"role\": \"<solver|thinker|verifier>\"}"}, {"role": "user", "content": "How many times does the letter r appear in the word strawberry?\n<state>turn=0 last_role=none last_verdict=none</state>"}, {"role": "assistant", "content": "{\"agent_id\": 3, \"role\": \"solver\"}"}, {"role": "user", "content": "Your reply was invalid: agent_id 3 is not in the list. Reply with only the JSON object {\"agent_id\": <id>, \"role\": \"<solver|thinker|verifier>\"} using an agent_id from the list."}, {"role": "assistant", "content": "{\"agent_id\": 0, \"role\": \"solver\"}"}]}
{"messages": [{"role": "system", "content": "You are a message dispatcher that coordinates a pool of agents to solve a problem. Read the problem, the reference thoughts from earlier solver turns and the <state> line, then decide which agent acts next and in which role.\n\nAgents (agent_id: model):\n0: openai/model-a\n1: anthropic/model-b\n2: gemini/model-c\n\nRoles:\n- solver: writes a complete answer to the problem.\n- thinker: analyses the problem and the current answer, then suggests how to improve it.\n- verifier: reviews the current answer and replies ACCEPT or REJECT. An ACCEPT ends the run.\n\nThe <state> line gives the turn number (starting at 0), the role that acted on the previous turn, and the verifier's verdict on the current answer (none if it has not been reviewed since it was written).\n\nReply with only this JSON object and nothing else:\n{\"agent_id\": <an agent_id from the list above>, \"role\": \"<solver|thinker|verifier>\"}"}, {"role": "user", "content": "How many times does the letter r appear in the word strawberry?\n<reference_thought_0>strawberry has one r in 'straw' and one in 'berry'.</reference_thought_0>\n<state>turn=1 last_role=solver last_verdict=none</state>"}, {"role": "assistant", "content": "{\"agent_id\": \"2\", \"role\": \"verifier\"}"}, {"role": "user", "content": "Your reply was invalid: agent_id must be an integer. Reply with only the JSON object {\"agent_id\": <id>, \"role\": \"<solver|thinker|verifier>\"} using an agent_id from the list."}, {"role": "assistant", "content": "{\"agent_id\": 2, \"role\": \"verifier\"}"}]}
{"messages": [{"role": "system", "content": "You are a message dispatcher that coordinates a pool of agents to solve a problem. Read the problem, the reference thoughts from earlier solver turns and the <state> line, then decide which agent acts next and in which role.\n\nAgents (agent_id: model):\n0: openai/model-a\n1: anthropic/model-b\n2: gemini/model-c\n\nRoles:\n- solver: writes a complete answer to the problem.\n- thinker: analyses the problem and the current answer, then suggests how to improve it.\n- verifier: reviews the current answer and replies ACCEPT or REJECT. An ACCEPT ends the run.\n\nThe <state> line gives the turn number (starting at 0), the role that acted on the previous turn, and the verifier's verdict on the current answer (none if it has not been reviewed since it was written).\n\nReply with only this JSON object and nothing else:\n{\"agent_id\": <an agent_id from the list above>, \"role\": \"<solver|thinker|verifier>\"}"}, {"role": "user", "content": "How many times does the letter r appear in the word strawberry?\n<reference_thought_0>strawberry has one r in 'straw' and one in 'berry'.</reference_thought_0>\n<state>turn=2 last_role=verifier last_verdict=REJECT</state>"}, {"role": "assistant", "content": "{\"agent_id\": 2, \"role\": \"critic\"}"}, {"role": "user", "content": "Your reply was invalid: role must be one of solver, thinker, verifier. Reply with only the JSON object {\"agent_id\": <id>, \"role\": \"<solver|thinker|verifier>\"} using an agent_id from the list."}, {"role": "assistant", "content": "{\"agent_id\": 2, \"role\": \"thinker\"}"}]}
{"messages": [{"role": "system", "content": "You are a message dispatcher that coordinates a pool of agents to solve a problem. Read the problem, the reference thoughts from earlier solver turns and the <state> line, then decide which agent acts next and in which role.\n\nAgents (agent_id: model):\n0: openai/model-a\n1: anthropic/model-b\n2: gemini/model-c\n\nRoles:\n- solver: writes a complete answer to the problem.\n- thinker: analyses the problem and the current answer, then suggests how to improve it.\n- verifier: reviews the current answer and replies ACCEPT or REJECT. An ACCEPT ends the run.\n\nThe <state> line gives the turn number (starting at 0), the role that acted on the previous turn, and the verifier's verdict on the current answer (none if it has not been reviewed since it was written).\n\nReply with only this JSON object and nothing else:\n{\"agent_id\": <an agent_id from the list above>, \"role\": \"<solver|thinker|verifier>\"}"}, {"role": "user", "content": "How many times does the letter r appear in the word strawberry?\n<reference_thought_0>strawberry has one r in 'straw' and one in 'berry'.</reference_thought_0>\n<state>turn=3 last_role=thinker last_verdict=REJECT</state>"}, {"role": "assistant", "content": "Agent 1 should write a new answer."}, {"role": "user", "content": "Your reply was invalid: no JSON object with \"agent_id\" and \"role\" found. Reply with only the JSON object {\"agent_id\": <id>, \"role\": \"<solver|thinker|verifier>\"} using an agent_id from the list."}, {"role": "assistant", "content": "{\"agent_id\": 1, \"role\": \"solver\"}"}]}
```

These examples teach the model to recover after a correction. The invalid
`assistant` message is context, not a target: exclude it from the training
loss (for example with a per-message weight of 0, where your fine-tuning
service supports one), or the model also learns to produce it.

### Parser rules

How `parse_route` reads a reply:

1. All `<think>…</think>` blocks, in any letter case, are removed first.
2. Every top-level JSON object in the remaining text is considered, whether
   it stands alone, sits in a code fence or is embedded in prose. Text that is
   not a JSON object is skipped, and objects nested inside another object are
   not searched.
3. Only objects with both `agent_id` and `role` count, and the last one is
   used, even when an earlier one would have been valid. Other keys are
   ignored.
4. `agent_id` must be a JSON integer (`"1"`, `1.0` and `true` are rejected)
   and must be in the roster.
5. `role` must be a string; letter case and surrounding whitespace are
   ignored.

## Conductor

### Input

The Conductor receives a system message that lists every worker, in slot
order, as `<index>: <model id>` indented by two spaces, and a user message
with `USER QUESTION:`, a space and the query. The system message for the
example pool:

<!-- check: conductor-system-prompt -->
```text
You are a Conductor that orchestrates a pool of worker LLMs to solve a task. Design an agentic workflow as THREE equal-length Python lists:
  model_id   = [int, ...]   # which worker (0-indexed) runs each step
  subtasks   = [str, ...]   # the natural-language instruction for each step
  access_list= [list, ...]  # for each step, the indices of EARLIER steps whose
                            # outputs that step may see ([] = none, "all" = every earlier step)
Rules: lists must be equal length (<=5 steps); every model_id must be one of the worker indices listed below; access_list may only reference strictly earlier steps (it is a DAG executed in order); workers never see the user question, only their own subtask and the outputs they may access, so every subtask must be self-contained; the LAST step's output is the final answer. Pick workers to match each subtask's demands.

AVAILABLE LANGUAGE MODELS:
  0: openai/model-a
  1: anthropic/model-b
  2: gemini/model-c

Output the three lists explicitly as 'model_id: [...]', 'subtasks: [...]', 'access_list: [...]', each on its own line. You may reason first, but the three lists must appear.
```

### Target

Three lines, each a label, a colon and a list, in this order:

```text
model_id: [<worker index>, ...]
subtasks: ["<instruction>", ...]
access_list: [<entry>, ...]
```

Write the lists as JSON (double-quoted strings), as the examples do. The
prompt allows reasoning before the lists, and the parser uses the last
occurrence of each list, but reasoning shares the output-token limit with the
lists; the examples have none.

<!-- check: conductor-jsonl -->
```jsonl
{"messages": [{"role": "system", "content": "You are a Conductor that orchestrates a pool of worker LLMs to solve a task. Design an agentic workflow as THREE equal-length Python lists:\n  model_id   = [int, ...]   # which worker (0-indexed) runs each step\n  subtasks   = [str, ...]   # the natural-language instruction for each step\n  access_list= [list, ...]  # for each step, the indices of EARLIER steps whose\n                            # outputs that step may see ([] = none, \"all\" = every earlier step)\nRules: lists must be equal length (<=5 steps); every model_id must be one of the worker indices listed below; access_list may only reference strictly earlier steps (it is a DAG executed in order); workers never see the user question, only their own subtask and the outputs they may access, so every subtask must be self-contained; the LAST step's output is the final answer. Pick workers to match each subtask's demands.\n\nAVAILABLE LANGUAGE MODELS:\n  0: openai/model-a\n  1: anthropic/model-b\n  2: gemini/model-c\n\nOutput the three lists explicitly as 'model_id: [...]', 'subtasks: [...]', 'access_list: [...]', each on its own line. You may reason first, but the three lists must appear."}, {"role": "user", "content": "USER QUESTION: Write a Python function that checks whether a string is a palindrome, and test it."}, {"role": "assistant", "content": "model_id: [1, 0, 2]\nsubtasks: [\"Write a Python function is_palindrome(text) that returns True when text reads the same forwards and backwards, ignoring case, spaces and punctuation.\", \"Write pytest tests for the is_palindrome(text) function above, covering an empty string, mixed case and punctuation.\", \"Review the is_palindrome function and the tests above. Fix any bug and return the final function and tests as one Python file.\"]\naccess_list: [[], [0], [0, 1]]"}]}
{"messages": [{"role": "system", "content": "You are a Conductor that orchestrates a pool of worker LLMs to solve a task. Design an agentic workflow as THREE equal-length Python lists:\n  model_id   = [int, ...]   # which worker (0-indexed) runs each step\n  subtasks   = [str, ...]   # the natural-language instruction for each step\n  access_list= [list, ...]  # for each step, the indices of EARLIER steps whose\n                            # outputs that step may see ([] = none, \"all\" = every earlier step)\nRules: lists must be equal length (<=5 steps); every model_id must be one of the worker indices listed below; access_list may only reference strictly earlier steps (it is a DAG executed in order); workers never see the user question, only their own subtask and the outputs they may access, so every subtask must be self-contained; the LAST step's output is the final answer. Pick workers to match each subtask's demands.\n\nAVAILABLE LANGUAGE MODELS:\n  0: openai/model-a\n  1: anthropic/model-b\n  2: gemini/model-c\n\nOutput the three lists explicitly as 'model_id: [...]', 'subtasks: [...]', 'access_list: [...]', each on its own line. You may reason first, but the three lists must appear."}, {"role": "user", "content": "USER QUESTION: What is the capital of Australia?"}, {"role": "assistant", "content": "model_id: [0]\nsubtasks: [\"What is the capital of Australia? Answer in one sentence.\"]\naccess_list: [[]]"}]}
```

### What a worker receives

Each step's worker gets a single user message: its subtask, preceded by the
subtask and output of each step its access list names. The blocks are labelled
with that step's worker index (`Agent 1` is worker 1), not the step index.
Despite its first line, the message never contains the user question. The last
step of the first example, when steps 0 and 1 answered `(output of step 0)`
and `(output of step 1)`:

<!-- check: conductor-worker-prompt -->
```text
USER QUESTION context:

<Subtask assigned to Agent 1>Write a Python function is_palindrome(text) that returns True when text reads the same forwards and backwards, ignoring case, spaces and punctuation.</Subtask assigned to Agent 1>
<Agent 1 response>(output of step 0)</Agent 1 response>
<Subtask assigned to Agent 0>Write pytest tests for the is_palindrome(text) function above, covering an empty string, mixed case and punctuation.</Subtask assigned to Agent 0>
<Agent 0 response>(output of step 1)</Agent 0 response>

Your subtask: Review the is_palindrome function and the tests above. Fix any bug and return the final function and tests as one Python file.
```

A step with an empty access list gets only `Your subtask: <subtask>`.

### Labelling rules

1. Every subtask must be self-contained: a worker sees only its subtask and
   the outputs its access list names, never the user question. Copy every fact
   the step needs into its subtask.
2. The last step's output is returned as the answer, so the last subtask must
   produce the complete final answer.
3. A step sees an earlier output only if its access list names that step. The
   outputs appear above the subtask, so refer to them as "above".
4. Choose each worker by its index in the system message; one worker can run
   several steps.
5. Use at most 5 steps; one step is fine for a simple question.
6. Keep `--slot-models` identical between generating the data and running
   (see the warning at the top).

### Parser rules

How `parse_workflow` finds the lists:

1. A list is found by its label at the start of a line (spaces and tabs may
   precede it), followed by optional spaces or tabs, `:` or `=`, optional
   whitespace including line breaks, and `[`. Labels are case-insensitive and
   have these spellings:

<!-- check: conductor-labels -->
| List | Accepted labels |
|---|---|
| `model_id` | `model_id`, `model_ids`, `model id`, `model ids` |
| `subtasks` | `subtasks`, `subtask` |
| `access_list` | `access_list`, `access list`, `access` |

2. A label anywhere else, such as mid-line or without a list after it, is
   ignored, so reasoning can mention the labels.
3. When a label appears more than once, its last occurrence is used.
4. The list ends at its matching `]`. Brackets inside quoted strings, including
   after escaped quotes, do not count.
5. The list must be a Python literal or, failing that, JSON: single or double
   quotes, `None` or `null`, and `True` or `true` all parse. Unquoted words are
   an error; items are never split on commas.
6. Curly quotes (`“ ” ‘ ’`) are replaced with straight quotes only when the list
   does not parse as written.
7. A missing or unparsable list is an error.

How `validate_workflow` checks them, in this order. The first failure is
reported, and no worker runs:

8. The three lists have the same length.
9. The workflow has 1 to 5 steps.
10. Each `model_id` is an integer from 0 to the pool size minus 1; `true`,
    `1.0` and `"1"` are rejected.
11. Each subtask is a string that is not empty or only whitespace. Surrounding
    whitespace is removed before use.
12. Each `access_list` entry is one of:
    - `[]`: the step sees no earlier output;
    - `"all"` or `["all"]`, in any letter case: the step sees every earlier
      step;
    - a list of integer indices of earlier steps, each from 0 to the step's
      own index minus 1. Duplicates are merged and order does not matter.

    Anything else is an error: a bare integer, `null`, any other string,
    `"all"` mixed with indices, a string or float index, or a reference to the
    same or a later step.
