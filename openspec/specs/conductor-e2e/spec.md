# conductor-e2e Specification

## Purpose

Run Fugu-Ultra's Conductor from the command line (`python -m openfugu.ultra`):
a Conductor model reached through litellm writes a multi-step workflow over a
pool of API worker models; the workflow is parsed and validated before any
worker runs, then executed in order, and the last step's output is the answer.

## Requirements

### Requirement: Validate the configuration before calling a model

The CLI SHALL require `--query` and `--slot-models`, take the Conductor
endpoint from `FUGU_CONDUCTOR_MODEL`, `FUGU_CONDUCTOR_API_KEY` and
`FUGU_CONDUCTOR_BASE_URL` (the key and base URL falling back to `FUGU_API_KEY`
and `FUGU_BASE_URL`), and check that litellm can resolve the provider of the
Conductor and of every worker and find an API key for each. On any failure it
SHALL print the problem and exit with status 1 without calling a model.

#### Scenario: No Conductor model is configured

- **WHEN** `FUGU_CONDUCTOR_MODEL` is not set
- **THEN** the CLI SHALL exit with status 1 and name the variable

#### Scenario: The Conductor has its own credentials

- **WHEN** `FUGU_CONDUCTOR_API_KEY` and `FUGU_CONDUCTOR_BASE_URL` are set
- **THEN** the Conductor call SHALL use them, and worker calls SHALL use
  `FUGU_API_KEY` and `FUGU_BASE_URL`

### Requirement: Prompt the Conductor with the real pool

The Conductor SHALL receive a system prompt that states the workflow format
and rules and lists every `--slot-models` entry as `<index>: <model id>`, and
the user message `USER QUESTION: <query>`. The exact prompt is specified in
`docs/FINE_TUNING_FORMAT.md`.

#### Scenario: Pool of any size

- **WHEN** `--slot-models` has n entries
- **THEN** the prompt SHALL list workers 0 to n-1 and a workflow using worker
  n-1 SHALL run on the last entry

### Requirement: Parse the workflow strictly

The CLI SHALL read the `model_id`, `subtasks` and `access_list` lists only
from labels at the start of a line followed by `:` or `=` and a list, using the
last occurrence of each label, and SHALL parse each list as a Python literal
or, failing that, JSON. A missing or unparsable list SHALL be an error.

#### Scenario: Labels inside reasoning

- **WHEN** the Conductor mentions a label in prose before the real lists
- **THEN** only the line-leading labels followed by a list SHALL be used

#### Scenario: Unquoted items

- **WHEN** a list contains unquoted words
- **THEN** parsing SHALL fail instead of splitting the items on commas

### Requirement: Validate the workflow before any worker runs

The CLI SHALL check that the three lists have the same length and 1 to 5
steps, that every `model_id` is an integer index into the pool, that every
subtask is a non-empty string, and that every `access_list` entry is `[]`,
`"all"`, `["all"]` (any case) or a list of indices of earlier steps. The first
broken rule SHALL be reported with the list, the step, the value and the rule,
and the CLI SHALL exit with status 1 without calling a worker.

#### Scenario: A step refers to itself or a later step

- **WHEN** an `access_list` entry names the same or a later step
- **THEN** validation SHALL fail and no worker SHALL be called

#### Scenario: Too many steps

- **WHEN** the workflow has 6 steps
- **THEN** validation SHALL fail instead of dropping steps

### Requirement: Execute the workflow in order

Each step SHALL run on its worker with a single user message holding its
subtask, preceded by the subtask and output of every step its access list
names, as `<Subtask assigned to Agent N>` and `<Agent N response>` blocks,
where N is that step's worker index. The last step's output SHALL be the
answer.

#### Scenario: A step sees every earlier step

- **WHEN** a step's access entry is `"all"` or `["all"]`
- **THEN** its message SHALL include the subtasks and outputs of all earlier
  steps

### Requirement: Report failures clearly

A failed Conductor call, an invalid workflow and a failed worker call SHALL
each be reported on stderr with a clear message and exit status 1, without a
traceback; an invalid workflow SHALL also print the Conductor's output.

#### Scenario: A worker call fails

- **WHEN** a worker call fails with a permanent error
- **THEN** the CLI SHALL print `worker call failed` with the model and error,
  and exit with status 1
