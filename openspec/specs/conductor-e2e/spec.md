# conductor-e2e Specification

## Purpose
TBD - created by archiving change real-conductor-e2e. Update Purpose after archive.
## Requirements
### Requirement: Drive the workflow executor with a local Conductor

The Fugu-Ultra workflow executor SHALL be able to use a locally-loaded Conductor
model to emit the workflow, instead of only a prompted litellm API model, so a
fully-local Conductor can drive the DAG over local workers.

#### Scenario: Local Conductor emits a parseable workflow that executes

- **WHEN** the executor is run with `--local-conductor <model path>` (a model
  that follows the 3-list workflow format) and a real query
- **THEN** the local Conductor SHALL generate a completion that parses into a
  valid 3-list workflow (equal-length model_id / subtasks / access_list)
- **AND** the executor SHALL run the workflow steps in topological order over the
  local worker pool to a non-empty final answer

#### Scenario: litellm Conductor path still works

- **WHEN** the executor is run with `--conductor <litellm id>` and no
  `--local-conductor`
- **THEN** it SHALL behave exactly as before (Conductor via litellm API)

### Requirement: Execute the workflow over a real local worker pool

The executor SHALL be able to run the workflow's worker steps on a pool of
locally-resident models, requiring no external API, mirroring the TRINITY
serving side.

#### Scenario: Local workers execute the DAG steps

- **WHEN** the executor is run with `--local-models <CSV of paths>`
- **THEN** each workflow step SHALL be answered by the corresponding local
  worker model, and the run SHALL require no API key

### Requirement: End-to-end verification reports honestly whether the workflow parsed

The change SHALL include an end-to-end test that loads a local Conductor and
local workers, runs a real query, and asserts a parsed, executed workflow with a
final answer — failing loudly (non-zero) when the Conductor does not emit a
parseable workflow, so a Conductor/DSL mismatch is surfaced, never hidden.

#### Scenario: A workflow-speaking Conductor yields a parsed, executed workflow

- **WHEN** the end-to-end test runs a local Conductor that follows the 3-list
  format plus a local pool on a real query
- **THEN** it SHALL assert the emitted workflow parsed into at least one step,
  was executed, and produced a non-empty final answer

#### Scenario: A Conductor that does not speak the workflow DSL fails loudly

- **WHEN** the loaded Conductor emits text that does not parse into a 3-list
  workflow (e.g. a model trained on a different output DSL)
- **THEN** the end-to-end test SHALL exit non-zero and print the raw completion,
  rather than reporting a false pass

