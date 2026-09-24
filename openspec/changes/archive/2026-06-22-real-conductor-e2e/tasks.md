## 1. Local Conductor generation

- [x] 1.1 Add a `LocalConductor` to `ultra.py`: load a trained Conductor
  checkpoint with transformers (try/except dtype= vs torch_dtype=), expose
  `conduct(messages) -> completion text` (greedy, enough new tokens for a
  3-list workflow). Place it on a configurable device (default `cuda:0`).
- [x] 1.2 Add a `--local-conductor <path>` CLI flag to `main()`; when given, use
  `LocalConductor` to emit the workflow instead of `LiteLLMWorker.conduct`.
  Keep the `--conductor <litellm id>` path unchanged when not given.

## 2. Local worker pool for DAG execution

- [x] 2.1 Add a `LocalPoolWorker` to `ultra.py` implementing the
  `(subtask, messages, agent_id) -> reply` worker protocol over local resident
  models (path or path@device entries; round-robin GPUs by default).
- [x] 2.2 Add a `--local-models <CSV>` CLI flag; when given, execute the
  workflow steps with the local pool (no API key). Keep `--slot-models`
  (litellm) working when not given.

## 3. End-to-end verification

- [x] 3.1 Write `eval/ultra_e2e.py`: load the trained local Conductor + local
  worker pool, run a real query through `ConductorExecutor`, and assert a
  parseable workflow (>=1 step), execution ran, and a non-empty final answer;
  exit non-zero on no parseable workflow or empty answer.
- [x] 3.2 Run `ultra_e2e.py` on the GPU server with the trained Conductor
  checkpoint + local pool; capture the run log to
  `results/conductor_e2e_run.txt`.

## 4. Documentation

- [x] 4.1 Update `README.md` with the trained-local-Conductor Fugu-Ultra run
  command (alongside the existing litellm command).
- [x] 4.2 Add the trained-Conductor end-to-end evidence to `results/README.md`.
