# Design — real-conductor-e2e

## Context

`ultra.py` reconstructs Fugu-Ultra's Conductor: a single LM emits a 3-list
workflow DAG (model_id / subtasks / access_list), executed in topological order
with access-list visibility injection. Today the Conductor is only a prompted
litellm model and workers only run via litellm. We GRPO-trained a real Conductor
(`conductor_toolscale_100/checkpoint-100`) that has never been used for
inference. This change connects it: local trained Conductor + local worker pool.

## Goals / Non-Goals

- **Goal**: emit the workflow with our trained local Conductor checkpoint;
  execute steps over a real local worker pool; verify end-to-end.
- **Goal**: keep the litellm Conductor + worker paths working unchanged.
- **Non-Goal**: changing the parser / DAG executor — it is faithful and
  self-tested; we only add generation/worker backends.
- **Non-Goal**: a benchmark of the trained Conductor's quality. The honest claim
  is "trained weights now drive real inference and emit/execute a real
  workflow," not a score.

## Decisions

- **Add a local-generation backend alongside `LiteLLMWorker.conduct`.** A small
  `LocalConductor` loads the checkpoint with transformers and exposes a
  `conduct(messages) -> text` method matching what `main()` already calls.
  Rationale: the executor consumes a completion string; where it comes from
  (API vs local) is swappable behind one method. Alternative: route the local
  model through litellm's local provider — rejected, adds a server dependency for
  no benefit (ponytail).

- **Reuse the TRINITY-side local worker pattern for DAG step execution.** A
  local worker pool implementing `(subtask, messages, agent_id) -> reply` over
  resident models, same as `serve.py`'s `LocalPoolWorker`. Rationale: one local
  worker abstraction across both orchestrators; serve the Conductor over exactly
  the models available locally.

- **The e2e test asserts structure + execution, not correctness of the answer.**
  It checks: a parseable workflow (≥1 step), execution ran, final answer
  non-empty. Rationale: the trained Conductor's *job* is to emit and drive a
  valid workflow; asserting a specific gold answer would conflate Conductor
  quality with worker quality. Honest scope: prove the mechanism runs on trained
  weights, fail loudly if no workflow parses.

## Risks / Trade-offs

- [Trained Conductor may emit a workflow the parser rejects] → The e2e test
  fails loudly (non-zero) rather than passing silently; that surfaces a real
  format mismatch to fix, not hide.
- [3B Conductor + workers exceed one GPU] → Conductor and workers take explicit
  devices (CLI), placed on separate GPUs like the TRINITY serving side.
- [Greedy decode may yield a trivial 1-step workflow] → Acceptable: ≥1 executed
  step with a non-empty answer satisfies "emits and executes a real workflow";
  multi-step richness is a quality question, out of scope here.

## Migration Plan

Additive: new `--local-conductor` and `--local-models` flags default off, so
existing `--conductor` + `--slot-models` invocations are unchanged. No rollback
beyond not passing the new flags.

## Open Questions

None blocking. Device placement is a CLI detail (Conductor on `cuda:0`, workers
on remaining GPUs).
