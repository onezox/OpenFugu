# Design — real-end2end-serving

## Context

OpenFugu has a faithful per-step Coordinator (`mini.py`), a trained per-step
router head (`trinity_perstep.npy`, shape `(10240,)`), and an OpenAI-compatible
server (`serve.py`). Today `serve.py` only loads a full `(19456,)` vector and
only runs workers via `LiteLLMWorker` (API) or `MockWorker`. The trained head
and the local worker pool it was trained on cannot be served together. This
change connects them and proves the loop with a live request.

## Goals / Non-Goals

- **Goal**: serve the trained head over the real local pool behind the existing
  endpoint; verify a live request returns a real worker answer.
- **Goal**: keep the litellm and mock paths working unchanged.
- **Non-Goal**: production hardening (auth, batching, streaming, TLS). The
  endpoint stays a stdlib `http.server` router surface.
- **Non-Goal**: retraining or new training. We serve the head already produced.

## Decisions

- **Head layering via a separate `--head` vector, not a new file format.**
  `FuguRouter` already loads a full vector then sets `self.head`. We add an
  optional head-only override: after the base SVF vector is applied, if a
  `(10240,)` head is provided, reshape it to `(10, 1024)` and assign
  `router.head`. Rationale: minimal, backward compatible, mirrors exactly what
  the per-step trainer does at eval time (`router.head = trained`). Alternative
  considered: bake head back into a full `(19456,)` vector — rejected, it forces
  an extra offline merge step and loses the clean "base SVF + trained head"
  separation the training code already uses.

- **Reuse the training-time local pool worker for serving.** The per-step
  trainer's `LocalPoolWorker` already implements the `(role, messages, agent_id)`
  worker protocol over local multi-vendor models. We expose an equivalent in the
  serving path (a `--local-models` CSV of paths + per-model device) rather than
  inventing a second dispatch mechanism. Rationale: serve over exactly the pool
  the head was trained against; one worker abstraction. Alternative: force users
  through litellm pointing at local vLLM — rejected, adds a server dependency the
  ponytail principle says we don't need.

- **End-to-end test boots a real server in-process / subprocess and uses HTTP.**
  The test starts the server (trained head + local pool), POSTs a real GSM8K
  question, and asserts a numeric answer + non-zero turns. Rationale: the only
  honest proof of "end to end" is a real socket request answered by real
  workers. Alternative: call `Coordinator.run` directly — rejected, that skips
  the serving layer which is the whole point of this change.

## Risks / Trade-offs

- [Local pool needs GPUs at serve time] → The local-pool mode is opt-in; litellm
  and mock paths remain for CPU/no-GPU environments. Documented in README.
- [Worker load time dominates a single request] → Acceptable for a smoke/e2e
  proof; the server keeps models resident across requests, so only the first
  request pays load cost.
- [Greedy workers may emit a non-numeric final] → The e2e test uses a question
  the pool reliably solves and asserts on extracted numeric answer; on miss it
  fails loudly rather than passing silently.

## Migration Plan

Additive only. New CLI flags (`--head`, `--local-models`) default to off, so
existing `--vector` + `--slot-models` invocations are unchanged. No rollback
needed beyond not passing the new flags.

## Open Questions

None blocking. Device placement for local workers is a CLI detail (default:
router on `cuda:0`, workers round-robin over remaining visible GPUs).
