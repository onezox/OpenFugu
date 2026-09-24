## 1. Serve the trained head

- [x] 1.1 Add an optional `--head` flag to `serve.py` that loads a head-only
  `(10240,)` vector and overrides `router.head` after the base SVF vector is
  applied (reshape to `(HEAD_ROWS, HIDDEN)`); reject wrong-length head vectors.
- [x] 1.2 Keep the full `(19456,)` `--vector` path working unchanged when no
  `--head` is given (backward compatibility).

## 2. Serve over the real local worker pool

- [x] 2.1 Expose a local-pool worker for serving that implements the
  `(role, messages, agent_id)` protocol over local multi-vendor models (reuse
  the per-step trainer's `LocalPoolWorker` pattern; place it where `serve.py`
  can import it).
- [x] 2.2 Add a `--local-models` CLI option (CSV of model paths, optional
  per-entry device) to `serve.py`; when given, use the local pool instead of
  litellm/mock and require no API key. Default device placement: router on
  `cuda:0`, workers round-robin over remaining GPUs.

## 3. End-to-end verification

- [x] 3.1 Write `eval/serve_e2e.py`: boot the server (trained head + local
  pool) as a subprocess, wait for readiness, POST a real GSM8K question to
  `/v1/chat/completions`, assert an OpenAI-shaped response whose content has the
  correct numeric answer and a non-zero turn count; assert the answer is NOT
  from `MockWorker`.
- [x] 3.2 Run `serve_e2e.py` on the GPU server against the trained
  `trinity_perstep.npy` head + the local pool; capture the run log to
  `results/serve_e2e_run.txt`.

## 4. Documentation

- [x] 4.1 Update `README.md` serve section with the real end-to-end command
  (trained head + local pool) alongside the existing litellm/mock commands.
- [x] 4.2 Add the end-to-end serving evidence to `results/README.md` (the live
  request returning a real worker answer).
