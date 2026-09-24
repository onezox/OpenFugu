## Why

OpenFugu can now train a TRINITY router at Fugu's real per-step granularity
(`train_trinity_perstep.py`: base 0.750 → 1.000) and serve a coordinator behind
one OpenAI-compatible endpoint (`serve.py`). But the two halves do not connect:
`serve.py` only accepts a full `(19456,)` base vector and only runs workers via
litellm (API) or a mock — there is no path to serve the **trained per-step head**
`(10240,)` over the **real local worker pool** the head was trained against, and
nothing verifies a real client request returning a real worker-produced answer.
"Real end-to-end" means closing that loop and proving it with a live request.

## What Changes

- `serve.py` accepts a trained **head-only** `(10240,)` vector (e.g.
  `trinity_perstep.npy`) layered on the base SVF vector, not just a full
  `(19456,)` vector. Backward compatible: a `(19456,)` vector still works.
- A new **local-pool worker** option for serving: the same multi-vendor
  `≤8B` models used in per-step training, resident on local GPUs, dispatched by
  `(role, messages, agent_id)` — so the served coordinator routes to the real
  workers its head was trained on (no API key required).
- An **end-to-end smoke** (`eval/serve_e2e.py` or equivalent) that boots the
  server with the trained head + local pool, POSTs a real GSM8K question to
  `/v1/chat/completions`, and asserts a real numeric answer comes back through
  the full per-step loop.
- README/results updated to document the real end-to-end serving path.

## Capabilities

### New Capabilities
- `e2e-serving`: serve the trained per-step TRINITY head over a real local
  worker pool behind the OpenAI-compatible endpoint, with an end-to-end test
  that a live request returns a real worker-produced answer.

### Modified Capabilities
<!-- No existing OpenSpec spec captures serving behavior yet; this is the first. -->

## Impact

- Code: `openfugu/serve.py` (head-only vector loading; local-pool worker
  selection), `openfugu/mini.py` (reuse/expose a local-pool worker class for
  serving), a new end-to-end test under `eval/`.
- Docs: `README.md` (serve section), `results/README.md` (e2e evidence).
- Dependencies: none new — transformers (already used by the router), the local
  worker models already present on the GPU server. litellm path unchanged.
- Runtime: serving with a local pool requires GPUs for the workers; the API
  (litellm) and mock paths remain for environments without local models.
