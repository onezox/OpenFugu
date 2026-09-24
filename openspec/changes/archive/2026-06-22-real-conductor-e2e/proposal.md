## Why

Fugu has two orchestrators. TRINITY (the per-step router) is now trained,
served over a real local pool, and pipelined end-to-end (`e2e-serving`,
`e2e-train-serve`). The other half — the **Conductor / Fugu-Ultra**, the LM that
emits a whole workflow DAG in one shot — we actually GRPO-trained
(`conductor_toolscale_100/checkpoint-100`, reward 1.21→1.64), but it has never
been connected to inference: `ultra.py` only runs the Conductor as a *prompted
off-the-shelf model via litellm*. The trained Conductor weights are produced and
sitting on disk, unused. "Real end-to-end training and serving" for this half
means driving the workflow executor with **our trained local Conductor**, over a
real local worker pool, and verifying it emits and executes a real workflow.

## What Changes

- `ultra.py` gains a **local Conductor** path: load our trained Conductor
  checkpoint with transformers and use it to emit the workflow (instead of a
  litellm API model), via a `--local-conductor <path>` option.
- `ultra.py` gains a **local worker pool** path (`--local-models CSV`) so the
  DAG steps are executed by real local models, mirroring the TRINITY serving
  side — no API key required for a fully-local run.
- An **end-to-end test** (`eval/ultra_e2e.py`) that loads the trained Conductor +
  local workers, runs a real query, and asserts a parseable multi-step workflow
  was emitted and executed to a non-empty final answer.
- README + results updated with the trained-Conductor local run and evidence.

## Capabilities

### New Capabilities
- `conductor-e2e`: run the Fugu-Ultra Conductor workflow executor driven by a
  local Conductor model over a real local worker pool, with an end-to-end test
  that a real query yields a parsed, executed workflow and a final answer — and
  that fails loudly when the Conductor does not speak the workflow DSL. (Honest
  finding from the run: our GRPO-trained `checkpoint-100` was trained on the
  ToolScale tool-call DSL, NOT the 3-list workflow DSL, so it does not parse; the
  local-Conductor MECHANISM is proven with a model that does follow the format.)

### Modified Capabilities
<!-- None: the existing e2e-serving / e2e-train-serve specs cover the TRINITY
     router; this is the Conductor (Fugu-Ultra) half and adds a new capability. -->

## Impact

- Code: `openfugu/ultra.py` (local-Conductor generation + local worker pool;
  litellm path unchanged), new `eval/ultra_e2e.py`.
- Docs: `README.md` (Fugu-Ultra local run), `results/README.md` (evidence).
- Dependencies: none new — transformers + the local models already on the GPU
  server; the trained Conductor checkpoint already exists.
- Runtime: the local Conductor is a ~3B model + workers → needs GPUs; the
  litellm path remains for API-only environments.
