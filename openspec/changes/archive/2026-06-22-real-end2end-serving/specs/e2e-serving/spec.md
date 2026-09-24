# e2e-serving

## ADDED Requirements

### Requirement: Serve a trained per-step head over a real local worker pool

The serving layer SHALL serve a trained per-step TRINITY router head over a
pool of locally-resident worker models behind the OpenAI-compatible
`/v1/chat/completions` endpoint, so that a client request is answered by the
full per-step Coordinator loop routing to the real workers the head was trained
against, without requiring any external API.

#### Scenario: Serve with a head-only trained vector

- **WHEN** the server is started with a base SVF vector and a separate trained
  head-only vector of length 10240 (e.g. `trinity_perstep.npy`)
- **THEN** the router SHALL apply the base SVF adaptation and replace the linear
  head with the trained head, and SHALL route using that trained head

#### Scenario: Serve with a full vector (backward compatible)

- **WHEN** the server is started with only a full-length 19456 vector and no
  head-only override
- **THEN** the server SHALL load it exactly as before (SVF + head from the same
  vector) and serve normally

#### Scenario: Route to local workers, not an API

- **WHEN** the server is started in local-pool mode (a set of local worker model
  paths) instead of litellm slot models
- **THEN** each routed turn SHALL be answered by the corresponding local worker
  model, and the server SHALL require no external API key

### Requirement: End-to-end verification of a live request

The change SHALL include an end-to-end test that boots the server with the
trained head and the local worker pool, issues a real HTTP request to the
endpoint, and asserts a real, worker-produced answer is returned through the
full per-step loop.

#### Scenario: A live GSM8K request returns a real numeric answer

- **WHEN** the end-to-end test POSTs a GSM8K question to
  `/v1/chat/completions` against the running server
- **THEN** the response SHALL be a valid OpenAI-shaped chat completion whose
  content contains a numeric answer produced by the local worker pool
- **AND** the response SHALL report a non-zero orchestration turn count

#### Scenario: The served answer comes from the real loop, not the mock

- **WHEN** the end-to-end test runs against the local-pool server
- **THEN** the answer SHALL be produced by a real local worker model, not the
  offline `MockWorker` stand-in
