# e2e-serving Specification

## Purpose

Serve Fugu as a single OpenAI-compatible chat model (`python -m openfugu.serve`):
for each request, an API router model and a pool of API worker models, all
reached through litellm, run the TRINITY coordination loop, and the client
receives one answer without seeing the pool.

## Requirements

### Requirement: Validate the configuration before serving

The server SHALL take the worker pool from `--slot-models` and the router
endpoint from the `FUGU_ROUTER_*` variables, each falling back to
`FUGU_CONDUCTOR_*` and then to `FUGU_API_KEY` / `FUGU_BASE_URL`. Before it
listens, it SHALL check that litellm can resolve the provider of the router and
of every worker, and can find an API key for each; on any failure it SHALL log
the problem and exit with status 1, without a traceback and without calling a
model.

#### Scenario: No router model is configured

- **WHEN** neither `FUGU_ROUTER_MODEL` nor `FUGU_CONDUCTOR_MODEL` is set
- **THEN** the server SHALL exit with status 1 and name both variables

#### Scenario: A worker has no credentials

- **WHEN** a `--slot-models` entry needs a provider key that is not set and
  `FUGU_API_KEY` is not set
- **THEN** the server SHALL exit with status 1 and name the worker index and
  model

#### Scenario: The slot list has an empty entry

- **WHEN** `--slot-models` contains an empty entry
- **THEN** the server SHALL exit with status 1

#### Scenario: Invalid options

- **WHEN** `--port` is outside 1 to 65535, `--max-turns` is below 1, or
  `--host` is empty
- **THEN** the server SHALL exit with status 2

### Requirement: Listen on localhost unless told otherwise

The server SHALL listen on 127.0.0.1 by default and on the address given by
`--host` otherwise. It SHALL have no authentication of its own.

#### Scenario: Default address

- **WHEN** the server starts without `--host`
- **THEN** it SHALL listen on 127.0.0.1 and the `--port` port (default 8088)

#### Scenario: Address that cannot be used

- **WHEN** the server cannot listen on the requested address
- **THEN** it SHALL log the address and exit with status 1

### Requirement: Answer chat completions through the coordination loop

`POST /v1/chat/completions` SHALL answer the text of the last user message
(for a list of content parts, the text parts joined by line breaks) by running
the Coordinator for up to `--max-turns` turns, and SHALL return an OpenAI-shaped
`chat.completion` whose message content is the latest solver answer, whose
`model` echoes the request's `model` (default `fugu`), and whose `usage`
reports the number of turns as `fugu_turns`.

#### Scenario: A verifier accepts the answer

- **WHEN** the router picks a solver and then a verifier, and the verifier
  replies ACCEPT
- **THEN** the response SHALL contain the solver's reply and `fugu_turns` 2

#### Scenario: Content parts

- **WHEN** the last user message's content is a list of text and image parts
- **THEN** the query SHALL be the text parts joined by line breaks

### Requirement: Route each turn with an API router

Each turn, the router model SHALL receive a system prompt listing the allowed
agents as `<agent_id>: <model id>` and a user message holding the query, the
`<reference_thought_N>` blocks of earlier solver turns and a
`<state>turn=T last_role=R last_verdict=V</state>` line, and SHALL answer
`{"agent_id": <int>, "role": "solver"|"thinker"|"verifier"}`. An invalid reply
SHALL be retried with a correction message naming the problem, and a transient
API failure after a backoff, for at most 3 calls per turn. When no call yields
a valid decision, or the API fails with a permanent error, the turn SHALL fall
back to the first allowed agent as solver and log a warning; routing SHALL
never fail the request. The exact prompts are specified in
`docs/FINE_TUNING_FORMAT.md`.

#### Scenario: Invalid router reply

- **WHEN** the router names an agent outside the pool
- **THEN** it SHALL be asked again with its reply and a correction message
  that states `agent_id <n> is not in the list`

#### Scenario: Router never gives a valid decision

- **WHEN** three router calls in a turn give no valid decision
- **THEN** the turn SHALL run the first allowed agent as solver

### Requirement: Coordinate the pool

The Coordinator SHALL run the chosen agent in the chosen role: a solver's
reply becomes the current answer and its thought joins the router's
observation; a thinker's `<suggestion>` is passed to the next worker and its
`<suggested_role>` replaces the router's role on the next turn; a verifier
reviews the current answer, and ACCEPT ends the run. A verifier chosen before
any solver has answered SHALL run as a solver. Agent ids SHALL index
`--slot-models`, and an id outside the pool SHALL never be wrapped around.

#### Scenario: The turns run out

- **WHEN** no verifier accepts within `--max-turns` turns
- **THEN** the response SHALL contain the latest solver answer

### Requirement: Report errors without leaking details

The server SHALL answer a malformed request with status 400 and
`{"error": "<message>"}` without calling a model, an unknown path with 404,
and a worker call that still fails after its retries, or any unexpected error,
with 500 and `{"error": "internal server error"}`. Error details SHALL go only
to the server log, with API keys removed.

#### Scenario: Malformed request

- **WHEN** the body is not valid JSON, is not a JSON object, has no messages,
  or the last user message has content that is neither a string nor a list
- **THEN** the server SHALL respond 400 and no model SHALL be called

#### Scenario: A worker fails

- **WHEN** a worker call fails with a permanent error
- **THEN** the server SHALL respond 500 with `internal server error` and log
  the error without the API key

### Requirement: Never parse an unread body as another request

When the server responds without having read the whole request body, it SHALL
send `Connection: close` and close the connection, so that on a keep-alive
connection the unread bytes are never parsed as a further request. It SHALL
reject a request body sent with `Transfer-Encoding` with 411 and a request
with an invalid or repeated `Content-Length` with 400, closing the connection
in both cases. Before closing any connection, it SHALL read and discard what
the client still sends, until the client closes or 5 seconds pass, so that
unread input never makes TCP reset the connection and destroy the response. A
request whose body was read in full SHALL keep the connection open.

#### Scenario: A request hidden in the body of a request to an unknown path

- **WHEN** a POST to an unknown path, or a GET, carries a body that is itself
  a complete HTTP request
- **THEN** the server SHALL send exactly one response, with
  `Connection: close`, and SHALL never answer the hidden request

#### Scenario: Chunked request body

- **WHEN** a POST to `/v1/chat/completions` uses `Transfer-Encoding: chunked`
- **THEN** the server SHALL respond 411, close the connection and call no
  model

#### Scenario: The body arrives after the response

- **WHEN** the server has answered a request without reading its body, and
  the body arrives afterwards
- **THEN** the client SHALL still receive the complete response

#### Scenario: Repeated Content-Length

- **WHEN** a request has more than one `Content-Length` header
- **THEN** the server SHALL respond 400 and close the connection

#### Scenario: Fully read requests on one connection

- **WHEN** a client sends several requests on one connection, each with its
  body in full
- **THEN** the server SHALL answer each of them on that connection
