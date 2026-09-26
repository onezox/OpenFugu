# OpenFugu

OpenFugu serves a pool of LLMs as one OpenAI-compatible chat model. For each
request, a router model decides turn by turn which model of the pool acts next
and in which role (solver, thinker or verifier), until a verifier accepts an
answer. A second entry point lets a Conductor model write a whole multi-step
workflow that the pool then runs. Every model is reached through
[litellm](https://github.com/BerriAI/litellm); nothing runs locally.

This is an API-only version of [OpenFugu](https://github.com/trotsky1997/OpenFugu)
(Apache-2.0), an independent reimplementation of the ideas in two Sakana AI
papers:

- [TRINITY: An Evolved LLM Coordinator](https://arxiv.org/abs/2512.04695)
  (arXiv:2512.04695), the basis of the turn-by-turn coordinator;
- [Learning to Orchestrate Agents in Natural Language with the Conductor](https://arxiv.org/abs/2512.04388)
  (arXiv:2512.04388), the basis of the workflow Conductor.

OpenFugu is not affiliated with or endorsed by Sakana AI. See [NOTICE](NOTICE).

## How it works

**`python -m openfugu.serve`** answers `POST /v1/chat/completions`. For each
request, the Coordinator takes the last user message as the query and runs up
to `--max-turns` turns. Each turn, the router model replies
`{"agent_id": <i>, "role": "<role>"}`, and the i-th model of `--slot-models`
runs in that role:

- a **solver** writes an answer;
- a **thinker** analyses the query and the current answer and suggests an
  improvement for the next turn;
- a **verifier** replies ACCEPT or REJECT. ACCEPT ends the run.

The response is the latest solver answer. The client never sees the pool.

**`python -m openfugu.ultra`** is a command-line tool. The Conductor model
writes a workflow of up to 5 steps: which worker runs each step, the
instruction for each step, and which earlier outputs each step may see. The
workflow is validated before any worker runs; the steps then run in order, and
the last step's output is the answer.

The router and the Conductor can be any chat models that follow their
prompts, but they are meant to be fine-tuned for the pool they command.
[docs/FINE_TUNING_FORMAT.md](docs/FINE_TUNING_FORMAT.md) specifies the
training data.

## Requirements

Python 3.11 to 3.14. The test suite passes on 3.11, 3.12, 3.13 and 3.14.
litellm 1.98.0 declares support for 3.10, but it cannot be imported there (it
uses `typing.NotRequired`, new in 3.11).

## Install

```sh
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Hosts without internet access.** litellm 1.98.0 ships a copy of the
`cl100k_base` tokenizer that fails tiktoken's checksum. On the first
`import litellm`, tiktoken deletes that copy and downloads the tokenizer
(1.7 MB) from `openaipublic.blob.core.windows.net` into litellm's package
directory; without network access, the import fails. To prepare an offline
host, run `python -c "import litellm"` once in an environment with internet
access and the same litellm version, copy
`litellm/litellm_core_utils/tokenizers/9b5ad71b2ce5302211f9c61530b329a4922fc6a4`
from its `site-packages` to a directory on the offline host, and set
`CUSTOM_TIKTOKEN_CACHE_DIR` to that directory. litellm also fetches its model
price list at import time and falls back to a bundled copy after a 5-second
timeout; set `LITELLM_LOCAL_MODEL_COST_MAP=True` to skip that fetch.

## Configuration

Endpoints are configured with environment variables. Model names are litellm
model ids of the form `provider/model`, for example `anthropic/<model>`. For
an OpenAI-compatible server, such as one hosting a fine-tuned model, use
`openai/<model>` together with a base URL.

| Variable | Used for | When unset or blank |
|---|---|---|
| `FUGU_ROUTER_MODEL` | Router model (`serve`) | `FUGU_CONDUCTOR_MODEL`; `serve` needs one of the two |
| `FUGU_ROUTER_API_KEY` | Router API key | `FUGU_CONDUCTOR_API_KEY`, then `FUGU_API_KEY`, then litellm's provider variable |
| `FUGU_ROUTER_BASE_URL` | Router base URL | `FUGU_CONDUCTOR_BASE_URL`, then `FUGU_BASE_URL`, then the provider's default |
| `FUGU_CONDUCTOR_MODEL` | Conductor model (`ultra`); router fallback | `ultra` fails at startup |
| `FUGU_CONDUCTOR_API_KEY` | Conductor API key; router fallback | `FUGU_API_KEY`, then litellm's provider variable |
| `FUGU_CONDUCTOR_BASE_URL` | Conductor base URL; router fallback | `FUGU_BASE_URL`, then the provider's default |
| `FUGU_API_KEY` | API key of every worker slot; last fallback for the router and the Conductor | litellm's provider variable for each model |
| `FUGU_BASE_URL` | Base URL of every worker slot; last fallback for the router and the Conductor | the provider's default for each model |

Each setting falls back from router to Conductor to base: `FUGU_ROUTER_*`,
then `FUGU_CONDUCTOR_*`, then `FUGU_API_KEY` and `FUGU_BASE_URL`. The
Conductor skips the first step. litellm's provider variables are its own, such
as `ANTHROPIC_API_KEY` for `anthropic/` models.

At startup, both entry points check that litellm can resolve every model's
provider and find a key for it. They make no network call for this. On
failure, they print the problem and exit with status 1. Notes:

- An endpoint that needs no key still needs one to pass the startup check; any
  placeholder works.
- `FUGU_API_KEY` and `FUGU_BASE_URL` apply to every worker slot. For a pool
  that mixes providers, either put the providers behind one gateway, or leave
  both unset and set each provider's own variable.

## The worker pool: `--slot-models`

Both entry points take the pool as `--slot-models`: a comma-separated list of
litellm model ids with no empty entries. Entry i is agent i: the router
answers with an index into this list, and the Conductor's workflow names
workers by index. The pool can have any number of models.

> [!WARNING]
> **Never change the slot order after fine-tuning.** A fine-tuned router or
> Conductor learns what each index is good at. Reordering `--slot-models`
> silently sends every decision to a different model, and nothing detects it.
> Serve with the exact list the training data was generated with; adding,
> removing or replacing a model means new training data and a new fine-tune.

## Running the server

```sh
export FUGU_ROUTER_MODEL=openai/my-router
export FUGU_ROUTER_BASE_URL=https://router.example.com/v1
export FUGU_ROUTER_API_KEY=...
export FUGU_BASE_URL=https://gateway.example.com/v1
export FUGU_API_KEY=...

python -m openfugu.serve \
  --slot-models "openai/model-a,openai/model-b,openai/model-c" \
  --host 127.0.0.1 --port 8088 --max-turns 5
```

| Option | Default | |
|---|---|---|
| `--slot-models` | required | The worker pool; see above. |
| `--host` | `127.0.0.1` | IPv4 address or host name to listen on. |
| `--port` | `8088` | 1 to 65535. |
| `--max-turns` | `5` | Turns per request; at least 1. |

The server logs to stderr and runs until interrupted. It serves:

- `POST /v1/chat/completions`: answers the request.
- `GET /v1/models`: lists the single model, `fugu`.
- `GET /health` (or `/`): returns `{"status": "ok", "model": "fugu"}`.

```sh
curl -s http://127.0.0.1:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "fugu", "messages": [{"role": "user", "content": "What is 17 * 23?"}]}'
```

```json
{"id": "chatcmpl-…", "object": "chat.completion", "created": 1790000000, "model": "fugu",
 "choices": [{"index": 0, "message": {"role": "assistant", "content": "…"}, "finish_reason": "stop"}],
 "usage": {"fugu_turns": 2}}
```

`model` echoes the request's `model` (`fugu` when it has none), and
`usage.fugu_turns` is the number of turns the request ran. Errors return
`{"error": "<message>"}`:

- **400** for a malformed request: invalid JSON, a body that is not a JSON
  object, a missing or empty `messages` list, a message that is not an object,
  or a last user message whose content is neither a string nor a list of
  content parts.
- **404** for any other path.
- **500** `internal server error` when a worker call still fails after its
  retries, or on an unexpected error. The details go only to the server log,
  with API keys removed.

A failing router never fails a request: when it gives no valid decision in 3
attempts, or its API fails with a permanent error, the turn falls back to the
first agent as solver and a warning is logged.

## Running the Conductor

```sh
export FUGU_CONDUCTOR_MODEL=openai/my-conductor
export FUGU_CONDUCTOR_BASE_URL=https://conductor.example.com/v1
export FUGU_CONDUCTOR_API_KEY=...
export FUGU_BASE_URL=https://gateway.example.com/v1
export FUGU_API_KEY=...

python -m openfugu.ultra \
  --query "Write a Python function that checks whether a string is a palindrome, and test it." \
  --slot-models "openai/model-a,openai/model-b,openai/model-c"
```

It prints the workflow, a line for each step and the final answer (long
texts are cut short). The exit status is 0 on success, 1 when the
configuration is invalid, the Conductor call fails, the workflow is invalid
(no worker runs) or a worker call fails, and 2 for invalid options.

## Tests

```sh
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest
python -m pyflakes openfugu tests
```

The tests make no model calls: litellm's `mock_response` answers every call,
and `FUGU_*` variables set on the machine are hidden from them. The first run
in a new environment still needs the tokenizer download described under
Install. `tests/test_fine_tuning_doc.py` fails when
[docs/FINE_TUNING_FORMAT.md](docs/FINE_TUNING_FORMAT.md) no longer matches the
prompts in the code.

## Known limitations

- **No authentication.** The server has no authentication, TLS, rate limiting
  or limit on concurrent requests (each request runs on its own thread). It
  listens on 127.0.0.1 by default; to expose it, put it behind a reverse proxy
  that handles authentication, TLS, request size and rate limits.
- **Only the last user message is used.** System messages, earlier turns and
  assistant messages are ignored. Of a content-part list, only the text parts
  are used, joined by line breaks. A request without a user message runs on an
  empty query. Other request fields, such as `temperature`, `max_tokens`,
  `tools` and `stream`, are ignored: every response is a single non-streaming
  completion, and `usage` reports only `fugu_turns`.
- **No server-side request timeout.** A request runs until a verifier accepts
  or the turns run out. A client that disconnects does not stop it. Set the
  reverse proxy's timeout from the worst case below.
- **Worst-case latency per request.** Each turn makes up to 3 router calls of
  at most 30 s each, plus up to 3 s of backoff (93 s), and one worker call of
  up to 3 attempts of 120 s each, plus up to 3 s of backoff (363 s): 456 s per
  turn, so 2,280 s (38 minutes) with the default 5 turns. A worker call that
  fails all 3 attempts ends the request with a 500 instead. These figures
  assume each call is cut off at its timeout.

## Project layout

| Path | Contents |
|---|---|
| `openfugu/llm.py` | Endpoint settings, startup checks, the retrying litellm call and the worker pool |
| `openfugu/mini.py` | The router (`ApiRouter`) and the turn-by-turn `Coordinator` |
| `openfugu/serve.py` | The HTTP server |
| `openfugu/ultra.py` | Conductor prompt, workflow parsing, validation and execution, and its command line |
| `tests/` | The test suite |
| `docs/FINE_TUNING_FORMAT.md` | Training-data format for the router and the Conductor |
| `docs/history/` | Background on the original research code |
| `openspec/specs/` | Behaviour specifications |

## Original research code

The original research code, with the local hidden-state router, training
scripts and evaluations, is at the tag
[`pre-api-only`](https://github.com/onezox/OpenFugu/tree/pre-api-only).

## License

Apache-2.0 ([LICENSE](LICENSE)). Attribution is in [NOTICE](NOTICE).
