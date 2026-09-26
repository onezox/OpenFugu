# OpenFugu — Apache-2.0. Part of an independent, open reimplementation of
# the Fugu orchestrator. NOT affiliated with Sakana AI. See NOTICE.
# Reference: OpenAI-compatible serving layer for the OpenFugu TRINITY coordinator. Original code.
"""
serve.py — Fugu as a single OpenAI-compatible model endpoint.

This is Fugu's real product surface: "one model to command them all". A client
POSTs to /v1/chat/completions as if calling one model; internally the TRINITY
coordinator asks an API router model which worker of the pool acts next, and
runs the step_trinity loop until a verifier accepts. The caller never sees the
pool.

stdlib http.server only — no FastAPI/uvicorn (ponytail: a router endpoint needs
a socket and a JSON handler, not a web framework).

Run:
  FUGU_CONDUCTOR_MODEL=... FUGU_API_KEY=... FUGU_BASE_URL=... \
  python -m openfugu.serve --slot-models <csv of litellm worker ids> --port 8088

Query:
  curl localhost:8088/v1/chat/completions -d '{"messages":[{"role":"user","content":"..."}]}'
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from openfugu.llm import (ConfigError, LLMCallError, WorkerPool, check_endpoint,
                          configure_logging, parse_slot_models, router_endpoint)
from openfugu.mini import MAX_TURNS, ApiRouter, Coordinator

logger = logging.getLogger("openfugu.serve")

MODEL_NAME = "fugu"
BIND_HOST = "0.0.0.0"


def _chat_response(text: str, model: str, usage_turns: int) -> dict:
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        # surface the orchestration depth without exposing which workers ran
        "usage": {"fugu_turns": usage_turns},
    }


class FuguServer(ThreadingHTTPServer):
    """HTTP server that holds the router, worker pool and turn budget for its handlers."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], router: ApiRouter, workers: WorkerPool,
                 max_turns: int = MAX_TURNS) -> None:
        super().__init__(address, Handler)
        self.router, self.workers, self.max_turns = router, workers, max_turns


class Handler(BaseHTTPRequestHandler):
    """Serves /v1/chat/completions, /v1/models and /health."""

    protocol_version = "HTTP/1.1"
    server: FuguServer

    def _send(self, code: int, body: dict):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/v1/models":
            self._send(200, {"object": "list", "data": [
                {"id": MODEL_NAME, "object": "model", "owned_by": "openfugu"}]})
        elif self.path in ("/health", "/"):
            self._send(200, {"status": "ok", "model": MODEL_NAME})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send(404, {"error": "not found"}); return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            messages = req.get("messages", [])
            if not messages:
                self._send(400, {"error": "messages required"}); return
            # the user query = last user message; coordinator runs the full loop
            query = next((m["content"] for m in reversed(messages)
                          if m.get("role") == "user"), "")
            coord = Coordinator(self.server.router, self.server.workers,
                                max_turns=self.server.max_turns)
            res = coord.run(query)
            self._send(200, _chat_response(res.final, req.get("model", MODEL_NAME),
                                           len(res.turns)))
        except LLMCallError as exc:            # expected: a model call failed after retries
            logger.error("POST /v1/chat/completions failed: %s", exc)
            self._send(500, {"error": "internal server error"})
        except Exception:
            logger.exception("POST /v1/chat/completions failed")
            self._send(500, {"error": "internal server error"})

    def log_message(self, format: str, *args) -> None:
        """Route http.server's access log to the logging module (DEBUG)."""
        logger.debug("%s - " + format, self.address_string(), *args)


def main(argv: list[str] | None = None) -> int:
    """Validate the configuration, then serve until interrupted."""
    ap = argparse.ArgumentParser(description="Serve Fugu as one OpenAI-compatible model.")
    ap.add_argument("--slot-models", metavar="CSV", required=True,
                    help="comma-separated litellm worker model ids, one per agent slot")
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS)
    args = ap.parse_args(argv)
    if not 0 < args.port < 65536:
        ap.error("--port must be between 1 and 65535")
    if args.max_turns < 1:
        ap.error("--max-turns must be at least 1")
    configure_logging(logging.INFO)

    try:
        slot_models = parse_slot_models(args.slot_models)
        workers = WorkerPool.from_env(slot_models)
        workers.check()
        endpoint = router_endpoint()
        check_endpoint(endpoint, "router")
    except ConfigError as exc:
        logger.error("configuration error: %s", exc)
        return 1
    router = ApiRouter(endpoint, slot_models)

    try:
        server = FuguServer((BIND_HOST, args.port), router, workers, args.max_turns)
    except OSError as exc:
        logger.error("cannot listen on %s:%d: %s", BIND_HOST, args.port, exc)
        return 1
    logger.info("router: %s", endpoint.model)
    logger.info("worker pool (%d): %s", len(workers), ", ".join(slot_models))
    logger.info("Fugu listening on %s:%d - POST /v1/chat/completions", BIND_HOST, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
