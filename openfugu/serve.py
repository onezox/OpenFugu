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
a socket and a JSON handler, not a web framework). The server listens on
127.0.0.1 unless --host says otherwise, and has no authentication; put a
reverse proxy in front of it to expose it.

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
import socket
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from openfugu.llm import (ConfigError, LLMCallError, WorkerPool, check_endpoint,
                          configure_logging, parse_slot_models, router_endpoint)
from openfugu.mini import MAX_TURNS, ApiRouter, Coordinator

logger = logging.getLogger("openfugu.serve")

MODEL_NAME = "fugu"
DEFAULT_HOST = "127.0.0.1"
LINGER_S = 5.0          # longest wait for a client to finish sending before a close


class BadRequest(ValueError):
    """The request body is malformed; the message is returned as a 400 error."""


def _message_text(content: object) -> str:
    """A message's text: a string, or the text parts of an OpenAI content-part
    list joined by newlines (other part types, such as images, are skipped)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part["text"] for part in content
                         if isinstance(part, dict) and part.get("type") == "text"
                         and isinstance(part.get("text"), str))
    raise BadRequest("message content must be a string or a list of content parts")


def _user_query(request: object) -> str:
    """The text of the last user message; the coordinator answers this query."""
    if not isinstance(request, dict):
        raise BadRequest("request body must be a JSON object")
    messages = request.get("messages", [])
    if not messages:
        raise BadRequest("messages required")
    if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
        raise BadRequest("messages must be a list of message objects")
    user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    return _message_text(user.get("content")) if user else ""


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

    def shutdown_request(self, request: socket.socket) -> None:
        """Close a connection without losing the last response.

        Closing a socket that still holds unread input, such as a request body
        the handler did not read, makes TCP reset the connection, and the reset
        can destroy the response before the client reads it. So the server
        sends its FIN first, then reads and discards whatever the client still
        sends until the client closes or LINGER_S passes, and only then closes.
        """
        deadline = time.monotonic() + LINGER_S
        try:
            request.shutdown(socket.SHUT_WR)
            while (remaining := deadline - time.monotonic()) > 0:
                request.settimeout(remaining)
                if not request.recv(65536):
                    break
        except OSError:                                 # timed out, reset or already closed
            pass
        self.close_request(request)


class Handler(BaseHTTPRequestHandler):
    """Serves /v1/chat/completions, /v1/models and /health."""

    protocol_version = "HTTP/1.1"
    server: FuguServer

    def _send(self, code: int, body: dict, *, close: bool = False):
        """Send a JSON response. `close` ends the connection after it; a response
        must close when the request body was not read in full, or the unread
        bytes would be parsed as the next request on this keep-alive connection."""
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        if close:
            self.send_header("Connection", "close")      # also sets self.close_connection
        self.end_headers()
        self.wfile.write(data)

    def _has_body(self) -> bool:
        """Whether the request declares a body: chunked, or a Content-Length other than 0."""
        return ("Transfer-Encoding" in self.headers
                or self.headers.get("Content-Length", "0").strip() != "0")

    def do_GET(self):
        close = self._has_body()                        # a GET body is never read
        if self.path == "/v1/models":
            self._send(200, {"object": "list", "data": [
                {"id": MODEL_NAME, "object": "model", "owned_by": "openfugu"}]}, close=close)
        elif self.path in ("/health", "/"):
            self._send(200, {"status": "ok", "model": MODEL_NAME}, close=close)
        else:
            self._send(404, {"error": "not found"}, close=close)

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send(404, {"error": "not found"}, close=self._has_body()); return
        if "Transfer-Encoding" in self.headers:         # http.server cannot decode chunked bodies
            self._send(411, {"error": "Content-Length required; Transfer-Encoding is not "
                                      "supported"}, close=True); return
        try:
            # Several Content-Length headers make the body length ambiguous.
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) > 1:
                raise ValueError("several Content-Length headers")
            n = int(lengths[0]) if lengths else 0
            if n < 0:
                raise ValueError("negative Content-Length")
        except ValueError:
            self._send(400, {"error": "invalid JSON body"}, close=True); return
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            self._send(400, {"error": "invalid JSON body"}); return
        try:
            query = _user_query(req)      # last user message; the coordinator runs the full loop
        except BadRequest as exc:
            self._send(400, {"error": str(exc)}); return
        try:
            coord = Coordinator(self.server.router, self.server.workers,
                                max_turns=self.server.max_turns)
            res = coord.run(query)
        except LLMCallError as exc:            # expected: a model call failed after retries
            logger.error("POST /v1/chat/completions failed: %s", exc)
            self._send(500, {"error": "internal server error"}); return
        except Exception:
            logger.exception("POST /v1/chat/completions failed")
            self._send(500, {"error": "internal server error"}); return
        self._send(200, _chat_response(res.final, req.get("model", MODEL_NAME), len(res.turns)))

    def log_message(self, format: str, *args) -> None:
        """Route http.server's access log to the logging module (DEBUG)."""
        logger.debug("%s - " + format, self.address_string(), *args)


def main(argv: list[str] | None = None) -> int:
    """Validate the configuration, then serve until interrupted."""
    ap = argparse.ArgumentParser(description="Serve Fugu as one OpenAI-compatible model.")
    ap.add_argument("--slot-models", metavar="CSV", required=True,
                    help="comma-separated litellm worker model ids, one per agent slot")
    ap.add_argument("--host", default=DEFAULT_HOST,
                    help=f"IPv4 address or hostname to listen on (default {DEFAULT_HOST}; "
                         f"the server has no authentication, so expose it only behind a "
                         f"reverse proxy)")
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS)
    args = ap.parse_args(argv)
    if not args.host.strip():
        ap.error("--host must not be empty")
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
        server = FuguServer((args.host, args.port), router, workers, args.max_turns)
    except OSError as exc:
        logger.error("cannot listen on %s:%d: %s", args.host, args.port, exc)
        return 1
    logger.info("router: %s", endpoint.model)
    logger.info("worker pool (%d): %s", len(workers), ", ".join(slot_models))
    logger.info("Fugu listening on %s:%d - POST /v1/chat/completions", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
