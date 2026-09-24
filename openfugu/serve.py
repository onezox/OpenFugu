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
import argparse, json, sys, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from openfugu.llm import ConfigError, check_endpoint, router_endpoint
from openfugu.mini import ApiRouter, Coordinator, LiteLLMWorker

ROUTER: ApiRouter | None = None
WORKER = None
MODEL_NAME = "fugu"
MAX_TURNS = 5


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


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

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
            coord = Coordinator(ROUTER, WORKER, max_turns=MAX_TURNS)
            res = coord.run(query)
            self._send(200, _chat_response(res.final, req.get("model", MODEL_NAME),
                                           len(res.turns)))
        except Exception as e:
            self._send(500, {"error": str(e)})

    def log_message(self, *a):       # quiet
        pass


def main():
    global ROUTER, WORKER, MAX_TURNS
    ap = argparse.ArgumentParser(description="Serve Fugu as one OpenAI-compatible model.")
    ap.add_argument("--slot-models", metavar="CSV", required=True,
                    help="comma-separated litellm worker model ids, one per agent slot")
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--max-turns", type=int, default=5)
    args = ap.parse_args()
    MAX_TURNS = args.max_turns
    slot_models = args.slot_models.split(",")

    try:
        endpoint = router_endpoint()
        check_endpoint(endpoint, "router")
    except ConfigError as exc:
        sys.exit(f"[serve] configuration error: {exc}")
    ROUTER = ApiRouter(endpoint, slot_models)
    print(f"[serve] router: {endpoint.model}", flush=True)

    WORKER = LiteLLMWorker(slot_models=slot_models)
    print(f"[serve] worker pool: litellm ({len(slot_models)} slots)", flush=True)

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"[serve] Fugu listening on :{args.port} — POST /v1/chat/completions", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
