#!/usr/bin/env python3
# OpenFugu — Apache-2.0. Part of an independent, open reimplementation of
# the Fugu orchestrator. NOT affiliated with Sakana AI. See NOTICE.
# Reference: end-to-end serving proof. Boots serve.py with the TRAINED per-step
# head + a REAL local worker pool, POSTs a GSM8K question to the OpenAI-compatible
# endpoint, and asserts a real worker-produced numeric answer. Original code.
"""
serve_e2e.py — end-to-end proof that Fugu serves as one model, for real.

This is the honest "end to end": it does NOT call Coordinator.run directly. It
boots the actual HTTP server (trained head + local pool), waits for readiness,
issues a real POST to /v1/chat/completions, and checks the answer came back
through the full per-step loop, produced by a real local worker (not MockWorker).

  python serve_e2e.py \
    --model <qwen3-0.6b dir> --vector model_iter_60.npy --head trinity_perstep.npy \
    --local-models "<llama path>,<gemma path>" --port 8099
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, time, urllib.request


def numeric_answer(text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", (text or "").replace(",", ""))
    return nums[-1] if nums else None


def wait_ready(port, proc, timeout=600):
    url = f"http://127.0.0.1:{port}/health"
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"[e2e] server exited early (code {proc.returncode})")
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(2)
    raise SystemExit("[e2e] server did not become ready in time")


def post_chat(port, question, timeout=600):
    body = json.dumps({"messages": [{"role": "user", "content": question}]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())
def main():
    ap = argparse.ArgumentParser(description="End-to-end serving proof (trained head + local pool).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--vector", default="model_iter_60.npy")
    ap.add_argument("--head", default=None, help="trained head-only (10240) vector")
    ap.add_argument("--local-models", required=True, metavar="CSV")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--serve-script", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "openfugu", "serve.py"))
    args = ap.parse_args()

    # a GSM8K question the local pool reliably solves
    Q = ("Natalia sold clips to 48 of her friends in April, and then she sold "
         "half as many clips in May. How many clips did she sell altogether in "
         "April and May?")
    GOLD = "72"

    cmd = [sys.executable, args.serve_script, "--model", args.model,
           "--vector", args.vector, "--local-models", args.local_models,
           "--port", str(args.port), "--max-turns", str(args.max_turns)]
    if args.head:
        cmd += ["--head", args.head]
    print(f"[e2e] booting server: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)

    # stream server stdout so we can see pool/head load + confirm NOT mock
    import threading
    server_lines = []
    def pump():
        for line in proc.stdout:
            server_lines.append(line.rstrip())
            print("    " + line.rstrip(), flush=True)
    threading.Thread(target=pump, daemon=True).start()

    try:
        wait_ready(args.port, proc)
        print("[e2e] server ready; POSTing a real GSM8K question ...", flush=True)
        t0 = time.time()
        resp = post_chat(args.port, Q)
        dt = time.time() - t0

        content = resp["choices"][0]["message"]["content"]
        turns = resp.get("usage", {}).get("fugu_turns", 0)
        got = numeric_answer(content)
        is_mock = any("MOCK" in l for l in server_lines)
        is_local = any("LOCAL" in l for l in server_lines)

        print(f"\n[e2e] answer={got!r} (gold {GOLD}) turns={turns} latency={dt:.1f}s", flush=True)
        print(f"[e2e] pool: local={is_local} mock={is_mock}", flush=True)
        print(f"[e2e] content[:200]={content[:200]!r}", flush=True)

        ok = (got == GOLD) and (turns > 0) and is_local and not is_mock
        if ok:
            print("\nPASS — real request answered correctly through the per-step loop "
                  "over a real local worker pool (not mock)")
            return 0
        print("\nFAIL — see above (answer/turns/pool check failed)")
        return 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
