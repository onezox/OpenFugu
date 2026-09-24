#!/usr/bin/env python3
# OpenFugu — Apache-2.0. Part of an independent, open reimplementation of
# the Fugu orchestrator. NOT affiliated with Sakana AI. See NOTICE.
# Reference: TRINITY (arXiv:2512.04695). Self-trains the coordinator via
# sep-CMA-ES on REAL data (GSM8K) with a real worker pool — the real-data
# upgrade of train_trinity.py's mock loop. Original code.
"""
train_trinity_real.py — TRINITY self-training on REAL GSM8K with a real pool.

Same sep-CMA-ES loop as train_trinity.py, but everything mock is now real:
  - tasks   : GSM8K questions (openai/gsm8k), answer = the number after '####'
  - features: a REAL Qwen3-0.6B penultimate hidden state of the question
  - workers : a real pool via litellm (Novita), differing models
  - reward  : numeric-answer match (the verifiable signal sep-CMA-ES optimizes)

The coordinator (bias-free linear head over the hidden state) learns which
worker to send each question to, to maximize solved rate. Minimal scale by
default so it runs cheaply; scale up via flags. Goal: beat random routing.

ponytail: reuses the ask/tell structure; no new framework, no sandbox — GSM8K
reward is just a number compare.
"""
from __future__ import annotations
import argparse, os, re, sys
import numpy as np

HIDDEN = 1024
HIDDEN_POS = -2


def numeric_answer(text: str):
    """Last integer/decimal in the text (GSM8K answers are numbers)."""
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return nums[-1] if nums else None


def gold_answer(ans_field: str):
    return ans_field.split("####")[-1].strip().replace(",", "")


class Backbone:
    """Real Qwen3-0.6B -> penultimate hidden state of a question (the router feature)."""
    def __init__(self, model_dir, device=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
        if device:
            self.model.to(device)
        self.device = next(self.model.parameters()).device
        self._cache = {}

    def feature(self, question: str) -> np.ndarray:
        if question in self._cache:
            return self._cache[question]
        torch = self.torch
        ids = self.tok(f"user: {question}", return_tensors="pt").to(self.device)
        with torch.no_grad():
            h = self.model.model(**ids).last_hidden_state[0, HIDDEN_POS, :]
        v = h.float().cpu().numpy()
        self._cache[question] = v
        return v


def route(head_vec, feat, n_workers):
    """Bias-free linear head -> worker id (argmax), faithful to mini.py."""
    W = head_vec.reshape(n_workers, HIDDEN)
    return int(np.argmax(W @ feat))


def main():
    ap = argparse.ArgumentParser(description="TRINITY self-train on real GSM8K (minimal).")
    ap.add_argument("--model", default=os.environ.get("FUGU_MODEL", "Qwen/Qwen3-0.6B"))
    ap.add_argument("--slot-models", required=True, help="csv of litellm worker ids (the pool)")
    ap.add_argument("--n-train", type=int, default=12, help="GSM8K questions (kept small/cheap)")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--sigma0", type=float, default=0.5)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="trinity_gsm8k.npy")
    args = ap.parse_args()

    import cma, litellm
    from datasets import load_dataset

    workers = args.slot_models.split(",")
    n_workers = len(workers)
    api_key = os.environ.get("FUGU_API_KEY") or os.environ.get("NOVITA_API_KEY") or os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("FUGU_BASE_URL") or os.environ.get("OPENAI_BASE_URL")

    ds = load_dataset("openai/gsm8k", "main", split=f"train[:{args.n_train}]")
    tasks = [(r["question"], gold_answer(r["answer"])) for r in ds]
    print(f"[real-train] {len(tasks)} GSM8K tasks, {n_workers} workers: {workers}", flush=True)

    bb = Backbone(args.model)
    feats = [bb.feature(q) for q, _ in tasks]      # cache real hidden states once
    print(f"[real-train] cached {len(feats)} real Qwen3-0.6B features (dim {feats[0].shape[0]})", flush=True)

    # worker call cache: (worker, question) -> solved? so CMA candidates reuse answers
    solve_cache: dict = {}
    def worker_solves(wid, q, gold):
        key = (wid, q)
        if key in solve_cache:
            return solve_cache[key]
        try:
            kw = dict(model=workers[wid],
                      messages=[{"role": "user",
                                 "content": q + "\nGive the final numeric answer at the end."}],
                      max_tokens=args.max_tokens, temperature=0.0)
            if api_key: kw["api_key"] = api_key
            if api_base: kw["api_base"] = api_base
            out = litellm.completion(**kw).choices[0].message.content or ""
            ok = 1.0 if numeric_answer(out) == gold else 0.0
        except Exception as e:
            print(f"   [warn] worker {wid} call failed: {str(e)[:60]}", flush=True)
            ok = 0.0
        solve_cache[key] = ok
        return ok

    def fitness(head_vec):
        tot = 0.0
        for (q, gold), feat in zip(tasks, feats):
            wid = route(head_vec, feat, n_workers)
            tot += worker_solves(wid, q, gold)
        return tot / len(tasks)

    # baseline: each worker alone + random (uses the same cache -> cheap)
    rng = np.random.default_rng(args.seed)
    per_worker = []
    for w in range(n_workers):
        per_worker.append(np.mean([worker_solves(w, q, g) for q, g in tasks]))
    best_single = max(per_worker)
    print("[baseline] per-worker solved rate: " +
          ", ".join(f"{workers[w].split('/')[-1]}={per_worker[w]:.2f}" for w in range(n_workers)), flush=True)

    # sep-CMA-ES over the head (SVF frozen for this minimal real run)
    dim = n_workers * HIDDEN
    es = cma.CMAEvolutionStrategy(np.zeros(dim), args.sigma0,
                                  {"seed": args.seed, "verbose": -9, "CMA_diagonal": True})
    best_vec, best_fit = None, -1.0
    for it in range(args.iters):
        cands = es.ask()
        fits = [fitness(c) for c in cands]
        es.tell(cands, [-f for f in fits])
        i = int(np.argmax(fits))
        if fits[i] > best_fit:
            best_fit, best_vec = fits[i], cands[i].copy()
        print(f"[iter {it}] best_solved={best_fit:.3f}  "
              f"(best single worker {best_single:.3f})  cache={len(solve_cache)}", flush=True)

    np.save(args.out, best_vec)
    print(f"\n[result] coordinator solved {best_fit:.3f} vs best single worker {best_single:.3f}")
    print(f"[result] learned routing per task:")
    for (q, g), feat in zip(tasks, feats):
        w = route(best_vec, feat, n_workers)
        print(f"   -> {workers[w].split('/')[-1]:24s} | {q[:50]}")
    if best_fit >= best_single:
        print("PASS — sep-CMA-ES self-trained TRINITY on REAL GSM8K, "
              "coordinator >= best single worker")


if __name__ == "__main__":
    main()
