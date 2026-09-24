#!/usr/bin/env python3
# OpenFugu — Apache-2.0. Part of an independent, open reimplementation of
# the Fugu orchestrator. NOT affiliated with Sakana AI. See NOTICE.
# Reference: TRINITY (arXiv:2512.04695). sep-CMA-ES router training on
# nvidia/ToolScale, reusing the verifiable tool-call reward from
# toolscale_data.py. Original code.
"""
train_trinity_toolscale.py — TRINITY router self-training on ToolScale.

GSM8K/AIME are single-domain (all math), so a coordinator has little to exploit
— every worker is roughly equally (un)able. nvidia/ToolScale is MULTI-DOMAIN
agentic tool-use, where different workers are good at different task types, which
is exactly the complementarity a router monetizes.

Same sep-CMA-ES loop as train_trinity_real.py, but:
  - tasks  : ToolScale (real agentic tool-call tasks)
  - reward : REUSED from toolscale_data.py (_parse_plan + _score against each
             task's evaluation_criteria.actions) — no new reward written
  - feature: real Qwen3-0.6B penultimate hidden state of the task
  - workers: real litellm pool (Novita), worker-call cached

Goal: coordinator >= best single worker by routing each task to the worker that
handles its tool-call structure best.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

HIDDEN = 1024
HIDDEN_POS = -2

# reuse the ToolScale reward + prompt verbatim (no reward duplication)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toolscale_data import _expected_actions, _parse_plan, _score, SYSTEM


class Backbone:
    """Real Qwen3-0.6B -> penultimate hidden state of a task (router feature)."""
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

    def feature(self, text: str) -> np.ndarray:
        if text in self._cache:
            return self._cache[text]
        torch = self.torch
        ids = self.tok(f"user: {text}", return_tensors="pt").to(self.device)
        with torch.no_grad():
            h = self.model.model(**ids).last_hidden_state[0, HIDDEN_POS, :]
        v = h.float().cpu().numpy()
        self._cache[text] = v
        return v


def route(head_vec, feat, n_workers):
    W = head_vec.reshape(n_workers, HIDDEN)
    return int(np.argmax(W @ feat))


def main():
    ap = argparse.ArgumentParser(description="TRINITY router training on ToolScale.")
    ap.add_argument("--model", default=os.environ.get("FUGU_MODEL", "Qwen/Qwen3-0.6B"))
    ap.add_argument("--slot-models", required=True, help="csv of litellm worker ids")
    ap.add_argument("--n-train", type=int, default=16)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--sigma0", type=float, default=0.5)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="trinity_toolscale.npy")
    args = ap.parse_args()

    import cma, litellm
    from datasets import load_dataset

    workers = args.slot_models.split(",")
    n_workers = len(workers)
    api_key = os.environ.get("FUGU_API_KEY") or os.environ.get("NOVITA_API_KEY") or os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("FUGU_BASE_URL") or os.environ.get("OPENAI_BASE_URL")

    ds = load_dataset("nvidia/ToolScale", split="train").shuffle(seed=args.seed)
    tasks = []
    for r in ds:
        us = r.get("user_scenario") or {}
        instr = (us.get("instructions") or {})
        q = instr.get("task_instructions") or instr.get("reason_for_call") or ""
        gold = _expected_actions(r.get("evaluation_criteria"))
        if q and gold:                                  # only tasks with a learnable target
            tasks.append((q, gold))
        if len(tasks) >= args.n_train:
            break
    print(f"[toolscale-train] {len(tasks)} tasks, {n_workers} workers: {workers}", flush=True)

    bb = Backbone(args.model)
    feats = [bb.feature(q) for q, _ in tasks]
    print(f"[toolscale-train] cached {len(feats)} real Qwen3-0.6B features", flush=True)

    solve_cache: dict = {}
    def worker_score(wid, q, gold):
        key = (wid, q)
        if key in solve_cache:
            return solve_cache[key]
        try:
            kw = dict(model=workers[wid],
                      messages=[{"role": "system", "content": SYSTEM},
                                {"role": "user", "content": f"USER QUESTION: {q}"}],
                      max_tokens=args.max_tokens, temperature=0.0)
            if api_key: kw["api_key"] = api_key
            if api_base: kw["api_base"] = api_base
            out = litellm.completion(**kw).choices[0].message.content or ""
            pred = _parse_plan(out)
            s = 0.0 if pred is None else _score(pred, gold)
        except Exception as e:
            print(f"   [warn] worker {wid} failed: {str(e)[:60]}", flush=True)
            s = 0.0
        solve_cache[key] = s
        return s

    def fitness(head_vec):
        return float(np.mean([worker_score(route(head_vec, f, n_workers), q, g)
                              for (q, g), f in zip(tasks, feats)]))

    per_worker = [np.mean([worker_score(w, q, g) for q, g in tasks]) for w in range(n_workers)]
    best_single = max(per_worker)
    print("[baseline] per-worker action-score: " +
          ", ".join(f"{workers[w].split('/')[-1]}={per_worker[w]:.3f}" for w in range(n_workers)), flush=True)

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
        print(f"[iter {it}] best_score={best_fit:.3f}  "
              f"(best single {best_single:.3f})  cache={len(solve_cache)}", flush=True)

    np.save(args.out, best_vec)
    print(f"\n[result] coordinator {best_fit:.3f} vs best single worker {best_single:.3f}")
    # learned routing distribution
    from collections import Counter
    routed = Counter(workers[route(best_vec, f, n_workers)].split("/")[-1] for f in feats)
    print(f"[result] routing distribution: {dict(routed)}")
    if best_fit > best_single + 0.01:
        print("PASS — coordinator beats best single worker on multi-domain ToolScale "
              "(orchestration value from worker complementarity)")
    elif best_fit >= best_single:
        print("TIE — coordinator matches best single worker")
    else:
        print("BELOW — coordinator under best single (insufficient routing signal)")


if __name__ == "__main__":
    main()
