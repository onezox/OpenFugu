#!/usr/bin/env python3
# OpenFugu — Apache-2.0. Part of an independent, open reimplementation of
# the Fugu orchestrator. NOT affiliated with Sakana AI. See NOTICE.
# Reference: adaptive worker selection (arXiv:2512.04388) at TRINITY's per-STEP
# granularity. The REAL k-of-n experiment: per question a random subset of
# workers is OFFERED; the router is masked to it and must pick the best AVAILABLE
# worker, fitness = terminal reward of the full per-step Coordinator rollout.
# Original code. Supersedes the reverted per-question train_adaptive_pool_real.py.
"""
train_adaptive_pool_perstep.py — adaptive k-of-n routing, per-STEP, real workers.

The earlier train_adaptive_pool_real.py was wrong twice over: (1) per-QUESTION,
not per-step, and (2) its reward grafted worker-id semantics onto ToolScale tool
names, so subset_reward was always 0. This is the correct version:

  per question: offer a random k-of-n subset of the LOCAL worker pool
  router      : Qwen3-0.6B hidden -> head, MASKED to the offered subset each turn
  rollout     : full per-step Coordinator loop (mini.py) over the masked pool
  reward      : terminal — did the final answer match the GSM8K number
  train       : sep-CMA-ES over the head (SVF frozen)

We compare the trained subset-aware router against its OWN starting point: the
base head, evaluated under the SAME availability masking. Both honor the offered
subset at eval (the router is masked to it every turn, so neither can route to an
absent worker). The only difference is whether the head was *trained* over random
subsets. Subset-aware training should beat the untrained base because it learns,
within whatever subset is offered, to pick the worker most likely to solve.
"""
from __future__ import annotations
import argparse, os, re, sys, glob
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "openfugu"))
sys.path.insert(0, "/root")
import mini
from mini import FuguRouter, Coordinator, HIDDEN, HEAD_ROWS, N_AGENTS


def numeric_answer(text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", (text or "").replace(",", ""))
    return nums[-1] if nums else None
class LocalPoolWorker:
    """Worker pool of local multi-vendor models (same as the per-step trainer).
    The Coordinator calls (role_name, messages, agent_id) -> reply; we dispatch
    to model[agent_id % n]. Solver replies are nudged toward <think>…</think> so
    the Coordinator can extract a thought into the router obs."""
    def __init__(self, specs, max_new=384):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.max_new = torch, max_new
        self.names, self.toks, self.models, self.devs = [], [], [], []
        for name, path, dev in specs:
            tk = AutoTokenizer.from_pretrained(path)
            if tk.pad_token is None:
                tk.pad_token = tk.eos_token
            try:
                m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(dev).eval()
            except TypeError:
                m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(dev).eval()
            self.names.append(name); self.toks.append(tk); self.models.append(m); self.devs.append(dev)
        self.cache = {}

    def __call__(self, role_name, messages, agent_id):
        wid = agent_id % len(self.models)
        key = (wid, role_name, messages[-1]["content"][:200])
        if key in self.cache:
            return self.cache[key]
        torch = self.torch
        tk, model, dev = self.toks[wid], self.models[wid], self.devs[wid]
        try:
            text = tk.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = "\n".join(m["content"] for m in messages)
        ids = tk(text, return_tensors="pt", truncation=True, max_length=2048).to(dev)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=self.max_new, do_sample=False,
                                 pad_token_id=tk.pad_token_id)
        reply = tk.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        self.cache[key] = reply
        return reply


def main():
    ap = argparse.ArgumentParser(description="Adaptive k-of-n per-step router training (real rollout).")
    ap.add_argument("--router-model", default=os.environ.get("FUGU_MODEL", "Qwen/Qwen3-0.6B"))
    ap.add_argument("--vector", default="/root/model_iter_60.npy")
    ap.add_argument("--n-train", type=int, default=8)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--k", type=int, default=2, help="workers offered per question (random k-of-n)")
    ap.add_argument("--sigma0", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="adaptive_perstep.npy")
    args = ap.parse_args()

    import cma, torch
    from datasets import load_dataset

    HUB = "/vePFS-Mindverse/share/huggingface/hub"
    def snap(repo):
        g = glob.glob(f"{HUB}/{repo}/snapshots/*/")
        return g[0] if g else None
    POOL = [(n, snap(r), d) for n, r, d in [
        ("deepseek-distill-7b", "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B", "cuda:1"),
        ("llama-3.2-3b",        "models--meta-llama--Llama-3.2-3B-Instruct",        "cuda:2"),
        ("gemma-3-4b",          "models--google--gemma-3-4b-it",                    "cuda:3"),
    ] if snap(r)]
    n_w = len(POOL)
    print(f"[adaptive-perstep] pool ({n_w}): {[n for n,_,_ in POOL]}  k={args.k}", flush=True)

    ds = load_dataset("openai/gsm8k", "main", split=f"train[:{args.n_train}]")
    tasks = [(r["question"], r["answer"].split("####")[-1].strip().replace(",", "")) for r in ds]

    # per-question random offered subset (mask over the FIRST n_w agent slots) [CODE]
    rng = np.random.default_rng(args.seed)
    masks = []
    for _ in tasks:
        m = np.zeros(N_AGENTS, dtype=bool)
        m[rng.choice(n_w, size=min(args.k, n_w), replace=False)] = True
        masks.append(m)

    router = FuguRouter(args.router_model, args.vector, device="cuda:0", seed=args.seed)
    base_head = router.head.clone()
    pool = LocalPoolWorker(POOL)

    def rollout(head_vec):
        # always honor the offered subset (router masked every turn) — this is
        # the env constraint; we never route to an absent worker.
        router.head = torch.from_numpy(head_vec.copy()).float().reshape(HEAD_ROWS, HIDDEN).to(router.device)
        solved = 0
        for (q, gold), msk in zip(tasks, masks):
            coord = Coordinator(router, pool, max_turns=args.max_turns, sample=False,
                                agent_mask=msk)
            res = coord.run(q)
            if numeric_answer(res.final) == gold:
                solved += 1
        return solved / len(tasks)

    base_aware = rollout(base_head.cpu().numpy().ravel())
    print(f"[adaptive-perstep] base head, subset-masked rollout solved={base_aware:.3f}  (n={len(tasks)})", flush=True)

    es = cma.CMAEvolutionStrategy(base_head.cpu().numpy().ravel(), args.sigma0,
                                  {"seed": args.seed, "verbose": -9, "CMA_diagonal": True})
    best_vec, best_fit = base_head.cpu().numpy().ravel(), base_aware
    for it in range(args.iters):
        cands = es.ask()
        fits = [rollout(c) for c in cands]
        es.tell(cands, [-f for f in fits])
        i = int(np.argmax(fits))
        if fits[i] > best_fit:
            best_fit, best_vec = fits[i], cands[i].copy()
        print(f"[iter {it}] best_subset_aware={best_fit:.3f} (base {base_aware:.3f})", flush=True)

    np.save(args.out, best_vec)
    print(f"\n[result] subset-aware trained={best_fit:.3f}  vs base (subset-masked)={base_aware:.3f}")
    print(f"[result] saved {args.out}")
    if best_fit > base_aware + 0.01:
        print("PASS — sep-CMA over random k-of-n subsets improved subset-aware per-step routing")
    else:
        print(f"NOTE — no improvement over base in {args.iters} iters (small scale / saturated)")


if __name__ == "__main__":
    main()
