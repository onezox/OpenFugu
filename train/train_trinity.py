#!/usr/bin/env python3
# OpenFugu — Apache-2.0. Part of an independent, open reimplementation of
# the Fugu orchestrator. NOT affiliated with Sakana AI. See NOTICE.
# Reference: TRINITY (arXiv:2512.04695, Sakana AI). Self-trains the coordinator
# head + SVF offsets from scratch via sep-CMA-ES (Ros & Hansen 2008). The
# training main loop is NOT in any released Sakana code; this is an original
# reconstruction. Mock-first so the whole loop runs with no GPU / no API.
"""
train_trinity.py — self-train a TRINITY coordinator with sep-CMA-ES.

Unlike openfugu/mini.py (which loads Sakana's released model_iter_60.npy), this
trains the coordinator's parameters OURSELVES, gradient-free, the way the paper
describes: a population of candidate parameter vectors is sampled, each is scored
by the mean terminal reward of running the coordination loop over a task set, and
sep-CMA-ES recombines the best.

--mock (default): a fully synthetic, backbone-free, API-free harness that proves
the optimization loop converges — the coordinator must learn to route each task
DOMAIN to the worker that is actually good at it. Reward should climb from chance
toward optimal as CMA-ES finds the head that maps features -> the right worker.

Real-backbone / real-worker modes are layered on the same loop later; mock first.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

# ---- problem geometry (mirrors mini.py constants, scaled for the mock) ------
N_DOMAINS = 4              # task domains (math / code / facts / reasoning)
N_WORKERS = 4             # worker pool size (L)
N_ROLES = 3               # solver / thinker / verifier
FEAT_DIM = 16             # mock "hidden state" dimensionality (mini.py uses 1024)

# head maps feature -> (L workers + 3 roles) logits, bias-free, like TRINITY
HEAD_ROWS = N_WORKERS + N_ROLES
PARAM_DIM = HEAD_ROWS * FEAT_DIM          # the trainable vector (mock analogue of 19456)


# ---- mock world -------------------------------------------------------------
class MockWorld:
    """Synthetic verifiable tasks + a worker pool with differing per-domain skill.

    Each task has a domain. Each worker has a competence profile over domains
    (worker d is the specialist for domain d). A task is 'solved' (reward 1) with
    probability = the chosen worker's competence on that task's domain. The only
    way to maximize reward is to route each domain to its specialist — so a
    coordinator that learns feature->worker routing will climb toward optimal,
    and random routing sits at the mean competence (chance)."""
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        # fixed feature signature per domain (the "hidden state" a transcript yields)
        self.domain_feat = rng.standard_normal((N_DOMAINS, FEAT_DIM))
        # competence[w, d] in [0,1]; specialist worker d is best at domain d
        base = rng.uniform(0.15, 0.35, (N_WORKERS, N_DOMAINS))
        for d in range(min(N_WORKERS, N_DOMAINS)):
            base[d, d] = rng.uniform(0.85, 0.97)          # specialist
        self.competence = base
        self.rng = rng
        # optimal = always pick the specialist; chance = mean competence
        self.optimal = float(np.mean([self.competence[d, d] for d in range(N_DOMAINS)]))
        self.chance = float(self.competence.mean())

    def sample_task(self, rng):
        d = int(rng.integers(N_DOMAINS))
        # feature = domain signature + small noise (transcript variation)
        feat = self.domain_feat[d] + 0.15 * rng.standard_normal(FEAT_DIM)
        return d, feat

    def solve(self, domain, worker_id, rng):
        p = self.competence[worker_id % N_WORKERS, domain]
        return 1.0 if rng.random() < p else 0.0


# ---- coordinator: the thing we train ---------------------------------------
def route(head_vec, feat):
    """head_vec (PARAM_DIM,) -> (worker_id, role_id). Bias-free linear head, argmax.
    Faithful to mini.py: logits = W @ h, split into worker / role."""
    W = head_vec.reshape(HEAD_ROWS, FEAT_DIM)
    logits = W @ feat
    worker = int(np.argmax(logits[:N_WORKERS]))
    role = int(np.argmax(logits[N_WORKERS:]))
    return worker, role


def evaluate(head_vec, world: MockWorld, n_tasks, seed):
    """Mean terminal reward over n_tasks — this is fitness J(theta) (paper eq.3)."""
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(n_tasks):
        domain, feat = world.sample_task(rng)
        worker, _role = route(head_vec, feat)
        total += world.solve(domain, worker, rng)
    return total / n_tasks


# ---- sep-CMA-ES training loop (original reconstruction) ---------------------
def train(world, num_iters=60, sigma0=0.3, n_tasks=64, num_repeats=4,
          seed=42, out="trinity_mock.npy", diagonal=True):
    import cma
    x0 = np.zeros(PARAM_DIM)                               # identity-ish start [paper: offsets+1.0->0]
    opts = {"seed": seed, "verbose": -9}
    if diagonal:
        opts["CMA_diagonal"] = True                        # the "sep" in sep-CMA-ES
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    print(f"[trinity-train] PARAM_DIM={PARAM_DIM}  popsize={es.popsize}  "
          f"diagonal={diagonal}  chance={world.chance:.3f}  optimal={world.optimal:.3f}", flush=True)

    best_vec, best_fit = x0, -1.0
    for it in range(num_iters):
        cands = es.ask()
        fits = []
        for c in cands:
            # average over repeats to denoise the Bernoulli reward (paper: replication)
            r = np.mean([evaluate(c, world, n_tasks, seed + it * 1000 + j)
                         for j in range(num_repeats)])
            fits.append(r)
        es.tell(cands, [-f for f in fits])                 # pycma minimizes -> negate
        i = int(np.argmax(fits))
        if fits[i] > best_fit:
            best_fit, best_vec = fits[i], cands[i].copy()
        if it % 5 == 0 or it == num_iters - 1:
            gap = (best_fit - world.chance) / (world.optimal - world.chance + 1e-9)
            print(f"[iter {it:3d}] best_reward={best_fit:.3f}  "
                  f"(chance {world.chance:.3f} -> optimal {world.optimal:.3f}, "
                  f"{gap*100:.0f}% of the way)", flush=True)
    np.save(out, best_vec)
    print(f"[trinity-train] saved {out}  final_reward={best_fit:.3f}", flush=True)
    return best_vec, best_fit


def main():
    ap = argparse.ArgumentParser(description="Self-train a TRINITY coordinator (sep-CMA-ES, mock).")
    ap.add_argument("--mock", action="store_true", default=True)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--sigma0", type=float, default=0.3)
    ap.add_argument("--n-tasks", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-diagonal", action="store_true", help="use full CMA instead of sep")
    ap.add_argument("--out", default="trinity_mock.npy")
    args = ap.parse_args()

    world = MockWorld(seed=args.seed)
    vec, fit = train(world, num_iters=args.iters, sigma0=args.sigma0,
                     n_tasks=args.n_tasks, num_repeats=args.repeats, seed=args.seed,
                     out=args.out, diagonal=not args.no_diagonal)
    # report learned routing: did it find each domain's specialist?
    print("\n[learned routing] domain -> chosen worker (specialist is domain==worker):")
    correct = 0
    for d in range(N_DOMAINS):
        w, _ = route(vec, world.domain_feat[d])
        ok = (w == d)
        correct += ok
        print(f"  domain {d}: -> worker {w}  {'OK (specialist)' if ok else 'miss'}")
    print(f"[result] routed {correct}/{N_DOMAINS} domains to their specialist; "
          f"final reward {fit:.3f} vs chance {world.chance:.3f} / optimal {world.optimal:.3f}")
    if correct == N_DOMAINS:
        print("PASS — sep-CMA-ES self-trained the coordinator to optimal routing")


if __name__ == "__main__":
    main()
