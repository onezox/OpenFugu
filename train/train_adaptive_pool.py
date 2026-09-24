#!/usr/bin/env python3
# OpenFugu — Apache-2.0. Part of an independent, open reimplementation of
# the Fugu orchestrator. NOT affiliated with Sakana AI. See NOTICE.
# Reference: Conductor adaptive worker selection (arXiv:2512.04388 §3.2, Sakana
# AI). The paper implements this NOT as a separate engine module but by data/
# prompt modification: per question, restrict the Conductor to a random k-of-n
# subset of the worker pool. Original mock reconstruction; no Sakana code copied.
"""
train_adaptive_pool.py — generalize the coordinator to ARBITRARY worker subsets.

Why this matters: a coordinator trained on a fixed pool learns fixed favorites
("always send code to worker 0"). But fugu's product promise — opt out any
provider, dodge export controls, swap the pool — requires the coordinator to
work over WHATEVER subset is offered at inference time. The paper achieves this
by training on randomly sampled k-of-n subsets per question, so the policy
learns "pick the best AVAILABLE worker for this task", conditioned on the
offered set, instead of a static mapping.

Mock-first (no GPU/API): workers have per-domain skill; each episode offers a
random k-subset that MAY EXCLUDE the domain specialist. The coordinator sees the
task feature AND a mask of which workers are available, and must route to the
best one present. We show the subset-conditioned policy beats a fixed-favorite
policy when evaluated on held-out random subsets — i.e. it generalizes.
"""
from __future__ import annotations
import argparse
import numpy as np

N_DOMAINS = 4
N_WORKERS = 6           # bigger pool than domains, so subsets meaningfully vary
FEAT_DIM = 12


class World:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.domain_feat = rng.standard_normal((N_DOMAINS, FEAT_DIM))
        # competence[w,d]; each domain has a clear best worker, but several decent ones
        comp = rng.uniform(0.1, 0.4, (N_WORKERS, N_DOMAINS))
        for d in range(N_DOMAINS):
            order = rng.permutation(N_WORKERS)
            comp[order[0], d] = rng.uniform(0.88, 0.97)   # specialist
            comp[order[1], d] = rng.uniform(0.6, 0.75)    # decent backup
        self.comp = comp
        self.rng = rng

    def task(self, rng):
        d = int(rng.integers(N_DOMAINS))
        return d, self.domain_feat[d] + 0.15 * rng.standard_normal(FEAT_DIM)

    def subset(self, rng, k):
        m = np.zeros(N_WORKERS)
        m[rng.choice(N_WORKERS, k, replace=False)] = 1.0
        return m

    def best_available(self, domain, mask):
        c = self.comp[:, domain].copy()
        c[mask == 0] = -1
        return int(np.argmax(c))


# ---- two policies ----------------------------------------------------------
def fixed_policy(head, feat, mask):
    """Subset-BLIND: ignores availability, routes by feature only (the failure
    mode of fixed-pool training)."""
    W = head.reshape(N_WORKERS, FEAT_DIM)
    return int(np.argmax(W @ feat))


def adaptive_policy(head, feat, mask):
    """Subset-CONDITIONED: scores workers from the feature, then masks out the
    unavailable ones before argmax — picks the best AVAILABLE worker."""
    W = head.reshape(N_WORKERS, FEAT_DIM)
    logits = W @ feat
    logits[mask == 0] = -1e9
    return int(np.argmax(logits))


def evaluate(head, world, policy, n, seed, k):
    rng = np.random.default_rng(seed)
    tot = 0.0
    for _ in range(n):
        d, feat = world.task(rng)
        mask = world.subset(rng, k)
        w = policy(head, feat, mask)
        if mask[w] == 0:                          # routed to an unavailable worker -> fail
            continue
        tot += 1.0 if rng.random() < world.comp[w, d] else 0.0
    return tot / n


def train(world, policy, k=3, iters=50, seed=42):
    """Gradient-free (CEM) over the head, evaluated on RANDOM subsets each gen —
    this is what teaches subset generalization."""
    rng = np.random.default_rng(seed)
    dim = N_WORKERS * FEAT_DIM
    mu, sigma = np.zeros(dim), 0.5 * np.ones(dim)
    for it in range(iters):
        pop = rng.normal(mu, sigma, (20, dim))
        fits = [evaluate(p, world, policy, 96, seed + it * 100 + i, k) for i, p in enumerate(pop)]
        elite = np.argsort(fits)[-5:]
        mu = pop[elite].mean(0)
        sigma = np.maximum(0.05, pop[elite].std(0))
    return mu


def main():
    ap = argparse.ArgumentParser(description="Adaptive k-of-n pool training (mock).")
    ap.add_argument("--k", type=int, default=3, help="workers offered per question")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    world = World(seed=args.seed)
    print(f"[adaptive-pool] pool={N_WORKERS} workers, offering random k={args.k} per task; "
          f"specialist may be ABSENT", flush=True)

    # train both policies on random subsets, eval on held-out random subsets
    head_fixed = train(world, fixed_policy, k=args.k, iters=args.iters, seed=args.seed)
    head_adapt = train(world, adaptive_policy, k=args.k, iters=args.iters, seed=args.seed)

    EV = 4000
    fixed = evaluate(head_fixed, world, fixed_policy, EV, args.seed + 7777, args.k)
    adapt = evaluate(head_adapt, world, adaptive_policy, EV, args.seed + 7777, args.k)
    # oracle: always best-available worker
    rng = np.random.default_rng(args.seed + 7777)
    orc = 0.0
    for _ in range(EV):
        d, _f = world.task(rng); m = world.subset(rng, args.k)
        w = world.best_available(d, m)
        orc += 1.0 if rng.random() < world.comp[w, d] else 0.0
    orc /= EV

    print(f"\n[held-out random subsets, k={args.k}]")
    print(f"  subset-blind  policy reward = {fixed:.3f}")
    print(f"  subset-aware  policy reward = {adapt:.3f}")
    print(f"  oracle (best-available)     = {orc:.3f}")
    gain = (adapt - fixed) / max(fixed, 1e-6) * 100
    print(f"[result] subset-conditioned routing beats subset-blind by {gain:+.0f}% "
          f"and reaches {adapt/orc*100:.0f}% of oracle")
    if adapt > fixed + 0.03 and adapt > 0.8 * orc:
        print("PASS — coordinator generalizes to arbitrary worker subsets "
              "(the basis for swap-the-pool / opt-out-any-provider)")


if __name__ == "__main__":
    main()
