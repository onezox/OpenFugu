#!/usr/bin/env python3
# OpenFugu — Apache-2.0. Part of an independent, open reimplementation of
# the Fugu orchestrator. NOT affiliated with Sakana AI. See NOTICE.
# Reference: TRINITY (arXiv:2512.04695) & Conductor (arXiv:2512.04388), Sakana AI.
# Quantifies the papers' central claim — a learned coordinator over a worker pool
# beats the best single worker — using OUR self-trained coordinator. Original
# code; mock harness so it runs with no GPU / no API.
"""
eval_orchestration.py — does orchestration actually beat the best single model?

Both Fugu papers headline the same result (their Table 1): a learned coordinator
routing over a pool scores ABOVE any individual worker. This evaluates exactly
that claim on the same mock world train_trinity.py trains in, comparing five
strategies on a held-out task set:

  1. each worker used ALONE (every task -> that one worker)
  2. random routing            (pick a worker uniformly at random)
  3. our TRAINED coordinator   (trinity_mock.npy, if present; else train fresh)
  4. oracle routing            (always the per-task specialist) = ceiling

The headline number: trained-coordinator reward vs the best single worker's
reward. If the coordinator wins, orchestration > any single model — the whole
premise of Fugu, reproduced end to end on something we trained ourselves.
"""
from __future__ import annotations
import argparse
import os
import numpy as np

# reuse the exact mock world + routing from the TRINITY trainer (lives in ../train)
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "train"))   # repo-root/train
sys.path.insert(0, _HERE)                                  # fallback: same dir
from train_trinity import MockWorld, route, train, N_DOMAINS, N_WORKERS


def eval_single_worker(world, worker_id, n_tasks, seed):
    """Every task goes to one fixed worker — its reward is its mean competence
    over the task distribution (uniform over domains)."""
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(n_tasks):
        domain, _feat = world.sample_task(rng)
        total += world.solve(domain, worker_id, rng)
    return total / n_tasks


def eval_random_routing(world, n_tasks, seed):
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(n_tasks):
        domain, _feat = world.sample_task(rng)
        worker = int(rng.integers(N_WORKERS))
        total += world.solve(domain, worker, rng)
    return total / n_tasks


def eval_coordinator(world, head_vec, n_tasks, seed):
    """Our trained coordinator: feature -> worker via the learned head."""
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(n_tasks):
        domain, feat = world.sample_task(rng)
        worker, _role = route(head_vec, feat)
        total += world.solve(domain, worker, rng)
    return total / n_tasks


def eval_oracle(world, n_tasks, seed):
    """Ceiling: always route a task to its specialist (domain d -> worker d)."""
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(n_tasks):
        domain, _feat = world.sample_task(rng)
        worker = domain if domain < N_WORKERS else 0
        total += world.solve(domain, worker, rng)
    return total / n_tasks


def main():
    ap = argparse.ArgumentParser(description="Orchestration-beats-single-model eval (mock).")
    ap.add_argument("--coordinator", default="trinity_mock.npy",
                    help="trained coordinator vector; trained fresh if missing")
    ap.add_argument("--n-tasks", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--world-seed", type=int, default=42,
                    help="MUST match the coordinator's training world (train_trinity default 42)")
    ap.add_argument("--train-iters", type=int, default=60,
                    help="iters if we must train a coordinator fresh")
    args = ap.parse_args()

    # The coordinator is tied to the world it trained in (its head maps THAT
    # world's domain feature-signatures -> workers). Evaluate in the SAME world
    # (default training seed 42); generalization is tested via held-out TASK
    # samples (fresh rng below), not by swapping the whole world.
    world = MockWorld(seed=args.world_seed)

    # get a trained coordinator: load if present, else self-train now
    if os.path.exists(args.coordinator):
        head = np.load(args.coordinator)
        print(f"[eval] loaded trained coordinator: {args.coordinator}", flush=True)
        print(f"[eval] world seed {args.world_seed} (must match the coordinator's "
              f"training world)", flush=True)
    else:
        print(f"[eval] no {args.coordinator}; self-training one via sep-CMA-ES ...", flush=True)
        head, _fit = train(world, num_iters=args.train_iters, out=args.coordinator)

    n, s = args.n_tasks, args.seed

    # 1. each worker alone
    single = [eval_single_worker(world, w, n, s) for w in range(N_WORKERS)]
    best_single = max(single)
    best_worker = int(np.argmax(single))

    # 2-4. random / coordinator / oracle
    rand = eval_random_routing(world, n, s)
    coord = eval_coordinator(world, head, n, s)
    oracle = eval_oracle(world, n, s)

    print("\n=== held-out evaluation (%d tasks, seed %d) ===" % (n, s))
    for w in range(N_WORKERS):
        star = "  <- best single" if w == best_worker else ""
        print(f"  worker {w} alone        : {single[w]:.3f}{star}")
    print(f"  random routing         : {rand:.3f}")
    print(f"  OUR coordinator        : {coord:.3f}")
    print(f"  oracle (specialist)    : {oracle:.3f}  (ceiling)")

    lift = (coord - best_single) / best_single * 100
    frac = coord / oracle * 100
    print(f"\n[result] coordinator {coord:.3f} vs best single worker {best_single:.3f} "
          f"(worker {best_worker})  ->  {lift:+.0f}%")
    print(f"[result] coordinator reaches {frac:.0f}% of the oracle ceiling")
    if coord > best_single + 0.02:
        print("PASS — orchestration beats the best single model "
              "(the central Fugu claim, on a coordinator we trained ourselves)")
    else:
        print("FAIL — coordinator did not beat the best single worker")


if __name__ == "__main__":
    main()
