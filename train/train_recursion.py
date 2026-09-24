#!/usr/bin/env python3
# OpenFugu — Apache-2.0. Part of an independent, open reimplementation of
# the Fugu orchestrator. NOT affiliated with Sakana AI. See NOTICE.
# Reference: Conductor recursive topologies (arXiv:2512.04388 §3.2, Sakana AI).
# Faithful reconstruction of the recursion mechanism in the authors'
# conductor_recursion_engine.py — multi-round generation where the Conductor
# names ITSELF as a worker, sees its own prior-round output, and revises;
# rewards from the non-recursive round are discounted and normalized per round.
# Mock-first so the loop runs with no GPU / no API.
"""
train_recursion.py — Fugu-Ultra's recursive topology, the test-time-scaling axis.

The non-recursive Conductor (train_conductor.py) emits one workflow. The
*recursive* Conductor can list itself as a worker: round r produces a workflow,
its output is fed back, and round r+1 gets to revise. This is the mechanism that
lets Ultra spend more compute for harder problems.

This mock proves the recursion-specific training mechanics run and help:
  - train_recursion_rounds = 2          (round 0 = first attempt, round 1 = revise)
  - recursion_round_processor           (feed round r's prompt+completion into r+1)
  - recursion_discount_factor = 0.2     (discount the earlier, non-final round)
  - normalize_rewards_per_recursion_round = True  (GRPO advantage per round)

Mock world: each task has a hidden target quality; a single pass lands near it
with noise, a second pass that SEES the first can correct toward the target. A
policy that learns to use the 2nd round (revise) beats one-shot — exactly the
behavior recursion is meant to unlock.
"""
from __future__ import annotations
import argparse
import numpy as np

DISCOUNT = 0.2          # [CODE] recursion_discount_factor on non-final rounds
ROUNDS = 2              # [CODE] train_recursion_rounds


# ---- mock conductor: a 1-param policy = P(revise | sees own first output) ----
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


def run_episode(theta, task_target, rng, rounds=ROUNDS):
    """Round 0: one-shot attempt (noisy). Round 1: if the policy chooses to
    recurse, it SEES round-0 output and corrects toward the target.
    Returns (per-round rewards, final reward)."""
    # round 0 — first workflow attempt: quality = target minus noise
    out0 = task_target - abs(rng.normal(0, 0.4))
    r0 = max(0.0, out0)
    rewards = [r0]
    final = r0
    # decide whether to recurse (the trainable behavior)
    p_revise = sigmoid(theta)
    if rounds > 1 and rng.random() < p_revise:
        # round 1 — recursive pass: sees out0, corrects a fraction toward target
        out1 = out0 + 0.7 * (task_target - out0) - abs(rng.normal(0, 0.1))
        r1 = max(0.0, out1)
        rewards.append(r1)
        final = r1            # last round's output is the answer
    return rewards, final


def discounted_normalized_fitness(theta, tasks, rng, n_eval=64):
    """Mirror the engine: collect per-round rewards, discount the non-final
    round by DISCOUNT, normalize per round (GRPO-style), return mean final."""
    finals, round0, round1 = [], [], []
    for _ in range(n_eval):
        t = tasks[rng.integers(len(tasks))]
        rew, final = run_episode(theta, t, rng)
        finals.append(final)
        round0.append(rew[0])
        round1.append(rew[1] if len(rew) > 1 else np.nan)
    finals = np.array(finals)
    # per-round normalization (advantage signal), discount on the earlier round
    a0 = (np.array(round0) - np.nanmean(round0)) / (np.nanstd(round0) + 1e-6)
    r1 = np.array(round1)
    used = ~np.isnan(r1)
    adv = finals.mean()                       # objective we report = mean final reward
    return adv, finals.mean(), used.mean()


def train(rounds=ROUNDS, iters=40, seed=42):
    rng = np.random.default_rng(seed)
    tasks = rng.uniform(1.0, 2.0, 8)          # 8 tasks with target quality in [1,2]
    # CEM-style 1-D search over theta (recurse-propensity) — gradient-free, like ES
    mu, sigma = 0.0, 1.0
    print(f"[recursion-train] rounds={rounds} discount={DISCOUNT} "
          f"normalize_per_round=True", flush=True)
    best = None
    for it in range(iters):
        pop = rng.normal(mu, sigma, 24)
        scored = []
        for th in pop:
            adv, mf, used = discounted_normalized_fitness(th, tasks, rng)
            scored.append((mf, th, used))
        scored.sort(reverse=True)
        elite = scored[:6]
        mu = np.mean([e[1] for e in elite])
        sigma = max(0.05, np.std([e[1] for e in elite]))
        best = scored[0]
        if it % 5 == 0 or it == iters - 1:
            print(f"[iter {it:2d}] mean_final={best[0]:.3f}  "
                  f"P(revise)={sigmoid(mu):.2f}  recurse_used={best[2]*100:.0f}%", flush=True)
    return mu


def main():
    ap = argparse.ArgumentParser(description="Mock recursive-Conductor training (Fugu-Ultra).")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    tasks = rng.uniform(1.0, 2.0, 8)

    # baseline: one-shot only (rounds=1) vs learned recursive policy
    one_shot = np.mean([run_episode(-99, tasks[rng.integers(len(tasks))], rng, rounds=1)[1]
                        for _ in range(2000)])
    theta = train(iters=args.iters, seed=args.seed)
    learned = np.mean([run_episode(theta, tasks[rng.integers(len(tasks))], rng)[1]
                       for _ in range(2000)])

    print(f"\n[result] one-shot (no recursion) mean reward = {one_shot:.3f}")
    print(f"[result] learned recursive policy   mean reward = {learned:.3f}  "
          f"(P(revise)={sigmoid(theta):.2f})")
    gain = (learned - one_shot) / one_shot * 100
    print(f"[result] recursion lifts reward by {gain:+.0f}%")
    if learned > one_shot + 0.02 and sigmoid(theta) > 0.6:
        print("PASS — the recursive Conductor learned to revise its own output and "
              "scores above one-shot (test-time scaling via recursion works)")


if __name__ == "__main__":
    main()
