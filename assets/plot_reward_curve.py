#!/usr/bin/env python3
# OpenFugu — Apache-2.0. Plot the Conductor GRPO reward curve from a training log.
"""Parse train_conductor.py step logs and plot reward / format / action curves.
Usage: python plot_reward_curve.py <log.txt> <out.png>"""
import re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log, out = sys.argv[1], sys.argv[2]
steps, reward, fmt, act = [], [], [], []
pat = re.compile(r"\[step (\d+)\] reward=([\d.]+) fmt=([\d.]+) act=([\d.]+)")
for line in open(log):
    m = pat.search(line)
    if m:
        steps.append(int(m.group(1)))
        reward.append(float(m.group(2)))
        fmt.append(float(m.group(3)))
        act.append(float(m.group(4)))

print(f"parsed {len(steps)} steps; reward {reward[0]:.2f} -> {reward[-1]:.2f}")

plt.figure(figsize=(9, 5.2))
plt.plot(steps, reward, color="#0b62d6", lw=2.2, label="total reward (format + action)")
plt.plot(steps, fmt, color="#22a06b", lw=1.5, ls="--", label="format reward")
plt.plot(steps, act, color="#d9730d", lw=1.5, ls="--", label="action reward (tool-call match)")
plt.axhline(reward[0], color="#999", lw=0.8, ls=":", alpha=0.7)
plt.annotate(f"start {reward[0]:.2f}", (steps[0], reward[0]),
             textcoords="offset points", xytext=(6, -12), color="#666", fontsize=9)
plt.annotate(f"end {reward[-1]:.2f}", (steps[-1], reward[-1]),
             textcoords="offset points", xytext=(-44, 6), color="#0b62d6", fontsize=9)
plt.title("OpenFugu Conductor — GRPO on nvidia/ToolScale\n"
          "Llama-3.2-3B-Instruct, 100 steps, β=0", fontsize=12)
plt.xlabel("training step")
plt.ylabel("reward (mean over group)")
plt.ylim(0, 2.05)
plt.legend(loc="lower right", framealpha=0.9)
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(out, dpi=140)
print(f"wrote {out}")
