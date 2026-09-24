#!/usr/bin/env python3
# OpenFugu — Apache-2.0. Part of an independent, open reimplementation of
# the Fugu orchestrator. NOT affiliated with Sakana AI. See NOTICE.
# Reference: Fugu-Ultra recursive Conductor (arXiv:2512.04388). Honest eval of
# the recursion CLAIM: does a second (revise) round, conditioned on the model's
# own first-round output, actually improve the plan? Original code.
"""
eval_recursion_real.py — does recursion actually help? (round-0 vs round-1)

The recursion training (train_recursion_real.py) is faithful: round-1's prompt
contains the model's own round-0 output + a correction instruction. But "GRPO
reward climbed" is NOT the right success metric, because on easy questions
round-0 is already good (reward saturates, no GRPO gradient). The HONEST metric
is held-out: for each question, score the round-0 plan, then feed it back and
score the round-1 (revised) plan. Recursion helps iff round-1 > round-0 — and
it should help most on the questions round-0 got wrong.

  base    : the recursion-finetuned Conductor (or checkpoint-100 if untrained)
  data    : held-out ToolScale (the eval split make_datasets reserves)
  metric  : mean tool-call score, round-0 vs round-1, + "fix rate" on round-0 misses
"""
import os, sys, re, json, argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, "/root/conductor_train")
from custom_data.toolscale_data import make_datasets, _parse_plan, _score

RECURSION_CORRECTION = (
    "Here is the final response obtained at the end of your routing steps: "
    "{worker_response}\n\n"
    "You now have a chance to correct or improve this response by outputting a new "
    "sequence of up to 5 routing steps, with the same format. Once again, the goal "
    "is to produce a final response that answers the original user question correctly. "
    "If the previous final response is already correct you may return it as is; "
    "otherwise revise, verify, and improve it."
)
def gen(model, tok, prompt_text, max_new=320):
    enc = tok(prompt_text, return_tensors="pt", padding=False,
              add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(input_ids=enc["input_ids"],
                             attention_mask=enc["attention_mask"],
                             max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def splice(tok, base_prompt, round0_text):
    """round-1 prompt = base + round-0 answer (assistant) + correction (user)."""
    worker_response = round0_text.strip() or "(no response produced)"
    correction = RECURSION_CORRECTION.format(worker_response=worker_response)
    msgs = [
        {"role": "assistant", "content": round0_text.strip()},
        {"role": "user", "content": correction},
        {"role": "assistant", "content": "Let me revise the tool calls.\n<think>"},
    ]
    try:
        tail = tok.apply_chat_template(msgs, tokenize=False, continue_final_message=True)
    except Exception:
        tail = (f"\n{round0_text.strip()}\n\n{correction}\n"
                "Let me revise the tool calls.\n<think>")
    return base_prompt + tail


def score_completion(comp, gold):
    pred = _parse_plan(comp)
    return 0.0 if pred is None else _score(pred, gold)


def main():
    ap = argparse.ArgumentParser(description="Honest recursion eval: round-0 vs round-1.")
    ap.add_argument("--model", default=os.environ.get("FUGU_EVAL_CKPT",
        "/vePFS-Mindverse/share/diz/openfugu/conductor_recursion_real"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = make_datasets(data_limit=1200, tokenizer=tok, seed=args.seed)
    ev = ds["eval_dataset"].select(range(min(args.n, len(ds["eval_dataset"]))))
    print(f"[eval-recur] model={args.model.split('/')[-1]} held-out n={len(ev)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="bfloat16").to("cuda").eval()

    r0_scores, r1_scores, fixed, broke = [], [], 0, 0
    for i, row in enumerate(ev):
        gold = json.loads(row["expected_actions"]) if row["expected_actions"] else []
        base = row["prompt"]
        c0 = gen(model, tok, base)
        s0 = score_completion(c0, gold)
        c1 = gen(model, tok, splice(tok, base, c0))
        s1 = score_completion(c1, gold)
        r0_scores.append(s0); r1_scores.append(s1)
        if s1 > s0 + 1e-6: fixed += 1
        elif s1 < s0 - 1e-6: broke += 1
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(ev)}] r0={np.mean(r0_scores):.3f} "
                  f"r1={np.mean(r1_scores):.3f}", flush=True)

    m0, m1 = float(np.mean(r0_scores)), float(np.mean(r1_scores))
    miss = [j for j, s in enumerate(r0_scores) if s < 0.999]
    fix_on_miss = np.mean([r1_scores[j] - r0_scores[j] for j in miss]) if miss else 0.0
    lift = (m1 - m0) / m0 * 100 if m0 > 0 else 0.0
    print(f"\n[result] round-0 mean score = {m0:.3f}")
    print(f"[result] round-1 mean score = {m1:.3f}  ({lift:+.1f}%)")
    print(f"[result] questions improved by revise = {fixed}/{len(ev)}; regressed = {broke}")
    print(f"[result] on round-0 misses (n={len(miss)}), revise delta = {fix_on_miss:+.3f}")
    if m1 > m0 + 0.005:
        print("PASS — the revise round improves plans over round-0 (recursion helps)")
    elif m1 >= m0 - 0.005:
        print("TIE — revise round neither helps nor hurts (round-0 already strong / saturated)")
    else:
        print("BELOW — revise round degrades plans")


if __name__ == "__main__":
    main()
