#!/usr/bin/env python3
# OpenFugu — Apache-2.0. Part of an independent, open reimplementation of
# the Fugu orchestrator. NOT affiliated with Sakana AI. See NOTICE.
# Reference: Fugu-Ultra recursive Conductor (arXiv:2512.04388; the released
# conductor_recursion_engine.py). The REAL recursion: round-0 emits a workflow,
# its output is fed BACK, round-1 revises — round-1's GRPO prompt literally
# contains the model's own round-0 attempt. Original code over trl.
"""
train_recursion_real.py — REAL recursive-Conductor GRPO finetune.

The earlier version of this file was single-round GRPO with "recursion" living
only in the docstring — the round-0 output was never actually fed back. This is
the real thing, faithful to conductor_recursion_engine's multi-round
_generate_and_score_completions:

  round 0 : the Conductor emits a workflow for the question (one greedy self-
            rollout per UNIQUE prompt)
  feedback: round-0's output is spliced into round-1's prompt via the recursion
            correction message (wording from recursion_formats.SIMPLE_*_V0)
  round 1 : GRPO-trained — the model REVISES, conditioned on seeing its own
            round-0 attempt; reward = score of the revised plan vs gold

We subclass GRPOTrainer and override ONLY _generate_and_score_completions: it
runs the round-0 self-rollout, rewrites every prompt to include round-0's output
+ the correction instruction, then defers to the parent for the actual GRPO
generation/scoring on the rewritten prompts. trl's verified PPO loss, advantage
normalization, and logprob machinery are untouched. The recursion is REAL because
round-1's prompt genuinely contains round-0's own output.

ponytail: reuse the trl GRPO stack; recursion lives in one method override + a
prompt builder, not a new RL engine.
"""
import os, sys, re, json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import GRPOTrainer, GRPOConfig
sys.path.insert(0, "/root/conductor_train")
from custom_data.toolscale_data import make_datasets, _parse_plan, _score

BASE = os.environ.get("FUGU_BASE_CKPT",
    "/vePFS-Mindverse/share/diz/openfugu/conductor_toolscale_100/checkpoint-100")
OUT = os.environ.get("FUGU_OUT", "/vePFS-Mindverse/share/diz/openfugu/conductor_recursion_real")
MAX_STEPS = int(os.environ.get("FUGU_STEPS", "30"))
SEED = 42

# recursion correction message — wording from
# conductor/custom_data/recursion_formats.py SIMPLE_recursion_FORMAT_V0.
RECURSION_CORRECTION = (
    "Here is the final response obtained at the end of your routing steps: "
    "{worker_response}\n\n"
    "You now have a chance to correct or improve this response by outputting a new "
    "sequence of up to 5 routing steps, with the same format. Once again, the goal "
    "is to produce a final response that answers the original user question correctly. "
    "If the previous final response is already correct you may return it as is; "
    "otherwise revise, verify, and improve it."
)
class RecursiveGRPOTrainer(GRPOTrainer):
    """GRPO trainer whose generation step is preceded by a round-0 self-rollout.
    Round 1 (the round actually trained) sees the model's own round-0 output in
    its prompt — this is what makes the recursion real rather than nominal."""

    @torch.no_grad()
    def _round0_rollout(self, base_prompts_text):
        """One greedy round-0 completion per (already num_generations-expanded)
        prompt. Returns the decoded round-0 completion text per prompt."""
        tok = self.processing_class
        dev = self.accelerator.device
        was_training = self.model.training
        self.model.eval()
        enc = tok(base_prompts_text, return_tensors="pt", padding=True,
                  padding_side="left", add_special_tokens=False).to(dev)
        from contextlib import nullcontext
        unwrapped = self.accelerator.unwrap_model(self.model)
        out = unwrapped.generate(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            max_new_tokens=self.max_completion_length, do_sample=False,
            pad_token_id=tok.pad_token_id)
        comp = out[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(comp, skip_special_tokens=True)
        if was_training:
            self.model.train()
        return texts

    def _splice_recursion(self, base_prompt_text, round0_text):
        """round-1 prompt = base prompt + round-0 answer (as assistant turn) +
        the recursion correction instruction (as a new user turn)."""
        tok = self.processing_class
        worker_response = round0_text.strip() or "(no response produced)"
        correction = RECURSION_CORRECTION.format(worker_response=worker_response)
        msgs = [
            {"role": "assistant", "content": round0_text.strip()},
            {"role": "user", "content": correction},
            {"role": "assistant", "content": "Let me revise the tool calls.\n<think>"},
        ]
        try:
            tail = tok.apply_chat_template(msgs, tokenize=False,
                                           continue_final_message=True)
        except Exception:
            tail = (f"\n{round0_text.strip()}\n\n{correction}\n"
                    "Let me revise the tool calls.\n<think>")
        return base_prompt_text + tail

    def _generate_and_score_completions(self, inputs):
        # 1) round-0 self-rollout on the ORIGINAL prompts
        base_prompts_text = [x["prompt"] for x in inputs]
        round0 = self._round0_rollout(base_prompts_text)
        # 2) rewrite every prompt to contain round-0's own output + correction
        for x, r0 in zip(inputs, round0):
            x["prompt"] = self._splice_recursion(x["prompt"], r0)
        # 3) standard GRPO on the recursion prompts (round-1 = the revise round)
        return super()._generate_and_score_completions(inputs)


def revise_reward(completions, expected_actions=None, **kw):
    """Score the REVISED (round-1) plan against gold tool calls."""
    out = []
    exp = expected_actions or [None] * len(completions)
    for comp, gold_json in zip(completions, exp):
        try:
            gold = json.loads(gold_json) if gold_json else []
            pred = _parse_plan(comp)
            out.append(0.0 if pred is None else _score(pred, gold))
        except Exception:
            out.append(0.0)
    return out


def format_reward(completions, **kw):
    return [1.0 if re.search(r"<answer>[\s\S]*?</answer>", "<think>" + c) else 0.0
            for c in completions]


class RewardLog(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kw):
        if logs and "reward" in logs:
            print(f"[step {state.global_step}] reward={logs.get('reward'):.3f} "
                  f"rev={logs.get('rewards/revise_reward/mean',0):.3f} "
                  f"fmt={logs.get('rewards/format_reward/mean',0):.3f} "
                  f"std={logs.get('reward_std',0):.3f}", flush=True)


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = make_datasets(data_limit=256, tokenizer=tok, seed=SEED)
    print(f"[recursion-real] base={BASE.split('/')[-1]} "
          f"train={len(ds['train_dataset'])} steps={MAX_STEPS}", flush=True)

    cfg = GRPOConfig(
        output_dir=OUT, per_device_train_batch_size=8, gradient_accumulation_steps=2,
        num_generations=8, max_prompt_length=896, max_completion_length=320,
        max_steps=MAX_STEPS, learning_rate=1e-5, logging_steps=1,
        save_strategy="steps", save_steps=MAX_STEPS, report_to=[],
        use_vllm=False, bf16=True, gradient_checkpointing=True,
        temperature=1.0, beta=0.0)            # no KL — matches Fugu-Ultra
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype="bfloat16").to("cuda")
    model.config.use_cache = False
    trainer = RecursiveGRPOTrainer(
        model=model, processing_class=tok,
        reward_funcs=[format_reward, revise_reward],
        args=cfg, train_dataset=ds["train_dataset"], callbacks=[RewardLog()])
    print("[recursion-real] REAL recursion: round-0 self-rollout fed back into "
          "round-1 GRPO prompt. starting...", flush=True)
    trainer.train()
    trainer.save_model(OUT)
    print(f"[recursion-real] DONE — saved to {OUT}", flush=True)


if __name__ == "__main__":
    main()
