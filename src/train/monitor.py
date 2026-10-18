"""Periodic evaluation during GRPO: the exploitation rates e_f(t).

At step 0 and every `every` optimizer steps, the current policy answers the held-out eval
problems of its pool (`samples` responses each, training-time sampling). The run's own
judge grades them; the gold checker and the detectors label them. Every response becomes a
judge record (src/pipeline/records.py) in eval.jsonl, and one summary line per evaluation
goes to eval_summary.jsonl with

    e/<family> = share of eval responses that the judge accepted, the gold checker marks
                 wrong, and that belong to the family (also e/no_box, see rewards.py).

Generation reuses the trainer's colocated vLLM engine after syncing it to the current
weights, so evaluation sees exactly the policy that training sees. Without vLLM (local
tests) it falls back to Hugging Face generate.
"""

import json
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import TrainerCallback

from src.detectors.families import FAMILIES, Detection
from src.pipeline.records import make_record, write_jsonl
from src.policy.local import Sample
from src.policy.prompt import build_prompt
from src.train.rewards import is_truncated, label_rollouts

E_KEYS = (*FAMILIES, "no_box")


def summarize_eval(step, pool, records, gen_seconds, judge_seconds):
    n = len(records) or 1
    no_box = [r["no_answer"] and not r["truncated"] for r in records]
    row = {
        "step": step, "pool": pool, "n": len(records),
        "accept_rate": sum(r["accepted"] for r in records) / n,
        "gold_accuracy": sum(r["gold_correct"] for r in records) / n,
        "false_positive_rate": sum(r["false_positive"] for r in records) / n,
        "exploit_rate": sum(r["exploit"] for r in records) / n,
        "family_rate": sum(r["family"] is not None for r in records) / n,
        "parse_fail_rate": sum(r["parse_fail"] for r in records) / n,
        "policy_truncated_rate": sum(r["truncated"] for r in records) / n,
        "no_box_rate": sum(no_box) / n,
        "mean_tokens": sum(r["n_tokens"] for r in records) / n,
        "gen_seconds": gen_seconds, "judge_seconds": judge_seconds,
    }
    exploits = Counter(r["family"] for r in records if r["exploit"])
    exploits["no_box"] = sum(r["false_positive"] and nb for r, nb in zip(records, no_box))
    row.update({f"e/{k}": exploits[k] / n for k in E_KEYS})
    return row


class EvalCallback(TrainerCallback):
    """Set `callback.trainer = trainer` after building the trainer, before train()."""

    def __init__(self, *, judge, judge_name, problems, pool, run_dir, run_id, policy_model, every,
                 samples, max_completion_length, temperature, top_p):
        self.judge, self.judge_name = judge, judge_name
        self.problems, self.pool = problems, pool
        self.run_dir, self.run_id, self.policy_model = Path(run_dir), run_id, policy_model
        self.every, self.samples = every, samples
        self.max_completion_length, self.temperature, self.top_p = max_completion_length, temperature, top_p
        self.trainer = None
        self.done_steps = set()

    # --- generation ------------------------------------------------------------
    def _prompt_ids(self, tokenizer):
        ids = [tokenizer(build_prompt(p["problem"], tokenizer), add_special_tokens=False)["input_ids"]
               for p in self.problems]
        return [x for x in ids for _ in range(self.samples)]

    def _generate_vllm(self, prompt_ids):
        t = self.trainer
        if t.state.global_step != t._last_loaded_step:
            t.vllm_generation.sync_weights()
            t._last_loaded_step = t.state.global_step
        _, completion_ids, _, _ = t.vllm_generation.generate(prompts=prompt_ids, images=None,
                                                             num_generations=self.samples)
        return [list(c) for c in completion_ids]

    @torch.no_grad()
    def _generate_hf(self, prompt_ids, tokenizer, batch_size=8):
        t = self.trainer
        model = t.accelerator.unwrap_model(t.model)
        was_training = model.training
        model.eval()
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        out = []
        try:
            for i in range(0, len(prompt_ids), batch_size):
                batch = prompt_ids[i:i + batch_size]
                width = max(len(x) for x in batch)
                input_ids = torch.tensor([[pad] * (width - len(x)) + x for x in batch], device=model.device)
                mask = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in batch], device=model.device)
                gen = model.generate(input_ids=input_ids, attention_mask=mask, do_sample=True,
                                     temperature=self.temperature, top_p=self.top_p,
                                     max_new_tokens=self.max_completion_length, pad_token_id=pad)
                for row in gen[:, width:].tolist():
                    out.append(self._strip_padding(row, pad))
        finally:
            model.train(was_training)
        return out

    def _strip_padding(self, row, pad):
        stop = set(self.trainer.eos_token_ids)
        for j, tok in enumerate(row):
            if tok in stop:
                return row[:j + 1]
        while row and row[-1] == pad:
            row = row[:-1]
        return row

    # --- evaluation --------------------------------------------------------------
    def evaluate(self, step):
        if step in self.done_steps:
            return None
        t = self.trainer
        tokenizer = t.processing_class
        prompt_ids = self._prompt_ids(tokenizer)
        start = time.time()
        completion_ids = self._generate_vllm(prompt_ids) if t.use_vllm else self._generate_hf(prompt_ids, tokenizer)
        gen_seconds = time.time() - start

        stop = set(t.eos_token_ids) | {tokenizer.pad_token_id}
        texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        truncated = [is_truncated(ids, self.max_completion_length, stop) for ids in completion_ids]
        expanded = [p for p in self.problems for _ in range(self.samples)]
        items = [{"question": p["problem"], "reference": p["reference"], "response": x, "truncated": tr}
                 for p, x, tr in zip(expanded, texts, truncated)]
        results, judge_seconds = self.judge.judge(items)
        labels = label_rollouts(texts, [p["reference"] for p in expanded], truncated)

        records = []
        for i, (p, text, ids, tr, lab, res) in enumerate(zip(expanded, texts, completion_ids, truncated,
                                                             labels, results)):
            n_tokens = len(ids) - (0 if tr or not ids or ids[-1] not in stop else 1)
            records.append(make_record(
                run_id=self.run_id, stage="grpo_eval", step=step, item=p, sample_idx=i % self.samples,
                policy_model=self.policy_model, sample=Sample(text, n_tokens, tr), gold=lab["gold"],
                detection=Detection(lab["family"], frozenset(lab["contains"])),
                judge_name=self.judge_name, verdict=res))
        write_jsonl(self.run_dir / "eval.jsonl", records)
        row = summarize_eval(step, self.pool, records, gen_seconds, judge_seconds)
        with (self.run_dir / "eval_summary.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        self.done_steps.add(step)
        print(f"[eval step {step}] accept {row['accept_rate']:.3f} gold {row['gold_accuracy']:.3f} "
              f"fp {row['false_positive_rate']:.3f} exploit {row['exploit_rate']:.3f} "
              f"no_box {row['no_box_rate']:.3f} ({gen_seconds:.0f}s gen, {judge_seconds:.0f}s judge)", flush=True)
        return row

    def on_train_begin(self, args, state, control, **kwargs):
        self.evaluate(state.global_step)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every == 0:
            self.evaluate(state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        self.evaluate(state.global_step)


class LogCallback(TrainerCallback):
    """Append every trainer log (loss, reward, KL, clip ratios, judge metrics) to train_log.jsonl."""

    def __init__(self, run_dir):
        self.path = Path(run_dir) / "train_log.jsonl"

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps({"step": state.global_step, **logs}) + "\n")
