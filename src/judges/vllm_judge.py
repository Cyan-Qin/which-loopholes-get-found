"""Run a judge with vLLM on the GPU.

Same interface as src.judges.local.LocalJudge: build the native prompt, generate
with the judge's decoding settings from configs/judges.yaml, parse the verdict.
"""

import time

from src.judges.local import JudgeResult
from src.judges.parse import parse_verdict
from src.judges.prompts import build_prompt, judge_answer

MAX_MODEL_LEN = 4096   # judge prompt (question + answer tail) plus up to 2048 generated tokens


class VLLMJudge:
    def __init__(self, cfg, gpu_memory_utilization=0.9, seed=0, **llm_kwargs):
        from vllm import LLM

        self.cfg = cfg
        self.llm = LLM(model=cfg["model"], dtype="bfloat16", seed=seed, max_model_len=MAX_MODEL_LEN,
                       gpu_memory_utilization=gpu_memory_utilization, **llm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()

    def judge(self, items, seed=0):
        """items: dicts with question, reference and response. Returns JudgeResults and seconds."""
        from vllm import SamplingParams

        answers = [judge_answer(it["response"]) for it in items]
        prompts = [build_prompt(self.cfg, it["question"], it["reference"], a, self.tokenizer)
                   for it, a in zip(items, answers)]
        params = SamplingParams(temperature=self.cfg["temperature"], top_p=self.cfg["top_p"],
                                max_tokens=self.cfg["max_new_tokens"], seed=seed)
        start = time.time()
        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        seconds = time.time() - start
        results = []
        for a, out in zip(answers, outputs):
            c = out.outputs[0]
            v = parse_verdict(self.cfg, c.text)
            results.append(JudgeResult(a, c.text, v.accepted, v.parse_fail, v.label, len(c.token_ids),
                                       c.finish_reason == "length"))
        return results, seconds
