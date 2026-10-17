"""Sample policy responses with vLLM on the GPU.

Same interface as src.policy.local.LocalPolicy, plus per-request seeds: each
problem draws from its own random stream, recorded with every sample. GPU
numerics still depend on batch composition, so a rerun does not reproduce
responses token for token (a check on the L4 matched 10 of 231 finished
responses exactly); a resumed run draws statistically equivalent samples.
"""

import time

from transformers import GenerationConfig

from src.device import configure_vllm
from src.policy.local import Sample
from src.policy.prompt import build_prompt, prompt_config


class VLLMPolicy:
    def __init__(self, model, max_new_tokens, gpu_memory_utilization=0.9, seed=0, **llm_kwargs):
        configure_vllm()
        from vllm import LLM  # imported here so the rest of the package works without vLLM

        self.model_name = model
        max_model_len = prompt_config()["max_prompt_tokens"] + max_new_tokens
        self.llm = LLM(model=model, dtype="bfloat16", seed=seed, max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization, **llm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        eos = GenerationConfig.from_pretrained(model).eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])

    def sample(self, problems, n, temperature, top_p, max_new_tokens, seed, request_seeds=None):
        """n samples per problem. Returns a list (per problem) of lists of Samples, and seconds."""
        from vllm import SamplingParams

        seeds = request_seeds if request_seeds is not None else [seed + i for i in range(len(problems))]
        params = [SamplingParams(n=n, temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, seed=s)
                  for s in seeds]
        prompts = [build_prompt(p, self.tokenizer) for p in problems]
        start = time.time()
        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        seconds = time.time() - start
        results = []
        for out in outputs:
            samples = []
            for c in out.outputs:
                ids = list(c.token_ids)
                if ids and ids[-1] in self.eos_ids:
                    ids = ids[:-1]
                samples.append(Sample(c.text, len(ids), c.finish_reason == "length"))
            results.append(samples)
        return results, seconds
