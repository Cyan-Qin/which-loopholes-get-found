"""Sample policy responses locally with Hugging Face transformers (CUDA, Apple MPS or CPU).

For smoke tests with small models. The difficulty run, the audit and GRPO
use vLLM on the GPU with the same prompt (src/policy/prompt.py).
"""

import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.device import default_device, default_dtype
from src.policy.prompt import build_prompt


@dataclass(frozen=True)
class Sample:
    text: str
    n_tokens: int       # generated tokens up to and excluding the end-of-turn token
    truncated: bool     # hit max_new_tokens without an end-of-turn token


class LocalPolicy:
    def __init__(self, model, device=None):
        self.model_name = model
        self.device = device or default_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=default_dtype(self.device)).to(self.device)
        self.model.eval()
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else min(self.eos_ids)

    @torch.inference_mode()
    def sample(self, problems, n, temperature, top_p, max_new_tokens, seed, request_seeds=None):
        """n samples per problem. Returns a list (per problem) of lists of Samples, and seconds.

        With request_seeds, the random state is reset per problem (as in VLLMPolicy).
        """
        torch.manual_seed(seed)
        start = time.time()
        results = []
        for i, problem in enumerate(problems):
            if request_seeds is not None:
                torch.manual_seed(request_seeds[i])
            prompt = build_prompt(problem, self.tokenizer)
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
            out = self.model.generate(**inputs, do_sample=True, temperature=temperature, top_p=top_p,
                                      max_new_tokens=max_new_tokens, num_return_sequences=n,
                                      pad_token_id=self.pad_id)
            samples = []
            for row in out[:, inputs["input_ids"].shape[1]:].tolist():
                end = next((i for i, t in enumerate(row) if t in self.eos_ids), None)
                tokens = row if end is None else row[:end]
                samples.append(Sample(self.tokenizer.decode(tokens, skip_special_tokens=True), len(tokens),
                                      end is None and len(row) >= max_new_tokens))
            results.append(samples)
        return results, time.time() - start
