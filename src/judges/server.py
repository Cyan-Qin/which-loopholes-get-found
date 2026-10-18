"""Query a judge served by `vllm serve` (OpenAI-compatible completions API).

During GRPO the policy's own vLLM engine runs inside the trainer, and the judge runs as a
separate server process on the same GPU. ServerJudge builds the judge's native prompt
locally (src/judges/prompts.py), sends raw prompts to /v1/completions with the judge's
decoding settings, and parses verdicts like the other judge backends.

Start the server with the command from `server_command(cfg)`, for example
    vllm serve hkust-nlp/R1-Distill-Verifier-1.5B --port 8100 --gpu-memory-utilization 0.2 \
        --max-model-len 4096 --dtype bfloat16 --seed 0
"""

import json
import time
import urllib.request

from transformers import AutoTokenizer

from src.judges.local import JudgeResult
from src.judges.parse import parse_verdict
from src.judges.prompts import build_prompt, judge_answer
from src.judges.vllm_judge import PROMPT_ROOM


def server_command(cfg, port=8100, gpu_memory_utilization=0.2):
    max_model_len = max(4096, cfg["max_new_tokens"] + PROMPT_ROOM)
    return ["vllm", "serve", cfg["model"], "--port", str(port), "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--max-model-len", str(max_model_len), "--dtype", "bfloat16", "--seed", "0"]


class ServerJudge:
    def __init__(self, cfg, base_url="http://127.0.0.1:8100", timeout=1800, chunk=512):
        self.cfg, self.base_url, self.timeout, self.chunk = cfg, base_url.rstrip("/"), timeout, chunk
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model"])

    def wait_until_ready(self, max_seconds=900):
        deadline = time.time() + max_seconds
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=5) as r:
                    if r.status == 200:
                        return True
            except OSError:
                pass
            time.sleep(5)
        raise TimeoutError(f"judge server at {self.base_url} not ready after {max_seconds}s")

    def _complete(self, prompts):
        body = {"model": self.cfg["model"], "prompt": prompts, "temperature": self.cfg["temperature"],
                "top_p": self.cfg["top_p"], "max_tokens": self.cfg["max_new_tokens"], "n": 1, "seed": 0}
        req = urllib.request.Request(f"{self.base_url}/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            choices = json.loads(r.read())["choices"]
        return sorted(choices, key=lambda c: c["index"])

    def judge(self, items):
        """items: dicts with question, reference and response. Returns JudgeResults and seconds."""
        answers = [judge_answer(it["response"]) for it in items]
        prompts = [build_prompt(self.cfg, it["question"], it["reference"], a, self.tokenizer)
                   for it, a in zip(items, answers)]
        start = time.time()
        choices = []
        for i in range(0, len(prompts), self.chunk):
            choices += self._complete(prompts[i:i + self.chunk])
        seconds = time.time() - start
        results = []
        for a, c in zip(answers, choices):
            text = c["text"]
            v = parse_verdict(self.cfg, text)
            n_tokens = len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
            results.append(JudgeResult(a, text, v.accepted, v.parse_fail, v.label, n_tokens,
                                       c.get("finish_reason") == "length"))
        return results, seconds


def main():
    import argparse
    import shlex
    from urllib.parse import urlparse

    import yaml

    from src.judges.prompts import judge_config
    from src.paths import ROOT

    server_cfg = yaml.safe_load((ROOT / "configs" / "grpo.yaml").read_text())["judge_server"]
    ap = argparse.ArgumentParser(description="Print the `vllm serve` command for a judge (used by scripts/run_grpo.sh).")
    ap.add_argument("--judge", required=True)
    ap.add_argument("--port", type=int, default=urlparse(server_cfg["url"]).port)
    ap.add_argument("--gpu-memory-utilization", type=float, default=server_cfg["gpu_memory_utilization"])
    args = ap.parse_args()
    print(shlex.join(server_command(judge_config(args.judge), args.port, args.gpu_memory_utilization)))


if __name__ == "__main__":
    main()
