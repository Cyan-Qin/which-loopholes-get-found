"""The policy prompt.

Every stage that queries the policy (pool building, the difficulty run, the
propensity audit, GRPO and evaluation) builds its prompt here, from the
`prompt` section of configs/data.yaml, so the prompt cannot drift between stages.
"""

from functools import lru_cache
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "data.yaml"


@lru_cache(maxsize=1)
def prompt_config():
    return yaml.safe_load(CONFIG.read_text())["prompt"]


def build_messages(problem, cfg=None):
    cfg = cfg or prompt_config()
    messages = [{"role": "user", "content": problem + cfg["suffix"]}]
    if cfg["system"] is not None:
        messages.insert(0, {"role": "system", "content": cfg["system"]})
    return messages


def build_prompt(problem, tokenizer, cfg=None):
    """Chat-templated prompt string, ending with the assistant turn header."""
    return tokenizer.apply_chat_template(build_messages(problem, cfg), tokenize=False, add_generation_prompt=True)
