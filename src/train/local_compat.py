"""Plain-PyTorch stand-in for TRL's Triton log-probability kernel, for `--local-test` only.

TRL 1.15 computes per-token log-probabilities with a fused LM head whose kernel needs
Triton, which ships with PyTorch on Linux only. On a Mac the kernel is None and the first
training step fails. This shim materializes the full logits instead (fine for a 0.5B model
and a few short completions) so the whole loop can be exercised locally. GPU runs use
TRL's kernel; grpo.py installs the shim only for --local-test and only when the kernel is
missing.
"""

import torch


class TorchChunkedLogProb:
    """Same call signature and outputs as trl.kernels.ChunkedLogProbFunction.apply."""

    @staticmethod
    def apply(hidden, weight, bias, targets, temperature, chunk_size, final_logit_softcapping=None,
              logit_scale=1.0, outputs=("entropy",)):
        logits = torch.nn.functional.linear(hidden, weight, bias).float() * logit_scale
        if final_logit_softcapping is not None:
            logits = torch.tanh(logits / final_logit_softcapping) * final_logit_softcapping
        mean_logits = logits.mean(-1) if "mean_logits" in outputs else None
        logits = logits / temperature
        logp = logits.log_softmax(-1)
        logprobs = logp.gather(-1, targets[:, None]).squeeze(-1)
        entropy = -(logp.exp() * logp).sum(-1) if "entropy" in outputs else None
        log_sum_sq = torch.logsumexp(2 * logp, -1) if "log_sum_sq_probs" in outputs else None
        is_top1 = (logits.argmax(-1) == targets) if "is_top1" in outputs else None
        return logprobs, entropy, log_sum_sq, mean_logits, is_top1


def install():
    """Patch TRL to use the shim if its Triton kernel is unavailable. Returns True if patched."""
    import trl.trainer.utils as trl_utils

    if trl_utils._ChunkedLogProbFunction is not None:
        return False
    trl_utils._ChunkedLogProbFunction = TorchChunkedLogProb
    return True
