"""Pick the device and dtype for local (transformers) runs."""

import torch


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_dtype(device):
    """bf16 on CUDA (Qwen2.5 is trained in bf16), fp16 on Apple MPS, fp32 on CPU."""
    return {"cuda": torch.bfloat16, "mps": torch.float16}.get(device, torch.float32)
