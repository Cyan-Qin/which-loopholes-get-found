"""Pick the device and dtype for local runs, and set vLLM environment defaults."""

import os

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


def configure_vllm():
    """Environment defaults to set before importing vLLM (existing values win).

    VLLM_USE_FLASHINFER_SAMPLER=0: use vLLM's PyTorch sampler. FlashInfer's sampler
    is JIT-compiled at startup and needs the CUDA toolkit (nvcc), which a machine
    with only the NVIDIA driver lacks; the PyTorch sampler handles our per-request
    seeds as well.
    """
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
