from src.device import configure_vllm, default_device, default_dtype


def test_configure_vllm_sets_defaults_without_overriding(monkeypatch):
    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)
    configure_vllm()
    import os
    assert os.environ["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "1")
    configure_vllm()
    assert os.environ["VLLM_USE_FLASHINFER_SAMPLER"] == "1"   # an explicit choice is kept


def test_default_device_and_dtype():
    import torch
    assert default_device() in {"cuda", "mps", "cpu"}
    assert default_dtype("cuda") == torch.bfloat16 and default_dtype("cpu") == torch.float32
