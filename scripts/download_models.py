"""Download every model the project uses into the Hugging Face cache.

The list comes from the configs (the policy in configs/data.yaml, the judges in
configs/judges.yaml) plus the small policy used for smoke tests, so it cannot
drift from what the code loads.

Usage:
    python scripts/download_models.py             # download all
    python scripts/download_models.py --dry-run   # list models and sizes only
    python scripts/download_models.py --only xverify_0p5b policy

Set HF_HOME to put the cache on the data disk, HF_HUB_DISABLE_XET=1 if
downloads stall, and HF_ENDPOINT=https://hf-mirror.com to use the mirror.
"""

import argparse
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
SMOKE_POLICY = "Qwen/Qwen2.5-0.5B-Instruct"
# Weights in safetensors only; skip duplicate formats some repos also carry.
ALLOW = ["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "*.py", "*.jinja"]


def model_list():
    data = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    judges = yaml.safe_load((ROOT / "configs" / "judges.yaml").read_text())["judges"]
    models = {"policy": data["prompt"]["model"], "smoke_policy": SMOKE_POLICY}
    models.update({name: j["model"] for name, j in judges.items()})
    return models


def size_gb(api, repo_id):
    info = api.model_info(repo_id, files_metadata=True)
    files = [s for s in info.siblings if s.rfilename.endswith(".safetensors")]
    if not files:
        raise RuntimeError(f"{repo_id} has no safetensors weights")
    return sum(s.size or 0 for s in files) / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="*", help="names from the configs, e.g. policy xverify_0p5b")
    args = ap.parse_args()

    models = model_list()
    if args.only:
        unknown = set(args.only) - set(models)
        if unknown:
            sys.exit(f"unknown names {sorted(unknown)}; choose from {sorted(models)}")
        models = {k: v for k, v in models.items() if k in args.only}

    api = HfApi()
    total = 0.0
    for name, repo in models.items():
        gb = size_gb(api, repo)
        total += gb
        print(f"{name:<28} {repo:<45} {gb:6.2f} GB", flush=True)
    print(f"{'total':<74} {total:6.2f} GB", flush=True)
    if args.dry_run:
        return

    for name, repo in models.items():
        path = snapshot_download(repo, allow_patterns=ALLOW)
        print(f"downloaded {name}: {path}", flush=True)


if __name__ == "__main__":
    main()
