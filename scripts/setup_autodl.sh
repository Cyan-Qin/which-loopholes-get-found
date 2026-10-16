#!/usr/bin/env bash
# One-time setup of an AutoDL instance. Run it in no-GPU mode (无卡模式) to save money;
# everything it installs stays on the instance for later GPU sessions.
#
#   bash setup_autodl.sh                       # host whose 最高CUDA >= 13.0 (default wheels)
#   CUDA_VARIANT=cu129 bash setup_autodl.sh    # host whose 最高CUDA is 12.9
#   SKIP_MODELS=1 bash setup_autodl.sh         # environment and data only
#   HF_ENDPOINT=https://hf-mirror.com bash setup_autodl.sh   # if huggingface.co is slow
#
# Layout:
#   /root/autodl-tmp/which-loopholes-get-found   code, data/ and runs/ (data disk)
#   /root/autodl-tmp/hf_cache                    model weights (HF_HOME, data disk)
#   conda env "loopholes" (Python 3.12)          packages (system disk)
# A full log is written to /root/autodl-tmp/setup.log.

set -euo pipefail

CUDA_VARIANT="${CUDA_VARIANT:-cu130}"
DATA="${DATA:-/root/autodl-tmp}"   # override only for testing elsewhere
REPO_URL=https://github.com/justLxy/which-loopholes-get-found.git
REPO="$DATA/which-loopholes-get-found"
ENV_NAME=loopholes
VLLM_VERSION=0.31.0
TORCH_VERSION=2.13.0
EXPECTED_POOL_MD5=fea6cc4341e9bbf85135915c50f0dba2

mkdir -p "$DATA"
exec > >(tee -a "$DATA/setup.log") 2>&1
step() { echo; echo "========== [$(date '+%H:%M:%S')] $*"; }
# AutoDL academic acceleration for GitHub and huggingface.co; off again for pip mirrors.
# shellcheck source=/dev/null
turbo_on() { if [ -f /etc/network_turbo ]; then source /etc/network_turbo >/dev/null; fi; }
turbo_off() { unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true; }

case "$CUDA_VARIANT" in
  cu130|cu129) ;;
  *) echo "CUDA_VARIANT must be cu130 or cu129, got $CUDA_VARIANT"; exit 1 ;;
esac

step "0/7 machine"
echo "CUDA_VARIANT=$CUDA_VARIANT"
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  nvidia-smi | grep -o "CUDA Version: [0-9.]*" || true
else
  echo "no GPU visible (expected in no-GPU mode)"
fi
df -h / "$DATA" | sed 1d

step "1/7 code"
turbo_on
if [ -d "$REPO/.git" ]; then
  git -C "$REPO" pull --ff-only
else
  git clone "$REPO_URL" "$REPO"
fi
turbo_off
git -C "$REPO" log -1 --format='commit %h %s'

step "2/7 conda env $ENV_NAME (Python 3.12)"
if command -v conda >/dev/null; then CONDA_BASE="$(conda info --base)"
elif [ -d /root/miniconda3 ]; then CONDA_BASE=/root/miniconda3
else echo "conda not found; pick an AutoDL image that includes Miniconda"; exit 1; fi
CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
set +u   # conda's activation scripts reference unset variables
# shellcheck source=/dev/null
source "$CONDA_SH"
if ! conda env list | grep -qE "^$ENV_NAME\s"; then
  # conda-forge avoids the Anaconda default-channel terms prompt in non-interactive shells
  conda create -y -n "$ENV_NAME" -c conda-forge --override-channels python=3.12
fi
conda activate "$ENV_NAME"
set -u
python --version

step "3/7 packages ($CUDA_VARIANT)"
cd "$REPO"
python -m pip install -q --upgrade pip
if [ "$CUDA_VARIANT" = cu129 ]; then
  # CUDA 12.9 builds: torch from the PyTorch index, vLLM from its GitHub release.
  python -m pip install "torch==$TORCH_VERSION" --index-url https://download.pytorch.org/whl/cu129
  turbo_on
  python -m pip install \
    "https://github.com/vllm-project/vllm/releases/download/v$VLLM_VERSION/vllm-$VLLM_VERSION+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
    --extra-index-url https://download.pytorch.org/whl/cu129
  turbo_off
fi
python -m pip install -r requirements-gpu.txt
python -m pip freeze > "requirements-lock-autodl-$CUDA_VARIANT.txt"
python - <<'EOF'
import importlib
for name in ["torch", "transformers", "vllm", "trl", "datasets", "math_verify"]:
    try:
        mod = importlib.import_module(name)
        print(f"  {name:<13} {getattr(mod, '__version__', 'ok')}")
    except Exception as e:  # report and continue: vllm may complain without a GPU
        print(f"  {name:<13} IMPORT FAILED: {type(e).__name__}: {e}")
import torch
print(f"  torch built for CUDA {torch.version.cuda}; GPU available now: {torch.cuda.is_available()}")
EOF

step "4/7 data"
export HF_HOME="$DATA/hf_cache" HF_HUB_DISABLE_XET=1
turbo_on
python -m src.data.download_hacking | tail -2
python -m src.data.build_pool | tail -1
python -m src.detectors.build_vocab
turbo_off
POOL_MD5=$(md5sum data/processed/pool_candidates.jsonl | cut -d' ' -f1)
if [ "$POOL_MD5" = "$EXPECTED_POOL_MD5" ]; then
  echo "candidate pool identical to the local build ($POOL_MD5)"
else
  echo "WARNING: candidate pool differs from the local build ($POOL_MD5 vs $EXPECTED_POOL_MD5)"
fi
git diff --quiet src/detectors/vocab.txt && echo "vocab.txt identical to the committed one" \
  || echo "WARNING: vocab.txt differs from the committed one"

step "5/7 models"
if [ "${SKIP_MODELS:-0}" = 1 ]; then
  echo "skipped (SKIP_MODELS=1)"
else
  if [ -z "${HF_ENDPOINT:-}" ]; then turbo_on; fi
  python scripts/download_models.py
  turbo_off
fi

step "6/7 tests"
python -m pytest -q 2>&1 | tail -3

step "7/7 summary"
df -h / "$DATA" | sed 1d
du -sh "$HF_HOME" 2>/dev/null || true
cat <<EOF

Done. In a new terminal:
  source $CONDA_SH && conda activate $ENV_NAME
  export HF_HOME=$HF_HOME HF_HUB_DISABLE_XET=1
  cd $REPO
The lock file requirements-lock-autodl-$CUDA_VARIANT.txt records the installed versions.
EOF
