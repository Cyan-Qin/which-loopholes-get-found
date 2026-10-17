#!/usr/bin/env bash
# One-time setup of a GPU machine: GCP, Vast.ai, AutoDL or any Linux box with an NVIDIA driver.
#
#   bash setup_gpu.sh                         # detect platform and CUDA wheels automatically
#   CUDA_VARIANT=cu129 bash setup_gpu.sh      # force CUDA 12.9 wheels (driver supports only 12.9)
#   SKIP_MODELS=1 bash setup_gpu.sh           # environment and data only
#   HF_ENDPOINT=https://hf-mirror.com bash setup_gpu.sh   # use the Hugging Face mirror
#   PIP_REQUIREMENTS=requirements.txt bash setup_gpu.sh   # CPU-only check without vllm/trl
#
# Layout (DATA is /root/autodl-tmp on AutoDL, /workspace on Vast, $HOME elsewhere):
#   $DATA/which-loopholes-get-found   code, data/ and runs/
#   $DATA/hf_cache                    model weights (HF_HOME)
#   $HOME/envs/loopholes              Python 3.12 environment (venv, or conda on AutoDL)
#   $DATA/loopholes-env.sh            `source` it in every new shell
# A full log is appended to $DATA/setup.log.

set -euo pipefail

if [ -z "${DATA:-}" ]; then
  if [ -d /root/autodl-tmp ]; then DATA=/root/autodl-tmp
  elif [ -d /workspace ] && [ -w /workspace ]; then DATA=/workspace
  else DATA="$HOME"; fi
fi
REPO_URL="${REPO_URL:-https://github.com/justLxy/which-loopholes-get-found.git}"
REPO="$DATA/which-loopholes-get-found"
ENV_DIR="${ENV_DIR:-$HOME/envs/loopholes}"
PIP_REQUIREMENTS="${PIP_REQUIREMENTS:-requirements-gpu.txt}"
VLLM_VERSION=0.31.0
TORCH_VERSION=2.13.0
EXPECTED_POOL_MD5=fea6cc4341e9bbf85135915c50f0dba2

mkdir -p "$DATA"
exec > >(tee -a "$DATA/setup.log") 2>&1
step() { echo; echo "========== [$(date '+%H:%M:%S')] $*"; }
# AutoDL academic acceleration for GitHub and huggingface.co; a no-op elsewhere.
# shellcheck source=/dev/null
turbo_on() { if [ -f /etc/network_turbo ]; then source /etc/network_turbo >/dev/null; fi; }
turbo_off() { if [ -f /etc/network_turbo ]; then unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; fi; }
md5() { if command -v md5sum >/dev/null; then md5sum "$1" | cut -d' ' -f1; else command md5 -q "$1"; fi; }
SUDO=""; if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null; then SUDO=sudo; fi

if [ -d /root/autodl-tmp ]; then PLATFORM=autodl
elif grep -qs "Google Compute Engine" /sys/class/dmi/id/product_name; then PLATFORM=gcp
elif [ -n "${VAST_CONTAINERLABEL:-}${CONTAINER_ID:-}" ]; then PLATFORM=vast
else PLATFORM=generic; fi

step "0/7 machine ($PLATFORM)"
CUDA_DRIVER=""
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  CUDA_DRIVER=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1)
  echo "driver supports CUDA up to $CUDA_DRIVER"
else
  echo "no GPU visible (fine for AutoDL no-GPU mode or a CPU-only check)"
fi
if [ -z "${CUDA_VARIANT:-}" ]; then
  if [ -z "$CUDA_DRIVER" ]; then CUDA_VARIANT=cu130
  else
    level=$(awk -v v="$CUDA_DRIVER" 'BEGIN { split(v, a, "."); print a[1] * 100 + a[2] }')
    if [ "$level" -ge 1300 ]; then CUDA_VARIANT=cu130
    elif [ "$level" -ge 1209 ]; then CUDA_VARIANT=cu129
    else echo "driver supports CUDA $CUDA_DRIVER; vllm $VLLM_VERSION needs 12.9 or later"; exit 1; fi
  fi
fi
case "$CUDA_VARIANT" in cu130|cu129) ;; *) echo "CUDA_VARIANT must be cu130 or cu129"; exit 1 ;; esac
echo "DATA=$DATA  ENV_DIR=$ENV_DIR  CUDA_VARIANT=$CUDA_VARIANT  requirements=$PIP_REQUIREMENTS"
df -h "$DATA" | sed 1d

step "1/7 code"
turbo_on
if [ -d "$REPO/.git" ]; then git -C "$REPO" pull --ff-only; else git clone "$REPO_URL" "$REPO"; fi
turbo_off
git -C "$REPO" log -1 --format='commit %h %s'

step "2/7 Python 3.12 environment"
if [ ! -x "$ENV_DIR/bin/python" ]; then
  mkdir -p "$(dirname "$ENV_DIR")"
  if command -v python3.12 >/dev/null; then
    if ! python3.12 -c "import ensurepip" 2>/dev/null; then
      $SUDO apt-get update -qq && $SUDO apt-get install -y -qq python3.12-venv
    fi
    python3.12 -m venv "$ENV_DIR"
  elif command -v conda >/dev/null; then
    # conda-forge avoids the Anaconda default-channel terms prompt in non-interactive shells
    conda create -y -p "$ENV_DIR" -c conda-forge --override-channels python=3.12
  else
    echo "need python3.12 (with venv) or conda; on Ubuntu: sudo apt-get install python3.12-venv"; exit 1
  fi
fi
PY="$ENV_DIR/bin/python"
"$PY" --version

step "3/7 packages ($CUDA_VARIANT)"
cd "$REPO"
"$PY" -m pip install -q --upgrade pip
if [ "$CUDA_VARIANT" = cu129 ] && [ "$PIP_REQUIREMENTS" = requirements-gpu.txt ]; then
  # CUDA 12.9 builds: torch from the PyTorch index, vLLM from its GitHub release.
  "$PY" -m pip install "torch==$TORCH_VERSION" --index-url https://download.pytorch.org/whl/cu129
  turbo_on
  "$PY" -m pip install \
    "https://github.com/vllm-project/vllm/releases/download/v$VLLM_VERSION/vllm-$VLLM_VERSION+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
    --extra-index-url https://download.pytorch.org/whl/cu129
  turbo_off
fi
"$PY" -m pip install -q -r "$PIP_REQUIREMENTS"
"$PY" -m pip freeze > "requirements-lock-$PLATFORM-$CUDA_VARIANT.txt"
"$PY" - <<'EOF'
import importlib
for name in ["torch", "transformers", "vllm", "trl", "datasets", "math_verify"]:
    try:
        mod = importlib.import_module(name)
        print(f"  {name:<13} {getattr(mod, '__version__', 'ok')}")
    except Exception as e:  # report and continue: vllm is absent in CPU-only checks
        print(f"  {name:<13} not available: {type(e).__name__}: {str(e)[:80]}")
import torch
print(f"  torch built for CUDA {torch.version.cuda}; GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    print(f"  {torch.cuda.get_device_name(0)}: bf16 matmul ok, norm {float((x @ x).float().norm()):.3g}")
EOF

cat > "$DATA/loopholes-env.sh" <<EOF
# source this file in every new shell
source "$ENV_DIR/bin/activate" 2>/dev/null || export PATH="$ENV_DIR/bin:\$PATH"
export HF_HOME="$DATA/hf_cache" HF_HUB_DISABLE_XET=1
cd "$REPO"
EOF
export HF_HOME="$DATA/hf_cache" HF_HUB_DISABLE_XET=1

step "4/7 data"
turbo_on
"$PY" -m src.data.download_hacking | tail -2
"$PY" -m src.data.build_pool | tail -1
"$PY" -m src.detectors.build_vocab
turbo_off
POOL_MD5=$(md5 data/processed/pool_candidates.jsonl)
if [ "$POOL_MD5" = "$EXPECTED_POOL_MD5" ]; then
  echo "candidate pool identical to the local build ($POOL_MD5)"
else
  echo "WARNING: candidate pool differs from the local build ($POOL_MD5 vs $EXPECTED_POOL_MD5)"
fi
if git diff --quiet src/detectors/vocab.txt; then echo "vocab.txt identical to the committed one"
else echo "WARNING: vocab.txt differs from the committed one"; fi

step "5/7 models"
if [ "${SKIP_MODELS:-0}" = 1 ]; then
  echo "skipped (SKIP_MODELS=1)"
else
  if [ -z "${HF_ENDPOINT:-}" ]; then turbo_on; fi
  "$PY" scripts/download_models.py
  turbo_off
fi

step "6/7 tests"
"$PY" -m pytest -q 2>&1 | tail -3

step "7/7 summary"
df -h "$DATA" | sed 1d
du -sh "$HF_HOME" 2>/dev/null || true
cat <<EOF

Done. In every new shell run:
  source $DATA/loopholes-env.sh
Installed versions are recorded in requirements-lock-$PLATFORM-$CUDA_VARIANT.txt.
EOF
