#!/usr/bin/env bash
# One GRPO run: start the judge's vLLM server, train against it, stop the server.
# Run on the GPU machine after `source ~/loopholes-env.sh`, from the repository root:
#   bash scripts/run_grpo.sh <judge> <pool> <seed> [extra arguments for src.train.grpo]
# Examples:
#   bash scripts/run_grpo.sh engineered:no_box hard 0            # pilot Q2, no server needed
#   bash scripts/run_grpo.sh r1_distill_verifier_1p5b hard 0     # pilot Q3, positive control
# Logs go to runs/grpo/<judge>-<pool>-s<seed>/ (train.out, judge_server.log and the JSONL files).

set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
if [ $# -lt 3 ]; then sed -n 2,9p "$0"; exit 1; fi
JUDGE=$1; POOL=$2; SEED=$3; shift 3
RUN_DIR="runs/grpo/${JUDGE//:/-}-${POOL}-s${SEED}"
export VLLM_USE_FLASHINFER_SAMPLER=0

if [ ! -f data/processed/pool_candidates.jsonl ]; then
  echo "data/processed/pool_candidates.jsonl is missing; build it first: python -m src.data.build_pool"; exit 1
fi
mkdir -p "$RUN_DIR"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv | tee "$RUN_DIR/gpu.txt"

SERVER_PID=""
stop_server() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "stopping judge server (pid $SERVER_PID)"; kill "$SERVER_PID"; wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap stop_server EXIT

if [[ "$JUDGE" != engineered:* ]]; then
  CMD=$(python -m src.judges.server --judge "$JUDGE")
  echo "[$(date '+%H:%M:%S')] starting judge server: $CMD"
  # shellcheck disable=SC2086
  $CMD > "$RUN_DIR/judge_server.log" 2>&1 &
  SERVER_PID=$!
fi

echo "[$(date '+%H:%M:%S')] training $JUDGE on $POOL, seed $SEED -> $RUN_DIR"
python -m src.train.grpo --judge "$JUDGE" --pool "$POOL" --seed "$SEED" --run-dir "$RUN_DIR" "$@" 2>&1 | tee -a "$RUN_DIR/train.out"
echo "[$(date '+%H:%M:%S')] finished; summarize with: python -m src.monitor.events $RUN_DIR --tau 0.05"
