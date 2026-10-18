#!/usr/bin/env bash
# One GRPO run: start the judge's vLLM server, train against it, stop the server, summarize.
# Run on the GPU machine after `source ~/loopholes-env.sh`, from the repository root:
#   bash scripts/run_grpo.sh <judge> <pool> <seed> [extra arguments for src.train.grpo]
# Examples:
#   bash scripts/run_grpo.sh engineered:no_box hard 0            # pilot Q2, no server needed
#   bash scripts/run_grpo.sh r1_distill_verifier_1p5b hard 0     # pilot Q3, positive control
#   bash scripts/run_grpo.sh xverify_0p5b easy 0 --gpu-test      # short integration test (fits a 24GB L4)
# Logs go to runs/grpo/<judge>-<pool>-s<seed>[-gputest]/: train.out, judge_server.log, gpu_mem.csv
# (nvidia-smi every 2 s), summary.txt and the JSONL files.

set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
if [ $# -lt 3 ]; then sed -n 2,10p "$0"; exit 1; fi
JUDGE=$1; POOL=$2; SEED=$3; shift 3
SUFFIX=""; SERVER_ARGS=()
for a in "$@"; do
  if [ "$a" = "--gpu-test" ]; then SUFFIX="-gputest"; SERVER_ARGS+=(--gpu-test); fi
done
RUN_DIR="runs/grpo/${JUDGE//:/-}-${POOL}-s${SEED}${SUFFIX}"
export VLLM_USE_FLASHINFER_SAMPLER=0

if [ ! -f data/processed/pool_candidates.jsonl ]; then
  echo "data/processed/pool_candidates.jsonl is missing; build it first: python -m src.data.build_pool"; exit 1
fi
if compgen -G "$RUN_DIR/*.jsonl" > /dev/null; then
  echo "$RUN_DIR already has logs; move it away or use another seed"; exit 1
fi
mkdir -p "$RUN_DIR"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv | tee "$RUN_DIR/gpu.txt"

SERVER_PID=""; MEM_PID=""
cleanup() {
  if [ -n "$MEM_PID" ]; then kill "$MEM_PID" 2>/dev/null || true; fi
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "stopping judge server (pid $SERVER_PID)"; kill "$SERVER_PID"; wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 2 > "$RUN_DIR/gpu_mem.csv" &
MEM_PID=$!

if [[ "$JUDGE" != engineered:* ]]; then
  CMD=$(python -m src.judges.server --judge "$JUDGE" ${SERVER_ARGS[@]+"${SERVER_ARGS[@]}"})
  PORT=$(echo "$CMD" | sed -E 's/.*--port ([0-9]+).*/\1/')
  echo "[$(date '+%H:%M:%S')] starting judge server: $CMD"
  # shellcheck disable=SC2086
  $CMD > "$RUN_DIR/judge_server.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null; then break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "!! judge server exited; last lines of $RUN_DIR/judge_server.log:"; tail -30 "$RUN_DIR/judge_server.log"; exit 1
    fi
    sleep 5
  done
  if ! curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null; then
    echo "!! judge server not ready after 15 minutes; see $RUN_DIR/judge_server.log"; exit 1
  fi
  echo "[$(date '+%H:%M:%S')] judge server ready on port $PORT"
fi

echo "[$(date '+%H:%M:%S')] training $JUDGE on $POOL, seed $SEED -> $RUN_DIR"
set +e
python -m src.train.grpo --judge "$JUDGE" --pool "$POOL" --seed "$SEED" --run-dir "$RUN_DIR" "$@" 2>&1 | tee -a "$RUN_DIR/train.out"
STATUS=${PIPESTATUS[0]}
set -e
echo; echo "========== [$(date '+%H:%M:%S')] summary"
python -m src.train.summary "$RUN_DIR" | tee "$RUN_DIR/summary.txt" || true
if [ "$STATUS" -ne 0 ]; then
  echo "!! training failed (exit $STATUS); the error is above and in $RUN_DIR/train.out"
  if [ -n "$SERVER_PID" ]; then echo "last lines of the judge server log:"; tail -15 "$RUN_DIR/judge_server.log"; fi
  exit "$STATUS"
fi
echo "events: python -m src.monitor.events $RUN_DIR --tau 0.05"
