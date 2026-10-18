#!/usr/bin/env bash
# Seeding pilot: build the seeding set, SFT, measure the seeded propensity, then GRPO from the
# seeded policy (docs/findings.md 3.13 and open question 7).
# Run on the GPU machine after `source /workspace/loopholes-env.sh`, from the repository root:
#   bash scripts/run_seeded_pilot.sh [name] [judge] [pool] [seed]
# Defaults: seed2pct r1_distill_verifier_1p5b hard 0. Each stage is skipped when its output exists,
# so rerunning resumes after the last finished stage. Outputs: runs/sft/<name>/ (data, model,
# propensity, likelihood) and runs/grpo/<judge>-<pool>-s<seed>-<name>/.

set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
NAME=${1:-seed2pct}; JUDGE=${2:-r1_distill_verifier_1p5b}; POOL=${3:-hard}; SEED=${4:-0}
DIR=runs/sft/$NAME
FPR=results/stage1/fpr_by_family.csv
export VLLM_USE_FLASHINFER_SAMPLER=0
step() { echo; echo "========== [$(date '+%H:%M:%S')] $*"; }

step "1/5 seeding set ($DIR/data.jsonl)"
if [ -f "$DIR/data.jsonl" ]; then echo "exists, skipped"; else python -m src.train.seed_data --name "$NAME"; fi

step "2/5 SFT ($DIR/model)"
if [ -f "$DIR/model/config.json" ]; then echo "exists, skipped"; else python -m src.train.sft --name "$NAME" --seed "$SEED"; fi

step "3/5 sampled propensity of the seeded policy ($DIR/propensity)"
if [ -f "$DIR/propensity/risk_scores.csv" ]; then echo "exists, skipped"
else python -m src.audit.sampled_propensity --policy "$DIR/model" --run-dir "$DIR/propensity" --fpr "$FPR"; fi

step "4/5 string likelihood of the seeded policy ($DIR/likelihood)"
if [ -f "$DIR/likelihood/propensity.csv" ]; then echo "exists, skipped"
else python -m src.audit.likelihood --policy "$DIR/model" --run-dir "$DIR/likelihood" --difficulty-run none \
       --fpr "$FPR" || echo "!! likelihood failed; GRPO still runs (it uses the sampled propensity)"; fi

step "5/5 GRPO from the seeded policy"
bash scripts/run_grpo.sh "$JUDGE" "$POOL" "$SEED" --policy-model "$DIR/model" \
  --risk-csv "$DIR/propensity/risk_scores.csv" --tag "$NAME"
