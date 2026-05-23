#!/usr/bin/env bash
# CoT baseline (run_basic_overwrite.py --baseline_only)
# Usage:
#   ./steering/run_cot_baseline.sh <GPU> <DATASET> <LOG_DIR> [NUM_SHARDS]
#
# Examples:
#   ./steering/run_cot_baseline.sh 0 amc23 log2/cot_amc23
#   ./steering/run_cot_baseline.sh 0,1 gpqa_diamond log2/cot_gpqa 2
#   ./steering/run_cot_baseline.sh 0 ineqmath log2/cot_ineq 1
#
# GPU: single id (0) or comma list (0,1) — one process per id, same LOG_DIR.

set -euo pipefail

GPU="${1:?GPU required (e.g. 0 or 0,1)}"
DATASET="${2:?DATASET required (amc23|aime24|aime25|math500|gpqa_diamond|ineqmath)}"
LOG="${3:?LOG_DIR required (e.g. log2/cot_amc23)}"
NUM_SHARDS="${4:-}"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
mkdir -p "$LOG/judge_reasons"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

if [[ -x "$REPO/.venv/bin/python3" ]]; then
  PYCMD="$REPO/.venv/bin/python3"
else
  PYCMD="${CONDA_PREFIX}/bin/python"
fi

MODEL="${MODEL:-Open-Reasoner-Zero/Open-Reasoner-Zero-7B}"
MAX_TOKEN="${MAX_TOKEN:-4000}"

case "$DATASET" in
  amc23) NUM_EX=40 ;;
  aime24|aime25) NUM_EX=30 ;;
  math500) NUM_EX=500 ;;
  gpqa_diamond) NUM_EX=198 ;;
  ineqmath) NUM_EX=100 ;;
  *) echo "Unknown DATASET: $DATASET"; exit 1 ;;
esac

IFS=',' read -ra GPUS <<< "$GPU"
NG="${NUM_SHARDS:-${#GPUS[@]}}"
if [[ "$NG" -ne "${#GPUS[@]}" ]]; then
  echo "NUM_SHARDS=$NG must match number of GPUs (${#GPUS[@]})"
  exit 1
fi

EXTRA=()
if [[ "$DATASET" == "ineqmath" ]]; then
  EXTRA+=(--ineqmath_split dev --ineqmath_test_limit 100)
fi

echo "dataset=$DATASET num_examples=$NUM_EX gpus=${GPUS[*]} log=$LOG"
pids=()
for ((i = 0; i < NG; i++)); do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$PYCMD" -u steering/run_basic_overwrite.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --baseline_only \
    --num_examples "$NUM_EX" \
    --max_token "$MAX_TOKEN" \
    --save_judge_reason \
    --judge_reason_dir "$LOG/judge_reasons" \
    --shard_id "$i" \
    --num_shards "$NG" \
    "${EXTRA[@]}" \
    > "$LOG/log_worker_${i}.txt" 2>&1 &
  pids+=($!)
  echo "  started shard $i on GPU ${GPUS[$i]} -> $LOG/log_worker_${i}.txt"
done

for pid in "${pids[@]}"; do wait "$pid"; done
echo "done. summary:"
grep -E 'Final Report|Base :' "$LOG"/log_worker_*.txt || true
