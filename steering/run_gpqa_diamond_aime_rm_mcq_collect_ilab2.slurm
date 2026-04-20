#!/bin/bash
# GPQA-Diamond + AIME RM + MCQ 规则判题 + collect 汇总
# 提交: 在仓库根目录  sbatch steering/run_gpqa_diamond_aime_rm_mcq_collect_ilab2.slurm
#
#SBATCH --job-name=GPQA_MCQ_AIME
#SBATCH --output=log2/gpqa_diamond_aime_mcq_%j.out
#SBATCH --error=log2/gpqa_diamond_aime_mcq_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=14G
#SBATCH --time=72:00:00
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:a5000:2

set -euo pipefail

_SL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "${_SL}/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p log2
export PYTHONPATH=$PWD:$PYTHONPATH

export PYTHONNOUSERSITE=1
if [ -n "${CONDA_INIT_SH:-}" ] && [ -f "${CONDA_INIT_SH}" ]; then
  # shellcheck source=/dev/null
  source "${CONDA_INIT_SH}"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV:-stllms_env}"

if [ -x "$PWD/.venv/bin/python3" ]; then PYCMD="$PWD/.venv/bin/python3"; else PYCMD="python"; fi

REWARD_MODEL="${REWARD_MODEL:-transformer_reward_model_aime_best.pt}"
REWARD="${REWARD:-0.9}"
MAX_TOKEN="${MAX_TOKEN:-4000}"
NSTEPS="${NSTEPS:-4}"
CONF="${CONF:-0.72}"
STEP="${STEP:-1.15}"

LOG_DIR="log2/gpqa_diamond_aime_mcq_${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR/judge_reasons"

SAVE_JUDGE_WITH_TEXT="${SAVE_JUDGE_WITH_TEXT:-1}"

NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr ',' '\n' | sed '/^$/d' | wc -l)
if [ "${NUM_GPUS:-0}" -eq 0 ]; then NUM_GPUS=2; fi

echo "GPQA-Diamond | LOG_DIR=$LOG_DIR  SAVE_JUDGE_WITH_TEXT=$SAVE_JUDGE_WITH_TEXT"
echo "Date: $(date)"

EXTRA=(--save_judge_with_text)
if [ "$SAVE_JUDGE_WITH_TEXT" != "1" ]; then
  EXTRA=()
fi

pids=()
for ((i=0; i<NUM_GPUS; i++)); do
  CUDA_VISIBLE_DEVICES=$i $PYCMD -u steering/run_basic_overwrite.py \
    --model "Open-Reasoner-Zero/Open-Reasoner-Zero-7B" \
    --sae_layer 20 \
    --n_clusters 10 \
    --reward_model_path "$REWARD_MODEL" \
    --step_size "$STEP" \
    --num_steps "$NSTEPS" \
    --max_token "$MAX_TOKEN" \
    --reward_threshold "$REWARD" \
    --confidence_threshold "$CONF" \
    --dataset gpqa_diamond \
    --num_examples 198 \
    --no-monitor \
    --save_judge_reason \
    "${EXTRA[@]}" \
    --judge_reason_dir "${LOG_DIR}/judge_reasons" \
    --shard_id "$i" \
    --num_shards "$NUM_GPUS" \
    > "${LOG_DIR}/log_worker_${i}.txt" 2>&1 &
  pids+=($!)
done

for pid in "${pids[@]}"; do wait "$pid"; done

RESULTS_FILE="log2/gpqa_diamond_aime_mcq_results_${SLURM_JOB_ID}.txt"
echo "=== GPQA-Diamond MCQ ===" > "$RESULTS_FILE"
$PYCMD collect_sweep_results.py \
  --log_dir "$LOG_DIR" \
  --output "$RESULTS_FILE" \
  --label "t${MAX_TOKEN}_s${STEP}_c${CONF}_n${NSTEPS}_mcq"

$PYCMD collect_gpqa_mcqa_summary.py "$LOG_DIR"

cat "$RESULTS_FILE"
echo "CSV: ${LOG_DIR}/gpqa_mcqa_summary.csv"
echo "Date: $(date)"
