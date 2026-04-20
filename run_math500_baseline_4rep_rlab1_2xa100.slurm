#!/bin/bash
# MATH-500 baseline（greedy）：Open-Reasoner-Zero-7B，连跑 4 次（array 0–3），2×A100 双卡分片。
# 节点：rlab1；分片：question_idx % 2 == shard_id。
# 输出：log2/math500_baseline_4rep_rlab1_${SLURM_ARRAY_JOB_ID}/rep_<0..3>/
# 汇总：python summarize_math500_baseline_reps.py log2/math500_baseline_4rep_rlab1_<ARRAY_JOB_ID>
#
#SBATCH --job-name=M500_Base4r1A100
#SBATCH --array=0-3
#SBATCH --output=log2/slurm_sweep_min/math500_baseline_4rep_rlab1_%A_%a.log
#SBATCH --error=log2/slurm_sweep_min/math500_baseline_4rep_rlab1_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=rlab1
#SBATCH --gres=gpu:a100:2

source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p log2 log2/slurm_sweep_min
export PYTHONPATH=$PWD:$PYTHONPATH

MODEL_NAME="Open-Reasoner-Zero/Open-Reasoner-Zero-7B"
MAX_TOKEN=4000
NUM_EXAMPLES=500
export ATTN_IMPLEMENTATION=flash_attention_2
DEBUG_PRINT=0
USE_PROJECT_VENV=0
MATH500_ALLOW_USER_SITE=0

unset PYTHONNOUSERSITE
if [ "${USE_PROJECT_VENV}" = "1" ] && [ -x "$PWD/.venv/bin/python3" ]; then
  PYCMD="$PWD/.venv/bin/python3"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
  PYCMD="${CONDA_PREFIX}/bin/python"
elif [ -x "$PWD/.venv/bin/python3" ]; then
  PYCMD="$PWD/.venv/bin/python3"
else
  PYCMD="python"
fi
if [ "${MATH500_ALLOW_USER_SITE}" != "1" ]; then
  export PYTHONNOUSERSITE=1
fi

MV_ERR="$("$PYCMD" -c "from math_verify import parse, verify" 2>&1)" || {
  echo "FATAL: math_verify"
  echo "$MV_ERR"
  exit 1
}
unset MV_ERR

RID=${SLURM_ARRAY_TASK_ID:-0}
AJOB="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
LOG_ROOT="log2/math500_baseline_4rep_rlab1_${AJOB}"
LOG_DIR="${LOG_ROOT}/rep_${RID}"
mkdir -p "${LOG_DIR}/judge"

export ATTN_IMPLEMENTATION="$ATTN_IMPLEMENTATION"

NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then NUM_GPUS=1; fi

EXTRA_PY_ARGS=(--save_judge_reason --judge_reason_dir "${LOG_DIR}/judge")
if [ "$DEBUG_PRINT" = "1" ]; then
  EXTRA_PY_ARGS+=(--print_response)
fi

echo "rep=${RID} LOG_DIR=${LOG_DIR} NUM_GPUS=${NUM_GPUS} nodelist=rlab1 MODEL=${MODEL_NAME} max_token=${MAX_TOKEN}"

pids=()
for ((i=0; i<NUM_GPUS; i++)); do
  CUDA_VISIBLE_DEVICES=$i $PYCMD -u run_basic_overwrite.py \
    --model "$MODEL_NAME" \
    --dataset math500 \
    --baseline_only \
    --num_examples "$NUM_EXAMPLES" \
    --max_token "$MAX_TOKEN" \
    --shard_id $i \
    --num_shards $NUM_GPUS \
    "${EXTRA_PY_ARGS[@]}" \
    > "${LOG_DIR}/log_worker_${i}.txt" 2>&1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done

grep -E 'Final Report|Base :' "${LOG_DIR}"/log_worker_*.txt > "${LOG_DIR}/final_report_grep.txt" 2>/dev/null || true
echo "Done rep=${RID} LOG_DIR=${LOG_DIR}"
grep -E 'Final Report|Base :' "${LOG_DIR}"/log_worker_*.txt || true
