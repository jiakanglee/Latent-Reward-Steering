#!/bin/bash
# MATH-500：117 题子集（quad125844 harm∪save∪both_bad 并集）上扫 steering。
# 网格：num_steps ∈ {1,2,3,4}，step_size ∈ {0.1,0.2,...,1.5} → 4×15=60 组。
# steer_only + prior judge（与 quad task_0 同分布的 base），reward/conf 与 top3 一致 0.8 / 0.69。
# 题单：aggregate_quad_steer_cells_for_sweep.py 生成的 sweep_cell_pools_union_three.json
#
# 汇总（ARRAY_JOB_ID 换成实际）:
#   python summarize_math500_steer_sweep.py log2/math500_steer_sweep_union117_<ARRAY_JOB_ID>
#
# 并发：按需改 #SBATCH --array=0-59%K
#
#SBATCH --job-name=M500_SwU117
#SBATCH --array=0-59%6
#SBATCH --output=log2/slurm_sweep_min/sweep_union117_%A_%a.log
#SBATCH --error=log2/slurm_sweep_min/sweep_union117_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:a100:3

source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p log2 log2/slurm_sweep_min
export PYTHONPATH=$PWD:$PYTHONPATH

# ========================= 配置 =========================
QUAD="$PWD/log2/math500_steer_quad500_125844"
IDX_FILE="$QUAD/sweep_cell_pools_union_three.json"
PRIOR_JUDGE="$QUAD/task_0_ns3_ss0p3/judge"
STEER_SWEEP_FULL=0
REWARD_THRESHOLD=0.8
CONFIDENCE_THRESHOLD=0.69
SAVE_JUDGE_WITH_TEXT=0
SAVE_JUDGE_PRIOR_BASE_META=0
ATTN_IMPLEMENTATION=flash_attention_2
MODEL_NAME="Open-Reasoner-Zero/Open-Reasoner-Zero-1.5B"
REWARD_MODEL="$PWD/PST/transformer_reward_model_aime_1.5B_best.pt"
SAE_LAYER=8
N_CLUSTERS=5
MAX_TOKEN=4000
USE_PROJECT_VENV=0
MATH500_ALLOW_USER_SITE=0
# =====================================================================

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

if [ ! -f "$REWARD_MODEL" ]; then
  echo "FATAL: 找不到 reward 权重: $REWARD_MODEL"
  exit 1
fi

if [ ! -f "$IDX_FILE" ]; then
  echo "FATAL: 找不到题单: $IDX_FILE"
  echo "  python aggregate_quad_steer_cells_for_sweep.py $QUAD -o $QUAD/sweep_cell_pools"
  exit 1
fi
if [ ! -d "$PRIOR_JUDGE" ]; then
  echo "FATAL: 找不到 prior judge: $PRIOR_JUDGE"
  exit 1
fi

TID=${SLURM_ARRAY_TASK_ID:-0}
# TID 0..59: ns = 1 + tid//15, ss = 0.1 * (tid%15 + 1)
read -r NUM_STEPS STEP_SIZE <<< "$("$PYCMD" -c "tid=int('${TID}'); print(tid//15+1, round(0.1*(tid%15+1), 1))")"

SS_STR=$(printf '%.1f' "$STEP_SIZE" | tr '.' 'p')
_ARRAY_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
LOG_BASE="log2/math500_steer_sweep_union117_${_ARRAY_ID}"
LOG_DIR="${LOG_BASE}/task_${TID}_ns${NUM_STEPS}_ss${SS_STR}"
mkdir -p "$LOG_DIR/judge"

export ATTN_IMPLEMENTATION="$ATTN_IMPLEMENTATION"

echo "IDX_FILE=$IDX_FILE"
echo "PRIOR_JUDGE=$PRIOR_JUDGE"
echo "ARRAY_TASK_ID=$TID  NUM_STEPS=$NUM_STEPS  STEP_SIZE=$STEP_SIZE"
echo "LOG_DIR=$LOG_DIR"

NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then NUM_GPUS=1; fi

export NUM_STEPS STEP_SIZE REWARD_THRESHOLD CONFIDENCE_THRESHOLD IDX_FILE PRIOR_JUDGE LOG_DIR STEER_SWEEP_FULL
META_JSON="$LOG_DIR/sweep_task_meta.json"
export META_JSON
"$PYCMD" -c "
import json, os
m = {
  'slurm_array_job_id': os.environ.get('SLURM_ARRAY_JOB_ID', ''),
  'slurm_array_task_id': int(os.environ.get('SLURM_ARRAY_TASK_ID', '0')),
  'num_steps': int(os.environ['NUM_STEPS']),
  'step_size': float(os.environ['STEP_SIZE']),
  'reward_threshold': float(os.environ['REWARD_THRESHOLD']),
  'confidence_threshold': float(os.environ['CONFIDENCE_THRESHOLD']),
  'example_idx_file': os.environ['IDX_FILE'],
  'prior_judge_dir': os.environ['PRIOR_JUDGE'],
  'subset': 'union117_harm_save_bothbad_quad125844',
  'steer_only': os.environ.get('STEER_SWEEP_FULL', '0') != '1',
  'log_dir': os.environ['LOG_DIR'],
}
path = os.environ['META_JSON']
open(path, 'w', encoding='utf-8').write(json.dumps(m, indent=2, ensure_ascii=False))
print('Wrote', path)
"

pids=()
for ((i=0; i<NUM_GPUS; i++)); do
  PY_ARGS=(--example_idx_file "$IDX_FILE" --num_steps "$NUM_STEPS" --step_size "$STEP_SIZE")
  PY_ARGS+=(--reward_threshold "$REWARD_THRESHOLD" --confidence_threshold "$CONFIDENCE_THRESHOLD")
  PY_ARGS+=(--no-monitor)
  PY_ARGS+=(--save_judge_reason --judge_reason_dir "${LOG_DIR}/judge")
  if [ "$SAVE_JUDGE_WITH_TEXT" = "1" ]; then
    PY_ARGS+=(--save_judge_with_text)
  fi
  if [ "$STEER_SWEEP_FULL" != "1" ]; then
    PY_ARGS+=(--steer_only --prior_judge_dir "$PRIOR_JUDGE")
    if [ "$SAVE_JUDGE_PRIOR_BASE_META" = "1" ]; then
      PY_ARGS+=(--save_judge_prior_base_meta)
    fi
  fi
  CUDA_VISIBLE_DEVICES=$i $PYCMD -u PST/run_basic_overwrite_update.py \

    --model "$MODEL_NAME" \
    --dataset math500 \
    --sae_layer "$SAE_LAYER" \
    --n_clusters "$N_CLUSTERS" \
    --reward_model_path "$REWARD_MODEL" \
    --max_token "$MAX_TOKEN" \
    --shard_id $i \
    --num_shards $NUM_GPUS \
    "${PY_ARGS[@]}" \
    > "${LOG_DIR}/log_worker_${i}.txt" 2>&1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done

grep -E 'Mode 2b|Final Report|Base :|Steer:' "${LOG_DIR}"/log_worker_*.txt > "${LOG_DIR}/final_report_grep.txt" 2>/dev/null || true
echo "Done LOG_DIR=$LOG_DIR  (grep -> ${LOG_DIR}/final_report_grep.txt)"
