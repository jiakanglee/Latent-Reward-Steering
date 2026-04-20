#!/bin/bash
# 120 题（非双对）上扫 steering：num_steps ∈ [1,5]，step_size ∈ [0.1,1.5] 步长 0.1 → 5×15=75 组。
# reward_threshold=0.8，confidence_threshold=0.69（写死在下方）。
# 所有开关均在下方「=== 配置 ===」里改；直接 sbatch，勿在终端 export。
#
# 输出：log2/math500_steer_sweep120_${SLURM_ARRAY_JOB_ID}/task_<id>_ns<N>_ss<S>/judge/
# Slurm 自身日志：集中到 log2/slurm_sweep_min/（每 array 任务一个小 .log，不重复 grep worker）
#
# 并发：默认一次最多 3 个 array task（按需改 % 后的数字或删掉 %3）
#
#SBATCH --job-name=M500_Sweep120
#SBATCH --array=0-74%3
#SBATCH --output=log2/slurm_sweep_min/sweep120_%A_%a.log
#SBATCH --error=log2/slurm_sweep_min/sweep120_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=ilab2
#SBATCH --gres=gpu:a4500:8

source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p log2 log2/slurm_sweep_min
export PYTHONPATH=$PWD:$PYTHONPATH

# ========================= 配置（只改这里） =========================
RUN_REF="$PWD/log2/math500_steer_overwrite_124530"
IDX_FILE="$RUN_REF/math500_not_both_ok_indices.json"
PRIOR_JUDGE="$RUN_REF/judge"
# 1=每组都跑 Base+Steer（极慢）；0=仅 Steer，prior 用 PRIOR_JUDGE
STEER_SWEEP_FULL=0
REWARD_THRESHOLD=0.8
CONFIDENCE_THRESHOLD=0.69
# judge JSON 是否带全文 / prior base 元数据（占磁盘）
SAVE_JUDGE_WITH_TEXT=0
SAVE_JUDGE_PRIOR_BASE_META=0
ATTN_IMPLEMENTATION=flash_attention_2
MODEL_NAME="Open-Reasoner-Zero/Open-Reasoner-Zero-7B"
# 与全量 MATH-500 steer 一致：AIME 上训练的 reward + 更长生成
REWARD_MODEL="$PWD/transformer_reward_model_aime_best.pt"
SAE_LAYER=20
N_CLUSTERS=10
MAX_TOKEN=4000
# 0=用 conda python；1=强制项目 .venv
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
  echo "FATAL: 找不到题号文件: $IDX_FILE"
  echo "  请先: python export_math500_not_both_ok_indices.py $PRIOR_JUDGE"
  exit 1
fi
if [ ! -d "$PRIOR_JUDGE" ]; then
  echo "FATAL: 找不到 prior judge: $PRIOR_JUDGE"
  exit 1
fi

TID=${SLURM_ARRAY_TASK_ID:-0}
read -r NUM_STEPS STEP_SIZE <<< "$("$PYCMD" -c "tid=int('${TID}'); print(tid//15+1, round(0.1*(tid%15+1), 1))")"

SS_STR=$(printf '%.1f' "$STEP_SIZE" | tr '.' 'p')
_ARRAY_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
LOG_BASE="log2/math500_steer_sweep120_${_ARRAY_ID}"
LOG_DIR="${LOG_BASE}/task_${TID}_ns${NUM_STEPS}_ss${SS_STR}"
mkdir -p "$LOG_DIR/judge"

export ATTN_IMPLEMENTATION="$ATTN_IMPLEMENTATION"

echo "IDX_FILE=$IDX_FILE"
echo "PRIOR_JUDGE=$PRIOR_JUDGE"
echo "ARRAY_TASK_ID=$TID  NUM_STEPS=$NUM_STEPS  STEP_SIZE=$STEP_SIZE"
echo "REWARD_THRESHOLD=$REWARD_THRESHOLD  CONFIDENCE_THRESHOLD=$CONFIDENCE_THRESHOLD"
echo "LOG_DIR=$LOG_DIR  STEER_SWEEP_FULL=$STEER_SWEEP_FULL"

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
  CUDA_VISIBLE_DEVICES=$i $PYCMD -u run_basic_overwrite.py \
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
