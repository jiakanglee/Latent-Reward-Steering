#!/bin/bash
# AIME 2024：在 0.29 与 0.31 之间超细扫 step_size，步长 0.001（固定 num_steps=2）。
#
# 区间：[0.290, 0.310] 含端点 → 0.290 + k*0.001, k=0..20 → 21 个点 → #SBATCH --array=0-20
# 资源：3×A4500（与脚本内 gres 一致）；reward=0.9 conf=0.72；max_token=4000。
# 勿写死 nodelist 以免长期 PD（ReqNodeNotAvail / reservation）；若必须 ilab2：取消下行注释并删掉 #SBATCH --exclude 类行。
# judge 含全文：--save_judge_with_text（与 quad500 / fine_ns2 一致）。
#
# 输出：log2/aime24_steer_micro029_031_ns2_${SLURM_ARRAY_JOB_ID}/task_<id>_ns2_0p290/ … 0p310/
#
#SBATCH --job-name=AIME24_Micro029
#SBATCH --array=0-20%1
#SBATCH --output=log2/slurm_sweep_min/aime24_steer_micro029_031_%A_%a.log
#SBATCH --error=log2/slurm_sweep_min/aime24_steer_micro029_031_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=ilab1
#SBATCH --gres=gpu:a4500:5

source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p log2 log2/slurm_sweep_min
export PYTHONPATH=$PWD:$PYTHONPATH

# ========================= 公共配置 =========================
DATASET=aime24
REWARD_MODEL="$PWD/transformer_reward_model_aime_best.pt"
MAX_TOKEN=4000
NUM_EXAMPLES=30
NUM_STEPS=2
MODEL_NAME="Open-Reasoner-Zero/Open-Reasoner-Zero-7B"
export ATTN_IMPLEMENTATION=flash_attention_2
SAE_LAYER=20
N_CLUSTERS=10
REWARD_THRESHOLD=0.9
CONFIDENCE_THRESHOLD=0.72
SAVE_JUDGE_REASON=1
SAVE_JUDGE_WITH_TEXT=1
DEBUG_PRINT=0
STEER_TOKEN_DEBUG=0
USE_PROJECT_VENV=0
MATH500_ALLOW_USER_SITE=0
# =====================================================================

TID=${SLURM_ARRAY_TASK_ID:-0}
STEP_SIZE=$(python3 -c "tid=int('${TID}'); print(f'{round(0.290 + tid * 0.001, 3):.3f}')")
SS_TAG=$(python3 -c "tid=int('${TID}'); s=round(0.290 + tid * 0.001, 3); print(f'{s:.3f}'.replace('.', 'p'))")
_ARRAY_JOB="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
LOG_BASE="log2/aime24_steer_micro029_031_ns2_${_ARRAY_JOB}"
LOG_DIR="${LOG_BASE}/task_${TID}_ns${NUM_STEPS}_${SS_TAG}"
mkdir -p "$LOG_DIR"

if [ ! -f "$REWARD_MODEL" ]; then
  echo "FATAL: 找不到 reward 权重: $REWARD_MODEL"
  exit 1
fi

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
  echo "FATAL: 无法 from math_verify import parse, verify"
  echo "$MV_ERR"
  exit 1
}
unset MV_ERR

export NUM_STEPS STEP_SIZE REWARD_THRESHOLD CONFIDENCE_THRESHOLD MAX_TOKEN NUM_EXAMPLES REWARD_MODEL LOG_DIR DATASET SS_TAG LOG_BASE
META_JSON="$LOG_DIR/run_meta.json"
export META_JSON
"$PYCMD" - <<'PY'
import json, os
m = {
    "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", ""),
    "slurm_array_task_id": int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
    "dataset": os.environ["DATASET"],
    "num_steps": int(os.environ["NUM_STEPS"]),
    "step_size": float(os.environ["STEP_SIZE"]),
    "step_size_tag": os.environ["SS_TAG"],
    "micro_sweep_range": "[0.290, 0.310] step 0.001",
    "reward_threshold": float(os.environ["REWARD_THRESHOLD"]),
    "confidence_threshold": float(os.environ["CONFIDENCE_THRESHOLD"]),
    "max_token": int(os.environ["MAX_TOKEN"]),
    "num_examples": int(os.environ["NUM_EXAMPLES"]),
    "reward_model": os.environ["REWARD_MODEL"],
    "log_dir": os.environ["LOG_DIR"],
    "save_judge_with_text": True,
    "note": "AIME24 micro step [0.290,0.310] step 0.001, ns=2, unlimited partition, 3xA4500",
}
with open(os.environ["META_JSON"], "w", encoding="utf-8") as f:
    json.dump(m, f, indent=2, ensure_ascii=False)
print("Wrote", os.environ["META_JSON"])
PY

NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then NUM_GPUS=1; fi

echo "ARRAY_TASK_ID=$TID  NUM_STEPS=$NUM_STEPS  STEP_SIZE=$STEP_SIZE  SS_TAG=$SS_TAG  LOG_DIR=$LOG_DIR  NUM_GPUS=$NUM_GPUS"

pids=()
for ((i=0; i<NUM_GPUS; i++)); do
  PY_ARGS=()
  if [ "$STEER_TOKEN_DEBUG" != "1" ]; then
    PY_ARGS+=(--no-monitor)
  fi
  if [ "$DEBUG_PRINT" = "1" ]; then
    PY_ARGS+=(--print_response)
  fi
  if [ "$SAVE_JUDGE_REASON" = "1" ]; then
    mkdir -p "${LOG_DIR}/judge"
    PY_ARGS+=(--save_judge_reason --judge_reason_dir "${LOG_DIR}/judge")
    if [ "$SAVE_JUDGE_WITH_TEXT" = "1" ]; then
      PY_ARGS+=(--save_judge_with_text)
    fi
  fi
  CUDA_VISIBLE_DEVICES=$i $PYCMD -u run_basic_overwrite.py \
    --model "$MODEL_NAME" \
    --dataset "$DATASET" \
    --sae_layer "$SAE_LAYER" \
    --n_clusters "$N_CLUSTERS" \
    --reward_model_path "$REWARD_MODEL" \
    --step_size "$STEP_SIZE" \
    --num_steps "$NUM_STEPS" \
    --reward_threshold "$REWARD_THRESHOLD" \
    --confidence_threshold "$CONFIDENCE_THRESHOLD" \
    --num_examples "$NUM_EXAMPLES" \
    --max_token "$MAX_TOKEN" \
    --shard_id $i \
    --num_shards $NUM_GPUS \
    "${PY_ARGS[@]}" \
    > "${LOG_DIR}/log_worker_${i}.txt" 2>&1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done

grep -E 'Final Report|Base :|Steer:' "${LOG_DIR}"/log_worker_*.txt > "${LOG_DIR}/final_report_grep.txt" 2>/dev/null || true

SUMMARY="${LOG_BASE}/sweep_collect_append.txt"
"$PYCMD" collect_sweep_results.py \
  --log_dir "$LOG_DIR" \
  --output "$SUMMARY" \
  --label "t${MAX_TOKEN}_ns${NUM_STEPS}_${SS_TAG}_r${REWARD_THRESHOLD}_c${CONFIDENCE_THRESHOLD}"

echo "Done LOG_DIR=$LOG_DIR"
