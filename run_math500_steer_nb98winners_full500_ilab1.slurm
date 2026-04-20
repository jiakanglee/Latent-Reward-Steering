#!/bin/bash
# MATH-500 全量 500 题：nb98 sweep（125592）上 net 较高的四组超参，在 ilab1 + 2×A4500 上各跑一遍。
#   task 0: ns=3 ss=0.3  （nb98 最优档）
#   task 1: ns=4 ss=0.2
#   task 2: ns=1 ss=0.7
#   task 3: ns=1 ss=1.4
# 与 nb98 / top3 对齐：reward=0.8, conf=0.69, max_token=4000；每 array task 内 2 GPU 按题号分片。
# judge/*.json 含 response_base + response_steer 全文（SAVE_JUDGE_WITH_TEXT=1）。
# 输出：log2/math500_steer_nb98winners_full500_${ARRAY_JOB_ID}/task_*_ns*_ss*/
#
#SBATCH --job-name=M500_NB98Win4
#SBATCH --array=0-3
#SBATCH --output=log2/slurm_sweep_min/math500_nb98winners_full500_%A_%a.log
#SBATCH --error=log2/slurm_sweep_min/math500_nb98winners_full500_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=ilab2
#SBATCH --gres=gpu:a4500:2

source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p log2 log2/slurm_sweep_min
export PYTHONPATH=$PWD:$PYTHONPATH

# ========================= 公共配置 =========================
REWARD_MODEL="$PWD/transformer_reward_model_aime_best.pt"
MAX_TOKEN=4000
NUM_EXAMPLES=500
MODEL_NAME="Open-Reasoner-Zero/Open-Reasoner-Zero-7B"
export ATTN_IMPLEMENTATION=flash_attention_2
SAE_LAYER=20
N_CLUSTERS=10
REWARD_THRESHOLD=0.8
CONFIDENCE_THRESHOLD=0.69
DEBUG_PRINT=0
SAVE_JUDGE_REASON=1
SAVE_JUDGE_WITH_TEXT=1
STEER_ANALYSIS=0
STEER_TOKEN_DEBUG=0
USE_PROJECT_VENV=0
MATH500_ALLOW_USER_SITE=0
# =====================================================================

TID=${SLURM_ARRAY_TASK_ID:-0}
case "$TID" in
  0) NUM_STEPS=3; STEP_SIZE=0.3 ;;
  1) NUM_STEPS=4; STEP_SIZE=0.2 ;;
  2) NUM_STEPS=1; STEP_SIZE=0.7 ;;
  3) NUM_STEPS=1; STEP_SIZE=1.4 ;;
  *) echo "FATAL: unexpected SLURM_ARRAY_TASK_ID=$TID"; exit 1 ;;
esac
SS_STR=$(printf '%.1f' "$STEP_SIZE" | tr '.' 'p')
_ARRAY_JOB="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"

if [ ! -f "$REWARD_MODEL" ]; then
  echo "FATAL: 找不到 reward 权重: $REWARD_MODEL"
  exit 1
fi

if [ "$STEER_ANALYSIS" = "1" ]; then
  SAVE_JUDGE_REASON=1
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

LOG_DIR="log2/math500_steer_nb98winners_full500_${_ARRAY_JOB}/task_${TID}_ns${NUM_STEPS}_ss${SS_STR}"
mkdir -p "$LOG_DIR"

export NUM_STEPS STEP_SIZE REWARD_THRESHOLD CONFIDENCE_THRESHOLD MAX_TOKEN NUM_EXAMPLES REWARD_MODEL LOG_DIR
META_JSON="$LOG_DIR/run_meta.json"
export META_PATH="$META_JSON"
export ATTN_IMPLEMENTATION="$ATTN_IMPLEMENTATION"
"$PYCMD" - <<'PY'
import json, os
m = {
    "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", ""),
    "slurm_array_task_id": int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
    "num_steps": int(os.environ["NUM_STEPS"]),
    "step_size": float(os.environ["STEP_SIZE"]),
    "reward_threshold": float(os.environ["REWARD_THRESHOLD"]),
    "confidence_threshold": float(os.environ["CONFIDENCE_THRESHOLD"]),
    "max_token": int(os.environ["MAX_TOKEN"]),
    "num_examples": int(os.environ["NUM_EXAMPLES"]),
    "reward_model": os.environ["REWARD_MODEL"],
    "log_dir": os.environ["LOG_DIR"],
    "note": "nb98 net leaders on full MATH-500; ilab1 2xA4500",
}
path = os.environ["META_PATH"]
with open(path, "w", encoding="utf-8") as f:
    json.dump(m, f, indent=2, ensure_ascii=False)
print("Wrote", path)
PY

NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then NUM_GPUS=1; fi

export NUM_EXAMPLES
export NUM_GPUS
export LOG_DIR
RUN_MANIFEST="${LOG_DIR}/math500_run_manifest.json"
export RUN_MANIFEST
"$PYCMD" -u <<'PY'
import json, os
n = int(os.environ["NUM_EXAMPLES"])
num_shards = int(os.environ["NUM_GPUS"])
job = os.environ.get("SLURM_JOB_ID", "")
log_rel = os.environ["LOG_DIR"]
log_abs = os.path.abspath(log_rel)
indices = list(range(n))
by_shard = {str(s): [i for i in indices if i % num_shards == s] for s in range(num_shards)}
manifest = {
    "dataset": "HuggingFaceH4/MATH-500",
    "num_questions": n,
    "question_indices": indices,
    "modes": ["baseline_generate", "steer_generate"],
    "judge_dir_relative": "judge",
    "slurm_job_id": job,
    "log_dir": log_rel,
    "log_dir_abs": log_abs,
    "num_shards": num_shards,
    "shard_rule": "question_idx % num_shards == shard_id (same as run_basic_overwrite.py)",
    "questions_by_shard": {k: {"count": len(v), "indices": v} for k, v in by_shard.items()},
}
path = os.environ["RUN_MANIFEST"]
with open(path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
print("Wrote", path)
PY

echo "ARRAY_TASK_ID=$TID  NUM_STEPS=$NUM_STEPS  STEP_SIZE=$STEP_SIZE  LOG_DIR=$LOG_DIR  NUM_GPUS=$NUM_GPUS"

pids=()
for ((i=0; i<NUM_GPUS; i++)); do
  PY_ARGS=()
  if [ "$STEER_TOKEN_DEBUG" != "1" ]; then
    PY_ARGS+=(--no-monitor)
  fi
  if [ "$STEER_ANALYSIS" = "1" ]; then
    PY_ARGS+=(--print_response)
  elif [ "$DEBUG_PRINT" = "1" ]; then
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
    --dataset math500 \
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
for pid in "${pids[@]}"; do wait $pid; done

echo "Done LOG_DIR=${LOG_DIR}"
grep -E 'Final Report|Base :|Steer:' "${LOG_DIR}"/log_worker_*.txt || true
