#!/bin/bash
# DeepSeek-R1-Distill-Llama-8B | GPQA-Diamond（198 题）steering **全网格**：
# num_steps ∈ {1,2,3,4}，step_size ∈ {0.1..1.5} 步长 0.1 → 4×15=60 组。
#
# **倒序遍历**：Slurm task_id=0 对应 (num_steps=4, step_size=1.5)，task_id=59 对应 (1, 0.1)。
# 实现：grid_idx = 59 - SLURM_ARRAY_TASK_ID，再按与正序相同的公式从 grid_idx 解出 ns/ss。
# 与 RM path score 无关；仅 steering 评测 + judge JSON。
#
# 资源：rlab6，每 task **4×A5000**，mem=13G（198 题四卡分片）。
# 输出：log2_llama-8B/gpqa_diamond_steer_sweep_${SLURM_ARRAY_JOB_ID}/task_<id>_ns<N>_ss<S>/
#
# 提交（仓库根）：sbatch steering/run_gpqa_diamond_steer_sweep_ds8b_rlab6.slurm
#
#SBATCH --job-name=GPQA_DS8B_Swp
#SBATCH --array=0-59%1
#SBATCH --output=log2_llama-8B/slurm_sweep_min/gpqa_ds8b_steer_%A_%a.log
#SBATCH --error=log2_llama-8B/slurm_sweep_min/gpqa_ds8b_steer_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=13G
#SBATCH --time=72:00:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=rlab6
#SBATCH --gres=gpu:a5000:4

export PYTHONNOUSERSITE=1
source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p log2_llama-8B log2_llama-8B/slurm_sweep_min
export PYTHONPATH=$PWD:$PYTHONPATH

DATASET=gpqa_diamond
REWARD_MODEL="$PWD/Vicky/transformer_reward_model_llama8b_aime2425_best.pt"
MAX_TOKEN=4000
NUM_EXAMPLES=198
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
SAE_LAYER=6
N_CLUSTERS=15
REWARD_THRESHOLD=0.9
CONFIDENCE_THRESHOLD=0.72
SAVE_JUDGE_REASON=1
SAVE_JUDGE_WITH_TEXT=0
DEBUG_PRINT=0
USE_PROJECT_VENV=0

if [ "${USE_PROJECT_VENV}" = "1" ] && [ -x "$PWD/.venv/bin/python3" ]; then
  PYCMD="$PWD/.venv/bin/python3"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
  PYCMD="${CONDA_PREFIX}/bin/python"
else
  PYCMD="python"
fi

TID=${SLURM_ARRAY_TASK_ID:-0}
read -r NUM_STEPS STEP_SIZE <<< "$(
  "$PYCMD" -c "tid=int('${TID}'); g=59-tid; print(g//15+1, round(0.1*(g%15+1), 1))"
)"
SS_STR=$(printf '%.1f' "$STEP_SIZE" | tr '.' 'p')
_ARRAY_JOB="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
LOG_BASE="log2_llama-8B/gpqa_diamond_steer_sweep_${_ARRAY_JOB}"
LOG_DIR="${LOG_BASE}/task_${TID}_ns${NUM_STEPS}_ss${SS_STR}"
mkdir -p "$LOG_DIR"

if [ ! -f "$REWARD_MODEL" ]; then
  echo "FATAL: 找不到 reward 权重: $REWARD_MODEL"
  exit 1
fi

MV_ERR="$("$PYCMD" -c "from math_verify import parse, verify" 2>&1)" || {
  echo "FATAL: 无法 from math_verify import parse, verify"
  echo "$MV_ERR"
  exit 1
}
unset MV_ERR

export NUM_STEPS STEP_SIZE REWARD_THRESHOLD CONFIDENCE_THRESHOLD MAX_TOKEN NUM_EXAMPLES REWARD_MODEL LOG_DIR DATASET MODEL_NAME SAE_LAYER N_CLUSTERS
META_JSON="$LOG_DIR/run_meta.json"
export META_JSON
"$PYCMD" - <<'PY'
import json, os
tid = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
m = {
    "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", ""),
    "slurm_array_task_id": tid,
    "model": os.environ["MODEL_NAME"],
    "dataset": os.environ["DATASET"],
    "sae_layer": int(os.environ["SAE_LAYER"]),
    "n_clusters": int(os.environ["N_CLUSTERS"]),
    "num_steps": int(os.environ["NUM_STEPS"]),
    "step_size": float(os.environ["STEP_SIZE"]),
    "reward_threshold": float(os.environ["REWARD_THRESHOLD"]),
    "confidence_threshold": float(os.environ["CONFIDENCE_THRESHOLD"]),
    "max_token": int(os.environ["MAX_TOKEN"]),
    "num_examples": int(os.environ["NUM_EXAMPLES"]),
    "reward_model": os.environ["REWARD_MODEL"],
    "log_dir": os.environ["LOG_DIR"],
    "sweep_grid_order": "reverse",
    "sweep_grid_index": 59 - tid,
    "note": "GPQA-Diamond DeepSeek-R1-Distill-Llama-8B full grid (reverse) | rlab6 4xA5000 mem13G | SAE L6 K15",
}
with open(os.environ["META_JSON"], "w", encoding="utf-8") as f:
    json.dump(m, f, indent=2, ensure_ascii=False)
print("Wrote", os.environ["META_JSON"])
PY

NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr ',' '\n' | sed '/^$/d' | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then NUM_GPUS=4; fi

echo "ARRAY_TASK_ID=$TID  (reverse grid_idx=$((59-TID)))  NUM_STEPS=$NUM_STEPS  STEP_SIZE=$STEP_SIZE  LOG_DIR=$LOG_DIR  NUM_GPUS=$NUM_GPUS"

pids=()
for ((i=0; i<NUM_GPUS; i++)); do
  PY_ARGS=()
  if [ "$DEBUG_PRINT" != "1" ]; then
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
  CUDA_VISIBLE_DEVICES=$i "$PYCMD" -u Vicky/run_basic_overwrite_llama8b.py \
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
    --sae_no_activation_mean \
    "${PY_ARGS[@]}" \
    --shard_id "$i" \
    --num_shards "$NUM_GPUS" \
    > "${LOG_DIR}/log_worker_${i}.txt" 2>&1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done

grep -E 'Final Report|Base :|Steer:' "${LOG_DIR}"/log_worker_*.txt > "${LOG_DIR}/final_report_grep.txt" 2>/dev/null || true

SUMMARY="${LOG_BASE}/sweep_collect_append.txt"
"$PYCMD" collect_sweep_results.py \
  --log_dir "$LOG_DIR" \
  --output "$SUMMARY" \
  --label "ds8b_gpqa_t${MAX_TOKEN}_ns${NUM_STEPS}_ss${SS_STR}_r${REWARD_THRESHOLD}_c${CONFIDENCE_THRESHOLD}"

echo "Done LOG_DIR=$LOG_DIR"
