#!/bin/bash
# DeepSeek-R1-Distill-Llama-8B | AIME 2024（30 题）steering 网格 —— 对齐 OR-7B 的 log2_OR-7B/aime24_steer_sweep_* 结构：
# num_steps ∈ {1,2,3,4}，step_size ∈ {0.1..1.5} 步长 0.1 → 4×15=60 组。
#
# SAE：load_sae("deepseek-r1-distill-llama-8b", layer=6, clusters=15)，与 Vicky/generate_data_llama_8B.py 一致；
#      ckpt 路径为 train-saes/results/vars/saes/sae_deepseek-r1-distill-llama-8b_layer6_clusters15.pt（仓库内相对路径）。
# Reward model：Vicky/transformer_reward_model_llama8b_aime2425_best.pt
#
# 资源：ilab2，每 task **6×A4500**，题按 shard 分到 6 卡。
# 输出：log2_llama-8B/aime24_steer_sweep_${SLURM_ARRAY_JOB_ID}/task_<id>_ns<N>_ss<S>/{judge,log_worker_*.txt,run_meta.json}
#
#SBATCH --job-name=AIME24_DS8B_Swp
#SBATCH --array=0-59%1
#SBATCH --output=log2_llama-8B/slurm_sweep_min/aime24_ds8b_steer_%A_%a.log
#SBATCH --error=log2_llama-8B/slurm_sweep_min/aime24_ds8b_steer_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=13G
#SBATCH --time=72:00:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=ilab2
#SBATCH --gres=gpu:a4500:6

export PYTHONNOUSERSITE=1
source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p log2_llama-8B log2_llama-8B/slurm_sweep_min
export PYTHONPATH=$PWD:$PYTHONPATH

# ========================= 公共配置 =========================
DATASET=aime24
REWARD_MODEL="$PWD/Vicky/transformer_reward_model_llama8b_aime2425_best.pt"
MAX_TOKEN=4000
NUM_EXAMPLES=30
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
  "$PYCMD" -c "tid=int('${TID}'); print(tid//15+1, round(0.1*(tid%15+1), 1))"
)"
SS_STR=$(printf '%.1f' "$STEP_SIZE" | tr '.' 'p')
_ARRAY_JOB="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
LOG_BASE="log2_llama-8B/aime24_steer_sweep_${_ARRAY_JOB}"
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
m = {
    "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", ""),
    "slurm_array_task_id": int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
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
    "note": "AIME24 DeepSeek-R1-Distill-Llama-8B sweep | ilab2 6xA4500 | SAE deepseek-r1-distill-llama-8b L6 K15",
}
with open(os.environ["META_JSON"], "w", encoding="utf-8") as f:
    json.dump(m, f, indent=2, ensure_ascii=False)
print("Wrote", os.environ["META_JSON"])
PY

NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr ',' '\n' | sed '/^$/d' | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then NUM_GPUS=6; fi

echo "ARRAY_TASK_ID=$TID  NUM_STEPS=$NUM_STEPS  STEP_SIZE=$STEP_SIZE  LOG_DIR=$LOG_DIR  NUM_GPUS=$NUM_GPUS"

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
  --label "ds8b_t${MAX_TOKEN}_ns${NUM_STEPS}_ss${SS_STR}_r${REWARD_THRESHOLD}_c${CONFIDENCE_THRESHOLD}"

echo "Done LOG_DIR=$LOG_DIR"
