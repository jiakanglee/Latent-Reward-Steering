#!/bin/bash
# Llama-8B steering sweep: num_steps ∈ {1,2,3,4} × step_size ∈ {0.1..1.5} → 4×15=60 combos
# dataset: aime24 (30题)
# reward_threshold=0.9, confidence_threshold=0.7 (固定)
#
#SBATCH --job-name=Llama8B_Sweep_AIME24
#SBATCH --array=0-59%1
#SBATCH --output=/common/home/jl3614/Desktop/Python_project/thinking-llms-interp/Vicky/logs/sweep_llama8b_24_%A_%a.log
#SBATCH --error=/common/home/jl3614/Desktop/Python_project/thinking-llms-interp/Vicky/logs/sweep_llama8b_24_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=rlab1
#SBATCH --gres=gpu:4500_ada:2

export PYTHONNOUSERSITE=1
source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p Vicky/logs
export PYTHONPATH=$PWD:$PYTHONPATH

# ========================= 公共配置 =========================
DATASET=aime24
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
REWARD_MODEL="$PWD/Vicky/transformer_reward_model_llama8b_aime2425_best.pt"
SAE_LAYER=6
N_CLUSTERS=15
MAX_TOKEN=4000
REWARD_THRESHOLD=0.9
CONFIDENCE_THRESHOLD=0.7
export ATTN_IMPLEMENTATION=flash_attention_2
# ============================================================

# --- 解析 array task id → num_steps, step_size ---
TID=${SLURM_ARRAY_TASK_ID:-0}
read -r NUM_STEPS STEP_SIZE <<< "$(
  python3 -c "tid=int('${TID}'); print(tid//15+1, round(0.1*(tid%15+1), 1))"
)"
SS_STR=$(printf '%.1f' "$STEP_SIZE" | tr '.' 'p')
_ARRAY_JOB="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
LOG_BASE="Vicky/logs/llama8b_sweep_2425_${_ARRAY_JOB}"
LOG_DIR="${LOG_BASE}/task_${TID}_ns${NUM_STEPS}_ss${SS_STR}"
mkdir -p "$LOG_DIR" "${LOG_DIR}/judge"

echo "======================================================"
echo "TASK $TID | dataset=$DATASET | num_steps=$NUM_STEPS | step_size=$STEP_SIZE"
echo "LOG_DIR=$LOG_DIR"
echo "======================================================"

if [ ! -f "$REWARD_MODEL" ]; then
  echo "FATAL: 找不到 reward 权重: $REWARD_MODEL"
  exit 1
fi

# --- 写 run_meta.json ---
python3 -c "
import json
m = {
    'slurm_array_job_id': '${_ARRAY_JOB}',
    'slurm_array_task_id': ${TID},
    'dataset': '${DATASET}',
    'num_steps': ${NUM_STEPS},
    'step_size': ${STEP_SIZE},
    'reward_threshold': ${REWARD_THRESHOLD},
    'confidence_threshold': ${CONFIDENCE_THRESHOLD},
    'max_token': ${MAX_TOKEN},
    'reward_model': '${REWARD_MODEL}',
    'log_dir': '${LOG_DIR}',
}
with open('${LOG_DIR}/run_meta.json', 'w') as f:
    json.dump(m, f, indent=2)
print('Wrote ${LOG_DIR}/run_meta.json')
"

# --- 多卡并行 ---
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then NUM_GPUS=1; fi
echo "Using $NUM_GPUS GPUs"

pids=()
for ((i=0; i<NUM_GPUS; i++)); do
  CUDA_VISIBLE_DEVICES=$i python3 -u Vicky/run_basic_overwrite_update.py \
    --model "$MODEL_NAME" \
    --dataset "$DATASET" \
    --sae_layer $SAE_LAYER \
    --n_clusters $N_CLUSTERS \
    --reward_model_path "$REWARD_MODEL" \
    --step_size $STEP_SIZE \
    --num_steps $NUM_STEPS \
    --reward_threshold $REWARD_THRESHOLD \
    --confidence_threshold $CONFIDENCE_THRESHOLD \
    --max_token $MAX_TOKEN \
    --num_examples 30 \
    --shard_id $i \
    --num_shards $NUM_GPUS \
    --no-monitor \
    --save_judge_reason \
    --judge_reason_dir "${LOG_DIR}/judge" \
    > "${LOG_DIR}/log_worker_${i}.txt" 2>&1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done

# --- 汇总结果 ---
grep -E 'Final Report|Base :|Steer:' "${LOG_DIR}"/log_worker_*.txt \
  > "${LOG_DIR}/final_report_grep.txt" 2>/dev/null || true

echo "Done: $DATASET ns=$NUM_STEPS ss=$STEP_SIZE"
