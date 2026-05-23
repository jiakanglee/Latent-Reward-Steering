#!/bin/bash
# Open-Reasoner-1.5B | GPQA-Diamond（198）| steering 超参 sweep
# step_size: 0.8~1.5（+0.1，共 8 档）× num_steps: 1~4（共 4 档）→ 32 组，array 0–31。
# 与 PST/run_sweep_step_aime25.slurm 对齐：sae_layer=8，n_clusters=5，模型 Open-Reasoner-Zero-1.5B。
# reward_threshold=0.9，confidence_threshold=0.72。
#
# 提交（仓库根）：  sbatch steering/run_rlab6_a5000_gpqa_or1.5b_steer_ss_ns_sweep.slurm
# 并发：默认最多同时 8 个 array 任务（各占 1 节点 ×4×A5000）；改大/改小请编辑下面 --array 的 % 后缀。

#SBATCH --job-name=GPQA15B_SSNS
#SBATCH --array=0-31%8
#SBATCH --output=log2/slurm_sweep_min/gpqa_or15b_ss_ns_sweep_%A_%a.log
#SBATCH --error=log2/slurm_sweep_min/gpqa_or15b_ss_ns_sweep_%A_%a.log
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
mkdir -p log2 log2/slurm_sweep_min
export PYTHONPATH=$PWD:$PYTHONPATH

# 与 PST/run_GPQA.slurm、PST/run_sweep_step_aime25.slurm 相同：AIME 上训的 1.5B LatentTransformer 权重
REWARD_MODEL="$PWD/PST/transformer_reward_model_aime_1.5B_best.pt"
MAX_TOKEN=4000
MODEL_NAME="Open-Reasoner-Zero/Open-Reasoner-Zero-1.5B"
SAE_LAYER=8
N_CLUSTERS=5
REWARD_THRESHOLD=0.9
CONFIDENCE_THRESHOLD=0.72
export ATTN_IMPLEMENTATION=flash_attention_2

PY="${CONDA_PREFIX}/bin/python"

MV_ERR="$("$PY" -c "from math_verify import parse, verify" 2>&1)" || {
  echo "FATAL: math_verify"
  echo "$MV_ERR"
  exit 1
}
unset MV_ERR

if [ ! -f "$REWARD_MODEL" ]; then
  echo "FATAL: reward 权重不存在: $REWARD_MODEL"
  exit 1
fi

TID=${SLURM_ARRAY_TASK_ID:-0}
# ss 共 8 档；ns 共 4 档；tid = (ns-1)*8 + ss_idx
STEPS=(0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5)
SS_IDX=$((TID % 8))
NS=$((TID / 8 + 1))
STEP_SIZE="${STEPS[$SS_IDX]}"

SS_TAG=$(echo "$STEP_SIZE" | tr '.' 'p')
_ARRAY_JOB="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
LOG_DIR="log2/gpqa_or15b_steer_sweep_${_ARRAY_JOB}/task_${TID}_ns${NS}_ss${SS_TAG}"
mkdir -p "${LOG_DIR}/judge_reasons"

echo "GPQA_OR1.5B_STEER_SWEEP | tid=$TID | ns=$NS | step_size=$STEP_SIZE | reward=$REWARD_THRESHOLD conf=$CONFIDENCE_THRESHOLD | LOG_DIR=$LOG_DIR"

NUM_GPUS=4
pids=()
for ((i=0; i<NUM_GPUS; i++)); do
  CUDA_VISIBLE_DEVICES=$i "$PY" -u steering/run_basic_overwrite.py \
    --model "$MODEL_NAME" \
    --dataset gpqa_diamond \
    --sae_layer "$SAE_LAYER" \
    --n_clusters "$N_CLUSTERS" \
    --reward_model_path "$REWARD_MODEL" \
    --step_size "$STEP_SIZE" \
    --num_steps "$NS" \
    --reward_threshold "$REWARD_THRESHOLD" \
    --confidence_threshold "$CONFIDENCE_THRESHOLD" \
    --num_examples 198 \
    --max_token "$MAX_TOKEN" \
    --no-monitor \
    --save_judge_reason \
    --judge_reason_dir "${LOG_DIR}/judge_reasons" \
    --shard_id "$i" \
    --num_shards "$NUM_GPUS" \
    > "${LOG_DIR}/log_worker_${i}.txt" 2>&1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done

grep -E 'Final Report|Base :|Steer:' "${LOG_DIR}"/log_worker_*.txt > "${LOG_DIR}/final_report_grep.txt" 2>/dev/null || true

RESULTS_FILE="${LOG_DIR}/sweep_row.txt"
echo "=== GPQA-Diamond OR-1.5B | ns=$NS ss=$STEP_SIZE | r=$REWARD_THRESHOLD c=$CONFIDENCE_THRESHOLD | t=$MAX_TOKEN ===" > "$RESULTS_FILE"
echo "t      | step   | conf   | nsteps | total | Base对 | Steer对 | Base对Steer错 | Base错Steer对" >> "$RESULTS_FILE"
"$PY" collect_sweep_results.py \
  --log_dir "$LOG_DIR" \
  --output "$RESULTS_FILE" \
  --label "ns${NS}_ss${SS_TAG}"

echo "Done LOG_DIR=$LOG_DIR"
