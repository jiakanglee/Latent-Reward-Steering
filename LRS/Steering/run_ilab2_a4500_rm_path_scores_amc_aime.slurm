#!/bin/bash
# AMC23 / AIME24 / AIME25：Steer 最优超参（control_steer_best_hparams.json）+ 每题保存 RM 全长轨迹打分 p_last / p_mean。
# 节点/GPU 与 AMC sweep 对齐：ilab2，每作业 5×A4500（与原文 sweep 同机型）。
#
# Array：0=amc23, 1=aime24, 2=aime25
# 提交：仓库根目录  sbatch steering/run_ilab2_a4500_rm_path_scores_amc_aime.slurm
#
#SBATCH --job-name=RM_PATH_AMC_AIME
#SBATCH --array=0-2
#SBATCH --output=log2/slurm_sweep_min/rm_path_scores_amc_aime_%A_%a.log
#SBATCH --error=log2/slurm_sweep_min/rm_path_scores_amc_aime_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=13G
#SBATCH --time=72:00:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=ilab2
#SBATCH --gres=gpu:a4500:6

source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p log2 log2/slurm_sweep_min
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONNOUSERSITE=1

REWARD_MODEL="$PWD/transformer_reward_model_aime_best.pt"
MAX_TOKEN=4000
MODEL_NAME="Open-Reasoner-Zero/Open-Reasoner-Zero-7B"
SAE_LAYER=20
N_CLUSTERS=10
REWARD_THRESHOLD=0.9
CONFIDENCE_THRESHOLD=0.72
export ATTN_IMPLEMENTATION=flash_attention_2

HPARAMS_JSON="${CONTROL_STEER_HPARAMS_JSON:-$PWD/steering/control_steer_best_hparams.json}"

TID=${SLURM_ARRAY_TASK_ID:-0}
read -r DATASET NUM_STEPS STEP_SIZE NUM_EXAMPLES <<< "$(
  "$CONDA_PREFIX/bin/python" - "$TID" "$HPARAMS_JSON" <<'PY'
import json, sys
tid = int(sys.argv[1])
path = sys.argv[2]
order = ["amc23", "aime24", "aime25"]
limits = {"amc23": 40, "aime24": 30, "aime25": 30}
ds = order[tid]
with open(path, encoding="utf-8") as f:
    raw = json.load(f)
cfg = raw[ds]
print(ds, int(cfg["num_steps"]), float(cfg["step_size"]), int(limits[ds]))
PY
)"

if [ ! -f "$REWARD_MODEL" ]; then
  echo "FATAL: reward 权重不存在: $REWARD_MODEL"
  exit 1
fi

_ARRAY_JOB="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
SS_TAG=$(echo "$STEP_SIZE" | tr '.' 'p')
LOG_DIR="log2/rm_path_scores_${_ARRAY_JOB}/${DATASET}_ns${NUM_STEPS}_ss${SS_TAG}"
mkdir -p "${LOG_DIR}/judge_reasons" "${LOG_DIR}/rm_path_scores"

echo "RM_PATH_SCORES | dataset=$DATASET | ns=$NUM_STEPS ss=$STEP_SIZE | ne=$NUM_EXAMPLES | LOG_DIR=$LOG_DIR"

NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr ',' '\n' | sed '/^$/d' | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then NUM_GPUS=5; fi

pids=()
for ((i=0; i<NUM_GPUS; i++)); do
  CUDA_VISIBLE_DEVICES=$i "$CONDA_PREFIX/bin/python" -u steering/run_basic_overwrite.py \
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
    --no-monitor \
    --save_judge_reason \
    --judge_reason_dir "${LOG_DIR}/judge_reasons" \
    --save_rm_path_scores \
    --rm_path_scores_dir "${LOG_DIR}/rm_path_scores" \
    --shard_id "$i" \
    --num_shards "$NUM_GPUS" \
    > "${LOG_DIR}/log_worker_${i}.txt" 2>&1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done

grep -E 'Final Report|Base :|Steer:' "${LOG_DIR}"/log_worker_*.txt > "${LOG_DIR}/final_report_grep.txt" 2>/dev/null || true
echo "Done LOG_DIR=$LOG_DIR"
