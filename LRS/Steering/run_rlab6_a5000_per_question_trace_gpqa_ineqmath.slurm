#!/bin/bash
# GPQA-Diamond（198）与 IneqMath dev（100）：与 rm_path_scores_139989 相同节点与分片
#（rlab6：GPQA 4×GPU、IneqMath 2×GPU；作业申请 4×A5000）；
# 每题 judge 全文 + 耗时/token/steer 次数 + 仅 steer 步 SAE 10 维 pre/post；不跑 --save_rm_path_scores。
#
# Attention：仅 GPQA 用 sdpa（对齐 gpqa_diamond_aime_mcq 历史）；IneqMath 仍 flash_attention_2。
# 可选：只重跑 GPQA 并写回已有 job 目录（不碰 ineqmath）：
#   TRACE_JOB_ID=144636 sbatch --array=0 steering/run_rlab6_a5000_per_question_trace_gpqa_ineqmath.slurm
#
# Array：0=gpqa_diamond（4×GPU），1=ineqmath（2×GPU）
# 提交：仓库根目录  sbatch steering/run_rlab6_a5000_per_question_trace_gpqa_ineqmath.slurm
#
#SBATCH --job-name=PQ_TRACE_GPQA_INEQ
#SBATCH --array=0-1
#SBATCH --output=log2/slurm_sweep_min/per_question_trace_gpqa_ineq_%A_%a.log
#SBATCH --error=log2/slurm_sweep_min/per_question_trace_gpqa_ineq_%A_%a.log
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
REPO_ROOT="$PWD"
mkdir -p log2 log2/slurm_sweep_min
export PYTHONPATH=$PWD:$PYTHONPATH

REWARD_MODEL="$PWD/transformer_reward_model_aime_best.pt"
MAX_TOKEN=4000
MODEL_NAME="Open-Reasoner-Zero/Open-Reasoner-Zero-7B"
SAE_LAYER=20
N_CLUSTERS=10
REWARD_THRESHOLD=0.9
CONFIDENCE_THRESHOLD=0.72

HPARAMS_JSON="${CONTROL_STEER_HPARAMS_JSON:-$PWD/steering/control_steer_best_hparams.json}"
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
read -r DATASET NUM_STEPS STEP_SIZE NUM_EXAMPLES NUM_GPUS <<< "$(
  "$PY" - "$TID" "$HPARAMS_JSON" <<'PY'
import json, sys
tid = int(sys.argv[1])
path = sys.argv[2]
order = ["gpqa_diamond", "ineqmath"]
limits = {"gpqa_diamond": 198, "ineqmath": 100}
gpus = {0: 4, 1: 2}
ds = order[tid]
with open(path, encoding="utf-8") as f:
    raw = json.load(f)
cfg = raw[ds]
print(ds, int(cfg["num_steps"]), float(cfg["step_size"]), int(limits[ds]), int(gpus[tid]))
PY
)"

if [ "$DATASET" = "gpqa_diamond" ]; then
  export ATTN_IMPLEMENTATION=sdpa
else
  export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
fi

if [ -n "${TRACE_JOB_ID:-}" ]; then
  _ARRAY_JOB="$TRACE_JOB_ID"
else
  _ARRAY_JOB="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
fi
SS_TAG=$(echo "$STEP_SIZE" | tr '.' 'p')
LOG_DIR="${REPO_ROOT}/log2/per_question_trace_${_ARRAY_JOB}/${DATASET}_ns${NUM_STEPS}_ss${SS_TAG}"
mkdir -p "${LOG_DIR}/judge_reasons" "${LOG_DIR}/sae_steer_trace"

echo "PER_QUESTION_TRACE | attn=$ATTN_IMPLEMENTATION | dataset=$DATASET | ns=$NUM_STEPS ss=$STEP_SIZE | ne=$NUM_EXAMPLES | NUM_GPUS=$NUM_GPUS | LOG_DIR=$LOG_DIR"

INEQ_EXTRA=()
if [ "$DATASET" = "ineqmath" ]; then
  INEQ_EXTRA+=(--ineqmath_split dev --ineqmath_test_limit 100)
fi

pids=()
for ((i=0; i<NUM_GPUS; i++)); do
  CUDA_VISIBLE_DEVICES=$i "$PY" -u steering/run_basic_overwrite.py \
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
    --save_judge_with_text \
    --judge_reason_dir "${LOG_DIR}/judge_reasons" \
    --save_sae_steer_trace \
    --sae_steer_trace_dir "${LOG_DIR}/sae_steer_trace" \
    --shard_id "$i" \
    --num_shards "$NUM_GPUS" \
    "${INEQ_EXTRA[@]}" \
    > "${LOG_DIR}/log_worker_${i}.txt" 2>&1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done

grep -E 'Final Report|Base :|Steer:' "${LOG_DIR}"/log_worker_*.txt > "${LOG_DIR}/final_report_grep.txt" 2>/dev/null || true
echo "Done LOG_DIR=$LOG_DIR"
