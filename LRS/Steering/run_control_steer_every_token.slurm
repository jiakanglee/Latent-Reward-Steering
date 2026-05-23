#!/bin/bash
# 对照组：六个 benchmark 各跑一遍 —— **每个生成 token 都 steer**（--steer_every_token），
# 不使用 reward / confidence **门控**；num_steps、step_size 从 steering/control_steer_best_hparams.json 读取。
#
# Array：0=math500, 1=amc23, 2=aime24, 3=aime25, 4=gpqa_diamond, 5=ineqmath
# 覆盖超参文件：CONTROL_STEER_HPARAMS_JSON=/path/to.json sbatch ...
#
#SBATCH --job-name=CTRL_STEER_ALL
#SBATCH --array=0-5
#SBATCH --output=log2/control_steer_every_%A_%a.out
#SBATCH --error=log2/control_steer_every_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=11G
#SBATCH --time=72:00:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=rlab6
#SBATCH --gres=gpu:a5000:2

export PYTHONNOUSERSITE=1
source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p log2
export PYTHONPATH=$PWD:$PYTHONPATH

USE_PROJECT_VENV="${USE_PROJECT_VENV:-0}"
if [ "$USE_PROJECT_VENV" = "1" ] && [ -x "$PWD/.venv/bin/python3" ]; then
  PYCMD="$PWD/.venv/bin/python3"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
  PYCMD="${CONDA_PREFIX}/bin/python"
else
  PYCMD="python"
fi

REWARD_MODEL="${REWARD_MODEL:-$PWD/transformer_reward_model_aime_best.pt}"
MAX_TOKEN="${MAX_TOKEN:-4000}"
# 门控关闭时仍传入（hook 内不用于是否 steer）；保留与主实验一致的标度
REWARD="${REWARD:-0.9}"
CONF="${CONF:-0.72}"
HPATH="${CONTROL_STEER_HPARAMS_JSON:-$PWD/steering/control_steer_best_hparams.json}"

TID=${SLURM_ARRAY_TASK_ID:-0}
read -r DATASET NUM_STEPS STEP_SIZE NUM_EXAMPLES <<< "$(
  "$PYCMD" - "$TID" "$HPATH" <<'PY'
import json, sys
tid = int(sys.argv[1])
path = sys.argv[2]
order = ["math500", "amc23", "aime24", "aime25", "gpqa_diamond", "ineqmath"]
limits = {"math500": 500, "amc23": 40, "aime24": 30, "aime25": 30, "gpqa_diamond": 198, "ineqmath": 100}
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

_ARRAY_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
SS_TAG=$(echo "$STEP_SIZE" | tr '.' 'p')
LOG_DIR="log2/control_steer_${_ARRAY_ID}/${DATASET}_tid${TID}_ns${NUM_STEPS}_ss${SS_TAG}"
mkdir -p "$LOG_DIR/judge_reasons"

echo "CONTROL steer_every_token | dataset=$DATASET | ns=$NUM_STEPS ss=$STEP_SIZE | ne=$NUM_EXAMPLES | LOG_DIR=$LOG_DIR"

NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr ',' '\n' | sed '/^$/d' | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then NUM_GPUS=1; fi

INEQ_EXTRA=()
if [ "$DATASET" = "ineqmath" ]; then
  INEQ_EXTRA+=(--ineqmath_split dev --ineqmath_test_limit 100)
fi

pids=()
for ((i=0; i<NUM_GPUS; i++)); do
  if [ "$NUM_GPUS" -gt 1 ]; then
    export CUDA_VISIBLE_DEVICES=$i
  fi
  "$PYCMD" -u steering/run_basic_overwrite.py \
    --model "Open-Reasoner-Zero/Open-Reasoner-Zero-7B" \
    --sae_layer 20 \
    --n_clusters 10 \
    --reward_model_path "$REWARD_MODEL" \
    --step_size "$STEP_SIZE" \
    --num_steps "$NUM_STEPS" \
    --max_token "$MAX_TOKEN" \
    --reward_threshold "$REWARD" \
    --confidence_threshold "$CONF" \
    --steer_every_token \
    --dataset "$DATASET" \
    --num_examples "$NUM_EXAMPLES" \
    --no-monitor \
    --save_judge_reason \
    --judge_reason_dir "${LOG_DIR}/judge_reasons" \
    --shard_id "$i" \
    --num_shards "$NUM_GPUS" \
    "${INEQ_EXTRA[@]}" \
    > "${LOG_DIR}/log_worker_${i}.txt" 2>&1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done

grep -E 'Final Report|Base :|Steer:' "${LOG_DIR}"/log_worker_*.txt > "${LOG_DIR}/final_report_grep.txt" 2>/dev/null || true
"$PYCMD" collect_sweep_results.py \
  --log_dir "$LOG_DIR" \
  --output "${LOG_DIR}/../control_steer_summary_${_ARRAY_ID}.txt" \
  --label "t${MAX_TOKEN}_s${STEP_SIZE}_c${CONF}_r${REWARD}_n${NUM_STEPS}_${DATASET}_everytok"

echo "Done $LOG_DIR"
