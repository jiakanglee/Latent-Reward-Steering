#!/bin/bash
# IneqMath dev（100 题）steering 网格：num_steps ∈ {1,2,3,4,5}，step_size ∈ {0.1,0.2,…,1.5} → 5×15=75 组。
# 判题逻辑与 steering/run_basic_overwrite.py 一致：idx<50 math_verify，idx≥50 GPQA 字母。
#
# 输出：log2/ineqmath_sweep_${SLURM_ARRAY_JOB_ID}/task_<id>_ns<N>_ss<ss>/log_worker_*.txt 与 judge_reasons/
# 汇总表（每 task 追加一行）：log2/ineqmath_sweep_table_${SLURM_ARRAY_JOB_ID}.txt
#
# 提交：仓库根目录  sbatch run_ineqmath_steer_sweep.slurm
# 仅扫 Steer（需已有 Base 的 judge）：STEER_SWEEP_FULL=0 PRIOR_JUDGE=/path/to/judge_reasons sbatch ...
#
#SBATCH --job-name=INEQM_SWEEP
#SBATCH --array=0-74%2
#SBATCH --output=log2/slurm_sweep_min/ineqmath_sweep_%A_%a.log
#SBATCH --error=log2/slurm_sweep_min/ineqmath_sweep_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=13G
#SBATCH --time=72:00:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=rlab6
#SBATCH --gres=gpu:a5000:2

# ========================= 配置（只改这里） =========================
REWARD_MODEL="${REWARD_MODEL:-transformer_reward_model_aime_best.pt}"
REWARD_THRESHOLD=0.9
CONFIDENCE_THRESHOLD=0.72
MAX_TOKEN=4000
MODEL_NAME="Open-Reasoner-Zero/Open-Reasoner-Zero-7B"
SAE_LAYER=20
N_CLUSTERS=10
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
# 1=每组跑 Base+Steer（慢）；0=--steer_only，Base 从 PRIOR_JUDGE 回填
STEER_SWEEP_FULL=1
PRIOR_JUDGE=""
SAVE_JUDGE_WITH_TEXT=0
# 0=conda（stllms_env 通常已装 flash_attn）；1=强制 .venv（需自行安装 flash-attn）
USE_PROJECT_VENV=0
# =====================================================================

export PYTHONNOUSERSITE=1
source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p log2 log2/slurm_sweep_min
export PYTHONPATH=$PWD:$PYTHONPATH

if [ "${USE_PROJECT_VENV}" = "1" ] && [ -x "$PWD/.venv/bin/python3" ]; then
  PYCMD="$PWD/.venv/bin/python3"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
  PYCMD="${CONDA_PREFIX}/bin/python"
else
  PYCMD="python"
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

if [ "$STEER_SWEEP_FULL" != "1" ]; then
  if [ -z "$PRIOR_JUDGE" ] || [ ! -d "$PRIOR_JUDGE" ]; then
    echo "FATAL: STEER_SWEEP_FULL=0 时需要有效目录 PRIOR_JUDGE（含 question_*_shard_*.json）"
    exit 1
  fi
fi

TID=${SLURM_ARRAY_TASK_ID:-0}
read -r NUM_STEPS STEP_SIZE <<< "$("$PYCMD" -c "tid=int('${TID}'); print(tid//15+1, round(0.1*(tid%15+1), 1))")"

SS_STR=$(printf '%.1f' "$STEP_SIZE" | tr '.' 'p')
_ARRAY_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
LOG_BASE="log2/ineqmath_sweep_${_ARRAY_ID}"
LOG_DIR="${LOG_BASE}/task_${TID}_ns${NUM_STEPS}_ss${SS_STR}"
mkdir -p "$LOG_DIR/judge_reasons"

SUMMARY_FILE="log2/ineqmath_sweep_table_${_ARRAY_ID}.txt"
if [ ! -f "$SUMMARY_FILE" ]; then
  umask 002
  echo "t      | step   | conf   | reward | nsteps | total | Base对 | Steer对 | Base对Steer错 | Base错Steer对" > "$SUMMARY_FILE"
fi

export ATTN_IMPLEMENTATION="$ATTN_IMPLEMENTATION"

echo "ARRAY_TASK_ID=$TID  NUM_STEPS=$NUM_STEPS  STEP_SIZE=$STEP_SIZE"
echo "LOG_DIR=$LOG_DIR  STEER_SWEEP_FULL=$STEER_SWEEP_FULL"

NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr ',' '\n' | sed '/^$/d' | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then NUM_GPUS=1; fi

META_JSON="$LOG_DIR/sweep_task_meta.json"
export META_JSON NUM_STEPS STEP_SIZE REWARD_THRESHOLD CONFIDENCE_THRESHOLD MAX_TOKEN LOG_DIR STEER_SWEEP_FULL PRIOR_JUDGE
"$PYCMD" -c 'import json, os
m = {
  "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", ""),
  "slurm_array_task_id": int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
  "num_steps": int(os.environ["NUM_STEPS"]),
  "step_size": float(os.environ["STEP_SIZE"]),
  "reward_threshold": float(os.environ["REWARD_THRESHOLD"]),
  "confidence_threshold": float(os.environ["CONFIDENCE_THRESHOLD"]),
  "max_token": int(os.environ["MAX_TOKEN"]),
  "steer_only": os.environ.get("STEER_SWEEP_FULL", "1") != "1",
  "prior_judge_dir": os.environ.get("PRIOR_JUDGE", ""),
  "log_dir": os.environ["LOG_DIR"],
}
path = os.environ["META_JSON"]
open(path, "w", encoding="utf-8").write(json.dumps(m, indent=2, ensure_ascii=False))
print("Wrote", path)
'

pids=()
for ((i=0; i<NUM_GPUS; i++)); do
  if [ "$NUM_GPUS" -gt 1 ]; then
    export CUDA_VISIBLE_DEVICES=$i
  fi
  PY_ARGS=(
    --model "$MODEL_NAME"
    --sae_layer "$SAE_LAYER"
    --n_clusters "$N_CLUSTERS"
    --reward_model_path "$REWARD_MODEL"
    --step_size "$STEP_SIZE"
    --num_steps "$NUM_STEPS"
    --max_token "$MAX_TOKEN"
    --reward_threshold "$REWARD_THRESHOLD"
    --confidence_threshold "$CONFIDENCE_THRESHOLD"
    --dataset ineqmath
    --ineqmath_split dev
    --ineqmath_test_limit 100
    --num_examples 100
    --no-monitor
    --save_judge_reason
    --judge_reason_dir "${LOG_DIR}/judge_reasons"
    --shard_id "$i"
    --num_shards "$NUM_GPUS"
  )
  if [ "$SAVE_JUDGE_WITH_TEXT" = "1" ]; then
    PY_ARGS+=(--save_judge_with_text)
  fi
  if [ "$STEER_SWEEP_FULL" != "1" ]; then
    PY_ARGS+=(--steer_only --prior_judge_dir "$PRIOR_JUDGE")
  fi
  $PYCMD -u steering/run_basic_overwrite.py "${PY_ARGS[@]}" \
    > "${LOG_DIR}/log_worker_${i}.txt" 2>&1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done

grep -E 'Final Report|Base :|Steer:' "${LOG_DIR}"/log_worker_*.txt > "${LOG_DIR}/final_report_grep.txt" 2>/dev/null || true

LABEL="t${MAX_TOKEN}_s${STEP_SIZE}_c${CONFIDENCE_THRESHOLD}_r${REWARD_THRESHOLD}_n${NUM_STEPS}"
"$PYCMD" collect_sweep_results.py \
  --log_dir "$LOG_DIR" \
  --output "$SUMMARY_FILE" \
  --label "$LABEL"

echo "Done LOG_DIR=$LOG_DIR"
