#!/bin/bash
# MATH-500：仅 Steer（不跑 Base 生成），题目集 = 全量 500 减去「某次 judge 里 base 与 steer 都对」的题目。
# 筛题用 judge 路径写在下面「=== 配置 ===」处，直接 sbatch 即可，无需在终端 export。
# Flash Attention：默认 ATTN_IMPLEMENTATION=flash_attention_2（与 run_math500_steer_overwrite.slurm 一致）。
# steer_only 时 judge JSON 默认仅 steer 字段 + response_steer；若要旧 base 元数据/全文：
#   export SAVE_JUDGE_PRIOR_BASE_META=1    → base_correct / judge_base_reason / len_base
#   export SAVE_JUDGE_WITH_PRIOR_BASE_TEXT=1 → 另含 response_base 拷贝
# Python：默认用 conda（避免 .venv 无 flash-attn）；若要坚持 .venv：export USE_PROJECT_VENV=1
#
#SBATCH --job-name=MATH500_SteerSubset
#SBATCH --output=log2/math500_steer_overwrite_subset_%j.out
#SBATCH --error=log2/math500_steer_overwrite_subset_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=rlab6
#SBATCH --gres=gpu:a5000:3

source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p log2
export PYTHONPATH=$PWD:$PYTHONPATH

# === 配置：筛题 + prior judge（只需改这一处；目录内须有 question_*_shard_*.json）===
# 重要：本作业依赖「某次全量 Base+Steer」产出的 judge/，勿整目录删除；清 log2 前请备份或改下面路径。
# 若已误删，只能恢复备份或重跑 run_math500_steer_overwrite.slurm 重新生成 judge/。
PREFERRED_JUDGE="$PWD/log2/math500_steer_overwrite_124343/judge"
# ==============================================================================

resolve_filter_judge_dir() {
  FILTER_JUDGE_DIR=""
  local preferred="$1"
  local n d t best_t cand
  if [ -d "$preferred" ]; then
    n=$(find "$preferred" -maxdepth 1 -name 'question_*_shard_*.json' 2>/dev/null | wc -l)
    if [ "$n" -ge 1 ]; then
      FILTER_JUDGE_DIR="$preferred"
      return 0
    fi
  fi
  best_t=0
  cand=""
  while IFS= read -r d; do
    [ -z "$d" ] && continue
    n=$(find "$d" -maxdepth 1 -name 'question_*_shard_*.json' 2>/dev/null | wc -l)
    [ "$n" -ge 1 ] || continue
    t=$(stat -c %Y "$d" 2>/dev/null || echo 0)
    if [ "$t" -ge "$best_t" ]; then
      best_t=$t
      cand="$d"
    fi
  done < <(find "$PWD/log2" -mindepth 1 -maxdepth 2 -type d -name judge 2>/dev/null || true)
  FILTER_JUDGE_DIR="$cand"
}
resolve_filter_judge_dir "$PREFERRED_JUDGE"
# 若首选目录不存在或无 JSON，已自动改用 log2 下最新且含 JSON 的 */judge
if [ -n "$FILTER_JUDGE_DIR" ] && [ "$FILTER_JUDGE_DIR" != "$PREFERRED_JUDGE" ]; then
  echo "注意: PREFERRED_JUDGE 不可用，实际使用 FILTER_JUDGE_DIR=$FILTER_JUDGE_DIR"
fi

unset PYTHONNOUSERSITE
# 默认用 conda activate 后的解释器（flash-attn 多装在此）；只有 export USE_PROJECT_VENV=1 才强制 .venv
if [ "${USE_PROJECT_VENV:-0}" = "1" ] && [ -x "$PWD/.venv/bin/python3" ]; then
  PYCMD="$PWD/.venv/bin/python3"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
  PYCMD="${CONDA_PREFIX}/bin/python"
elif [ -x "$PWD/.venv/bin/python3" ]; then
  PYCMD="$PWD/.venv/bin/python3"
else
  PYCMD="python"
fi
if [ "${MATH500_ALLOW_USER_SITE:-0}" != "1" ]; then
  export PYTHONNOUSERSITE=1
fi

MV_ERR="$("$PYCMD" -c "from math_verify import parse, verify" 2>&1)" || {
  echo "FATAL: 无法 from math_verify import parse, verify"
  echo "当前解释器: $($PYCMD -c 'import sys; print(sys.executable)')"
  echo "$MV_ERR"
  exit 1
}
unset MV_ERR

if [ -z "$FILTER_JUDGE_DIR" ] || [ ! -d "$FILTER_JUDGE_DIR" ]; then
  echo "FATAL: 找不到可用 judge 目录（已试首选 PREFERRED_JUDGE=${PREFERRED_JUDGE:-}，并在 log2/*/judge 中查找）。"
  echo "  请恢复全量跑次的 judge/，或编辑本脚本顶部 PREFERRED_JUDGE= 指向含 question_*_shard_*.json 的目录。"
  exit 1
fi
n_json=$(find "$FILTER_JUDGE_DIR" -maxdepth 1 -name 'question_*_shard_*.json' 2>/dev/null | wc -l)
if [ "$n_json" -eq 0 ]; then
  echo "FATAL: 目录内没有 question_*_shard_*.json: $FILTER_JUDGE_DIR"
  exit 1
fi

PRIOR_JUDGE_DIR="$FILTER_JUDGE_DIR"

MODEL_NAME="${MODEL_NAME:-Open-Reasoner-Zero/Open-Reasoner-Zero-7B}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
MAX_TOKEN="${MAX_TOKEN:-2500}"
SAVE_JUDGE_REASON="${SAVE_JUDGE_REASON:-1}"
SAVE_JUDGE_WITH_TEXT="${SAVE_JUDGE_WITH_TEXT:-1}"
SAVE_JUDGE_WITH_PRIOR_BASE_TEXT="${SAVE_JUDGE_WITH_PRIOR_BASE_TEXT:-0}"
SAVE_JUDGE_PRIOR_BASE_META="${SAVE_JUDGE_PRIOR_BASE_META:-0}"
STEER_TOKEN_DEBUG="${STEER_TOKEN_DEBUG:-0}"
DEBUG_PRINT="${DEBUG_PRINT:-0}"
STEER_ANALYSIS="${STEER_ANALYSIS:-0}"

REWARD_MODEL="${REWARD_MODEL:-transformer_reward_model_best.pt}"
SAE_LAYER="${SAE_LAYER:-20}"
N_CLUSTERS="${N_CLUSTERS:-10}"
STEP_SIZE="${STEP_SIZE:-1.1125}"
NUM_STEPS="${NUM_STEPS:-4}"
REWARD_THRESHOLD="${REWARD_THRESHOLD:-0.8}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.72}"

LOG_DIR="log2/math500_steer_overwrite_subset_${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then NUM_GPUS=1; fi

echo "FILTER_JUDGE_DIR=${FILTER_JUDGE_DIR}"
echo "PRIOR_JUDGE_DIR=${PRIOR_JUDGE_DIR}"
FILTER_JUDGE_DIR="$FILTER_JUDGE_DIR" "$PYCMD" -u <<'PY'
import json, os
from glob import glob
d = os.environ["FILTER_JUDGE_DIR"]
ver = {}
for p in glob(os.path.join(d, "question_*_shard_*.json")):
    with open(p, encoding="utf-8") as f:
        j = json.load(f)
    i = j.get("question_idx")
    if i is None:
        continue
    ver[i] = (bool(j.get("base_correct")), bool(j.get("steer_correct")))
both_ok = sum(1 for b, s in ver.values() if b and s)
# 无 verdict 的题号仍会跑；此处仅统计「有记录且双对」的排除量
print(f"prior_verdicts={len(ver)} both_ok_skipped={both_ok} (其余题含无记录均会进入 run 列表)")
PY

pids=()
for ((i=0; i<NUM_GPUS; i++)); do
  PY_ARGS=(--filter_from_judge_dir "$FILTER_JUDGE_DIR" --steer_only --prior_judge_dir "$PRIOR_JUDGE_DIR")
  if [ "$STEER_TOKEN_DEBUG" != "1" ]; then
    PY_ARGS+=(--no-monitor)
  fi
  if [ "$STEER_ANALYSIS" = "1" ] || [ "$DEBUG_PRINT" = "1" ]; then
    PY_ARGS+=(--print_response)
  fi
  if [ "$SAVE_JUDGE_REASON" = "1" ]; then
    mkdir -p "${LOG_DIR}/judge"
    PY_ARGS+=(--save_judge_reason --judge_reason_dir "${LOG_DIR}/judge")
    if [ "$SAVE_JUDGE_WITH_TEXT" = "1" ]; then
      PY_ARGS+=(--save_judge_with_text)
    fi
    if [ "$SAVE_JUDGE_WITH_PRIOR_BASE_TEXT" = "1" ]; then
      PY_ARGS+=(--save_judge_with_prior_base_text)
    fi
    if [ "$SAVE_JUDGE_PRIOR_BASE_META" = "1" ]; then
      PY_ARGS+=(--save_judge_prior_base_meta)
    fi
  fi
  CUDA_VISIBLE_DEVICES=$i $PYCMD -u steering/run_basic_overwrite.py \
    --model "$MODEL_NAME" \
    --dataset math500 \
    --sae_layer "$SAE_LAYER" \
    --n_clusters "$N_CLUSTERS" \
    --reward_model_path "$REWARD_MODEL" \
    --step_size "$STEP_SIZE" \
    --num_steps "$NUM_STEPS" \
    --reward_threshold "$REWARD_THRESHOLD" \
    --confidence_threshold "$CONFIDENCE_THRESHOLD" \
    --max_token "$MAX_TOKEN" \
    --shard_id $i \
    --num_shards $NUM_GPUS \
    "${PY_ARGS[@]}" \
    > "${LOG_DIR}/log_worker_${i}.txt" 2>&1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait $pid; done

echo "LOG_DIR=${LOG_DIR}"
grep -E 'Mode filter_from_judge|Final Report|Base :|Steer:' "${LOG_DIR}"/log_worker_*.txt || true
