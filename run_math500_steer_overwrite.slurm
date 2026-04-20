#!/bin/bash
# MATH-500：迭代 steering（非 baseline_only），与 run_basic_overwrite.py 一致；判题 math_verify。
# Python：默认 conda（CONDA_PREFIX）；flash-attn 若在 conda 而项目有 .venv，旧逻辑会误用无 FA 的 .venv — 现已默认 conda，需 .venv 时 export USE_PROJECT_VENV=1。
# 与 run_math500_baseline_overwrite.slurm 共用环境/PYCMD 策略，仅多 steering 超参与不传 --baseline_only。
#
# 主配置在下方「=== 配置 ===」（reward / max_token 等）；默认全量 500 题。
# 本脚本默认传 --no-monitor（不刷 [Tok] Score / Latent）；要看逐 token steering 调试：
#   export STEER_TOKEN_DEBUG=1
# 默认已 SAVE_JUDGE_REASON=1、SAVE_JUDGE_WITH_TEXT=1：每题 judge/*.json 含判别 + response_base/steer 全文。
# 关 judge：export SAVE_JUDGE_REASON=0；只要小 JSON、不要全文：export SAVE_JUDGE_WITH_TEXT=0
# 若仍要把全文打到 stdout：export STEER_ANALYSIS=1 或 DEBUG_PRINT=1（log 会很大）。
# 跑完后: python analyze_math500_steer_judge_json.py "${LOG_DIR}/judge" [--list-empty] [--list-false]
# 汇总：各分片 Final Report 中 "Base :" / "Steer:" 的分子相加，与 NUM_EXAMPLES 对照。
#   grep -E 'Final Report|Base :|Steer:' "${LOG_DIR}"/log_worker_*.txt
#
#SBATCH --job-name=MATH500_SteerOverwrite
#SBATCH --output=log2/math500_steer_overwrite_%j.out
#SBATCH --error=log2/math500_steer_overwrite_%j.err
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

# ========================= 配置（只改这里） =========================
# MATH-500，Steering 用 AIME 上训练的 reward；更长生成减轻截断
REWARD_MODEL="$PWD/transformer_reward_model_aime_best.pt"
MAX_TOKEN=4000
NUM_EXAMPLES=500
MODEL_NAME="Open-Reasoner-Zero/Open-Reasoner-Zero-7B"
export ATTN_IMPLEMENTATION=flash_attention_2
SAE_LAYER=20
N_CLUSTERS=10
STEP_SIZE=1.1125
NUM_STEPS=4
REWARD_THRESHOLD=0.8
CONFIDENCE_THRESHOLD=0.72
DEBUG_PRINT=0
SAVE_JUDGE_REASON=1
SAVE_JUDGE_WITH_TEXT=1
STEER_ANALYSIS=0
STEER_TOKEN_DEBUG=0
USE_PROJECT_VENV=0
MATH500_ALLOW_USER_SITE=0
# =====================================================================

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
  echo "当前解释器: $($PYCMD -c 'import sys; print(sys.executable)')"
  echo "PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-unset}"
  echo "--- Python 报错 ---"
  echo "$MV_ERR"
  echo "---"
  echo "若用 .venv: .venv/bin/python3 -m pip install math-verify 'antlr4-python3-runtime>=4.9'"
  echo "若用 conda 且 NOUSERSITE=1: conda 内 python -m pip install --force-reinstall 'antlr4-python3-runtime==4.13.2' math-verify"
  echo "或: sbatch 前 export MATH500_ALLOW_USER_SITE=1"
  exit 1
}
unset MV_ERR

LOG_DIR="log2/math500_steer_overwrite_${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then NUM_GPUS=1; fi

# 题号与分片（与 run_basic_overwrite.py 一致：question_idx % num_shards == shard_id）
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
    --num_examples "$NUM_EXAMPLES" \
    --max_token "$MAX_TOKEN" \
    --shard_id $i \
    --num_shards $NUM_GPUS \
    "${PY_ARGS[@]}" \
    > "${LOG_DIR}/log_worker_${i}.txt" 2>&1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait $pid; done

echo "LOG_DIR=${LOG_DIR}"
echo "MANIFEST=${LOG_DIR}/math500_run_manifest.json （题号 0..N-1 与各 GPU 分片）"
echo "JUDGE=${LOG_DIR}/judge （子集重跑请把 run_math500_steer_overwrite_subset.slurm 里 PREFERRED_JUDGE 指到本目录）"
echo "各分片 Final Report（Base + Steer）：分别汇总 \"Base :\" / \"Steer:\" 分子，分母之和各应为 ${NUM_EXAMPLES}。"
grep -E 'Final Report|Base :|Steer:' "${LOG_DIR}"/log_worker_*.txt || true
