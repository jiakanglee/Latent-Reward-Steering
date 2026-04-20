#!/bin/bash
# MATH-500 baseline：与 run_basic_overwrite.py 相同 greedy 生成；仅 --baseline_only；
# 判题为 rule / math_verify（需 pip install math-verify）。
#
# 使用前请按需修改 #SBATCH：partition、nodelist、gres、mem、time。
# 全量 500 题建议 DEBUG_PRINT=0（默认），否则 --print_response 会极大膨胀 log_worker_*.txt。
# 跑完后汇总：全局正确数 = 各分片 "Base : 分子/分母" 的分子之和；准确率 = 总和/500。
#   grep -E 'Final Report|Base :' "${LOG_DIR}"/log_worker_*.txt
#
#SBATCH --job-name=MATH500_BaseOverwrite
#SBATCH --output=log2/math500_baseline_overwrite_%j.out
#SBATCH --error=log2/math500_baseline_overwrite_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=rlab6
#SBATCH --gres=gpu:a5000:3

# 不修改 run_basic_overwrite.py：须与历史一致使用「能正常 model.generate 的 Python/nnsight」。
# 默认优先 .venv（你原先 Slurm/本地可用的栈）；新版 conda 里 nnsight 的 model.generate 可能返回 InterleavingTracer 报错。
# 强制用 conda 跑本脚本时：export USE_CONDA_PYTHON=1
# conda + PYTHONNOUSERSITE=1 时 antlr 须在 conda 内：见下方 MATH500_ALLOW_USER_SITE / pip 说明
source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p log2
export PYTHONPATH=$PWD:$PYTHONPATH

unset PYTHONNOUSERSITE
if [ "${USE_CONDA_PYTHON:-0}" = "1" ]; then
  PYCMD="${CONDA_PREFIX}/bin/python"
  if [ ! -x "$PYCMD" ]; then PYCMD="python"; fi
  if [ "${MATH500_ALLOW_USER_SITE:-0}" != "1" ]; then
    export PYTHONNOUSERSITE=1
  fi
elif [ -x "$PWD/.venv/bin/python3" ]; then
  PYCMD="$PWD/.venv/bin/python3"
else
  PYCMD="${CONDA_PREFIX}/bin/python"
  if [ ! -x "$PYCMD" ]; then PYCMD="python"; fi
  if [ "${MATH500_ALLOW_USER_SITE:-0}" != "1" ]; then
    export PYTHONNOUSERSITE=1
  fi
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

MODEL_NAME="${MODEL_NAME:-Open-Reasoner-Zero/Open-Reasoner-Zero-7B}"
MAX_TOKEN="${MAX_TOKEN:-4000}"
NUM_EXAMPLES="${NUM_EXAMPLES:-500}"
DEBUG_PRINT="${DEBUG_PRINT:-0}"

LOG_DIR="log2/math500_baseline_overwrite_${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then NUM_GPUS=1; fi

EXTRA_PY_ARGS=()
if [ "$DEBUG_PRINT" = "1" ]; then
  EXTRA_PY_ARGS+=(--print_response)
fi

pids=()
for ((i=0; i<NUM_GPUS; i++)); do
  CUDA_VISIBLE_DEVICES=$i $PYCMD -u steering/run_basic_overwrite.py \
    --model "$MODEL_NAME" \
    --dataset math500 \
    --baseline_only \
    --num_examples "$NUM_EXAMPLES" \
    --max_token "$MAX_TOKEN" \
    --shard_id $i \
    --num_shards $NUM_GPUS \
    "${EXTRA_PY_ARGS[@]}" \
    > "${LOG_DIR}/log_worker_${i}.txt" 2>&1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait $pid; done

echo "LOG_DIR=${LOG_DIR}"
echo "各分片 Final Report（baseline_only）：全局正确数 = 各 \"Base : x/N\" 的 x 相加；分母之和应为 ${NUM_EXAMPLES}。"
grep -E 'Final Report|Base :' "${LOG_DIR}"/log_worker_*.txt || true
