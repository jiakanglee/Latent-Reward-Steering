#!/bin/bash
# 单题导出 GPQA 解码（scripts/dump_gpqa_decode_sample.py）
# 提交: 在仓库根目录  sbatch log2/slurms/run_dump_gpqa_decode_sample.slurm
#
#SBATCH --job-name=GPQA_DUMP1
#SBATCH --output=log2/gpqa_dump_decode_%j.out
#SBATCH --error=log2/gpqa_dump_decode_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1

set -euo pipefail

_SL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "${_SL}/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p log2
export PYTHONPATH=$PWD:$PYTHONPATH

export PYTHONNOUSERSITE=1
if [ -n "${CONDA_INIT_SH:-}" ] && [ -f "${CONDA_INIT_SH}" ]; then
  # shellcheck source=/dev/null
  source "${CONDA_INIT_SH}"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV:-stllms_env}"

if [ -x "$PWD/.venv/bin/python3" ]; then PYCMD="$PWD/.venv/bin/python3"; else PYCMD="python"; fi

MODEL="${MODEL:-Open-Reasoner-Zero/Open-Reasoner-Zero-7B}"
QUESTION_IDX="${QUESTION_IDX:-0}"
MAX_TOKEN="${MAX_TOKEN:-4000}"
OUT_FILE="${OUT_FILE:-log2/gpqa_decode_sample_q${QUESTION_IDX}_job${SLURM_JOB_ID}.txt}"

$PYCMD -u scripts/dump_gpqa_decode_sample.py \
  --model "$MODEL" \
  --question_idx "$QUESTION_IDX" \
  --max_token "$MAX_TOKEN" \
  --out "$OUT_FILE"

echo "Wrote: $OUT_FILE"
