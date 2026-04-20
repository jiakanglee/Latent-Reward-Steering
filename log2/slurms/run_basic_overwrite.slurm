#!/bin/bash
# 主实验：迭代 steering（run_basic_overwrite.py）
# 提交: 在仓库根目录  sbatch log2/slurms/run_basic_overwrite.slurm
#
#SBATCH --job-name=IterSteer
#SBATCH --output=log2/iter_steer_%j.out
#SBATCH --error=log2/iter_steer_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
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

REWARD_MODEL="${REWARD_MODEL:-transformer_reward_model_best.pt}"
STEP_SIZE="${STEP_SIZE:-2.0}"
NUM_STEPS="${NUM_STEPS:-5}"
MAX_TOKEN="${MAX_TOKEN:-2000}"

$PYCMD -u run_basic_overwrite.py \
  --model "Open-Reasoner-Zero/Open-Reasoner-Zero-7B" \
  --sae_layer 20 \
  --n_clusters 10 \
  --reward_model_path "$REWARD_MODEL" \
  --step_size "$STEP_SIZE" \
  --num_steps "$NUM_STEPS" \
  --example_idx 11 14 33 36 60 64 80 92 \
  --max_token "$MAX_TOKEN" \
  --print_response \
  --no-monitor

echo "✅ Done: $(date)"
