#!/bin/bash
# 采集 SAE 轨迹（collect_data/generate_data_7B.py）→ 供 train_reward_model 使用
# 提交: 在仓库根目录  sbatch collect_data/generate_data_7B.slurm
#
#SBATCH --job-name=GenData7B
#SBATCH --output=log2/generate_data_7b_%j.out
#SBATCH --error=log2/generate_data_7b_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1

set -euo pipefail

_SL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "${_SL}/.." && pwd)"
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

DATASET="${DATASET:-aime24_aime25}"
NUM_EXAMPLES="${NUM_EXAMPLES:-200}"
MAX_TOKEN="${MAX_TOKEN:-4000}"

$PYCMD -u collect_data/generate_data_7B.py \
  --dataset "$DATASET" \
  --num_examples "$NUM_EXAMPLES" \
  --max_token "$MAX_TOKEN" \
  --sae_layer 20 \
  --n_clusters 10

echo "Done: $(date)"
