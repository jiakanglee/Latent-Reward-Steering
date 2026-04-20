#!/bin/bash
# 训练 Transformer reward model（train_latent_classifier_7B.py）
# 提交: 在仓库根目录  sbatch log2/slurms/train_latent_classifier_7B.slurm
#
#SBATCH --job-name=Train_Transformer
#SBATCH --output=log2/train_tf_%j.out
#SBATCH --error=log2/train_tf_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
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

DATA_FILE="${DATA_FILE:-collected_sae_latents_10dim_2000.pt}"
SAVE_PATH="${SAVE_PATH:-transformer_reward_model.pt}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-0.0003}"

echo "🚀 train_latent_classifier_7B | data=$DATA_FILE -> $SAVE_PATH"
echo "Date: $(date)"

$PYCMD -u train_latent_classifier_7B.py \
  --data_file "$DATA_FILE" \
  --save_path "$SAVE_PATH" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --hidden_dim "$HIDDEN_DIM"

echo "✅ Done: $(date)"
