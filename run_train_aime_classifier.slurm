#!/bin/bash
#SBATCH --job-name=TrainAIMEClf
#SBATCH --output=log2/train_aime_clf_%j.out
#SBATCH --error=log2/train_aime_clf_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1

# Train latent classifier on AIME24+AIME25 collected data
# Data: collected_sae_latents_10dim_4000_aime24_aime25.pt
# Output: transformer_reward_model_aime.pt, transformer_reward_model_aime_best.pt

export PYTHONNOUSERSITE=1
source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p log2
export PYTHONPATH=$PWD:$PYTHONPATH

if [ -x "$PWD/.venv/bin/python3" ]; then PYCMD="$PWD/.venv/bin/python3"; else PYCMD="python"; fi

DATA_FILE="collected_sae_latents_10dim_4000_aime24_aime25.pt"
SAVE_PATH="transformer_reward_model_aime.pt"

echo "Training on AIME24+AIME25 latent data"
echo "Data: $DATA_FILE"
echo "Date: $(date)"

$PYCMD -u train_latent_classifier_7B.py \
  --data_file "$DATA_FILE" \
  --save_path "$SAVE_PATH" \
  --epochs 30 \
  --lr 0.0005 \
  --hidden_dim 128

echo "Done. Best model: transformer_reward_model_aime_best.pt"
echo "Date: $(date)"
