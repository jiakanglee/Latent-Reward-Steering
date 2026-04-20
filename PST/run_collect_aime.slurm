#!/bin/bash
#SBATCH --job-name=CollectAIME
#SBATCH --output=PST/logs/collect_aime_%j.out
#SBATCH --error=PST/logs/collect_aime_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=ilab1
#SBATCH --gres=gpu:a4500:2

# Collect SAE latents on AIME24 + AIME25, max_token=4000
# Output: collected_sae_latents_5dim_4000_aime24_aime25.pt

export PYTHONNOUSERSITE=1
source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p PST/logs
export PYTHONPATH=$PWD:$PYTHONPATH

if [ -x "$PWD/.venv/bin/python3" ]; then PYCMD="$PWD/.venv/bin/python3"; else PYCMD="python"; fi

echo "Collecting AIME24 + AIME25 | max_token=4000"
echo "Date: $(date)"

# Add --load_in_8bit if OOM on A4000 (16GB)
$PYCMD -u PST/generate_data_1.5B.py \
  --dataset aime24_aime25 \
  --max_token 4000 \
  --output_file PST/collected_sae_latents_5dim_4000_aime24_aime25.pt

echo "Done. Date: $(date)"
