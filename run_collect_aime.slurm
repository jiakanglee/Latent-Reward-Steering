#!/bin/bash
#SBATCH --job-name=CollectAIME
#SBATCH --output=log2/collect_aime_%j.out
#SBATCH --error=log2/collect_aime_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=rlab1
#SBATCH --gres=gpu:4500_ada:2

# Collect SAE latents on AIME24 + AIME25, max_token=4000
# Output: collected_sae_latents_10dim_4000_aime24_aime25.pt

export PYTHONNOUSERSITE=1
source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p log2
export PYTHONPATH=$PWD:$PYTHONPATH

if [ -x "$PWD/.venv/bin/python3" ]; then PYCMD="$PWD/.venv/bin/python3"; else PYCMD="python"; fi

echo "Collecting AIME24 + AIME25 | max_token=4000"
echo "Date: $(date)"

# Add --load_in_8bit if OOM on A4000 (16GB)
$PYCMD -u generate_data_7B.py \
  --dataset aime24_aime25 \
  --max_token 4000 \
  --output_file collected_sae_latents_10dim_4000_aime24_aime25.pt

echo "Done. Date: $(date)"
