#!/bin/bash
#SBATCH --job-name=CollectMBPP
#SBATCH --output=log2/collect_mbpp_%j.out
#SBATCH --error=log2/collect_mbpp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=rlab6
#SBATCH --gres=gpu:a5000:4

# Collect SAE latents on MBPP train+test, max_token=2000 | 4 卡并行
# Output: collected_sae_latents_10dim_2000_mbpp_train_test.pt (覆盖旧版)

export PYTHONNOUSERSITE=1
source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p log2
export PYTHONPATH=$PWD:$PYTHONPATH

if [ -x "$PWD/.venv/bin/python3" ]; then PYCMD="$PWD/.venv/bin/python3"; else PYCMD="python"; fi

OUTPUT_BASE="collected_sae_latents_10dim_2000_mbpp_train_test"

echo "Collecting MBPP train+test (874) | max_token=2000 | 4 GPUs"
echo "Date: $(date)"

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then NUM_GPUS=1; fi

pids=()
for ((i=0; i<NUM_GPUS; i++)); do
  CUDA_VISIBLE_DEVICES=$i $PYCMD -u collect_data/generate_data_7B.py \
    --dataset mbpp_train_test \
    --max_token 2000 \
    --output_file "${OUTPUT_BASE}.pt" \
    --shard_id $i \
    --num_shards $NUM_GPUS \
    > "log2/collect_mbpp_shard_${i}_${SLURM_JOB_ID}.txt" 2>&1 &
  pids+=($!)
done

for pid in "${pids[@]}"; do wait $pid; done
echo "  All shards done. Merging..."

$PYCMD merge_collect_shards.py --base "$OUTPUT_BASE"

echo "Done. Date: $(date)"
