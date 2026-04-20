#!/bin/bash
#SBATCH --job-name=Collect_Llama8B_MultiGPU
#SBATCH --output=logs/collect_llama8b_multigpu_%j.out
#SBATCH --error=logs/collect_llama8b_multigpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=ilab2
#SBATCH --gres=gpu:a4500:4

# --- 配置变量 ---
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
LAYER_INDEX=6
N_CLUSTERS=15
# MAX_TOKENS: math500=2000, others=4000 (auto-set in generate_data_llama_8B.py when not passed)
NUM_EXAMPLES=""
# ---------------

export PYTHONNOUSERSITE=1
source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p Vicky/logs
export PYTHONPATH=$PWD:$PYTHONPATH

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then NUM_GPUS=1; fi
echo "Using $NUM_GPUS GPUs"

for DATASET in aime24_aime25; do
  MAX_TOKENS=4000
  OUTPUT_BASE="Vicky/collected_sae_latents_8b_${N_CLUSTERS}dim_${MAX_TOKENS}_${DATASET}"
  echo "=== Dataset: $DATASET ==="

  pids=()
  for ((i=0; i<NUM_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES=$i python -u Vicky/generate_data_llama_8B.py \
      --model "$MODEL_NAME" \
      --sae_layer $LAYER_INDEX \
      --n_clusters $N_CLUSTERS \
      --max_token $MAX_TOKENS \
      --dataset $DATASET \
      ${NUM_EXAMPLES:+--num_examples $NUM_EXAMPLES} \
      --output_file "${OUTPUT_BASE}.pt" \
      --shard_id $i \
      --num_shards $NUM_GPUS \
      > "Vicky/logs/collect_llama8b_${DATASET}_shard_${i}_${SLURM_JOB_ID}.txt" 2>&1 &
    pids+=($!)
  done

  for pid in "${pids[@]}"; do wait $pid; done
  echo "All shards done for $DATASET. Merging..."
  python merge_collect_shards.py --base "$OUTPUT_BASE"
  echo "Done: $DATASET"
done

echo "All datasets done."
