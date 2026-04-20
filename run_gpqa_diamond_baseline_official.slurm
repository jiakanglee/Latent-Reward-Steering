#!/bin/bash
#SBATCH --job-name=GPQA_OfficialBase
#SBATCH --output=log2/gpqa_diamond_official_baseline_%j.out
#SBATCH --error=log2/gpqa_diamond_official_baseline_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=rlab4
#SBATCH --gres=gpu:a4000:4

# GPQA-Diamond baseline: T=1 top_p=1 top_k=0, evaluate_answer(mcqa)。资源对齐 run_gpqa_diamond_aime_rm.slurm

export PYTHONNOUSERSITE=1
source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p log2
export PYTHONPATH=$PWD:$PYTHONPATH

if [ -x "$PWD/.venv/bin/python3" ]; then PYCMD="$PWD/.venv/bin/python3"; else PYCMD="python"; fi

LOG_DIR="log2/gpqa_diamond_official_${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then NUM_GPUS=1; fi

pids=()
for ((i=0; i<NUM_GPUS; i++)); do
  CUDA_VISIBLE_DEVICES=$i $PYCMD -u run_gpqa_diamond_baseline_official.py \
    --model "Open-Reasoner-Zero/Open-Reasoner-Zero-7B" \
    --max_token 2500 \
    --shard_id $i \
    --num_shards $NUM_GPUS \
    --output_jsonl "${LOG_DIR}/results_shard_${i}.jsonl" \
    > "${LOG_DIR}/log_worker_${i}.txt" 2>&1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait $pid; done

echo "各分片结果（全局准确率 = 分子相加 / 198）："
grep "correct / total" "${LOG_DIR}"/log_worker_*.txt || true
