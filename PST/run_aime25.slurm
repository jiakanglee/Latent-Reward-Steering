#!/bin/bash
#SBATCH --job-name=AIME25
#SBATCH --output=PST/logs/aime25_%j.out
#SBATCH --error=PST/logs/aime25_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=rlab4
#SBATCH --gres=gpu:a4000:4

# AIME 2025 评测：30 题 | AIME reward model | t4000 n4 reward=0.9 conf=0.72 step=1.15

export PYTHONNOUSERSITE=1
source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p PST/logs logs
export PYTHONPATH=$PWD:$PYTHONPATH

if [ -x "$PWD/.venv/bin/python3" ]; then PYCMD="$PWD/.venv/bin/python3"; else PYCMD="python"; fi

echo "AIME 2025: 30 questions | AIME RM | t4000 n4 reward=0.9 conf=0.72 step=1.15"
echo "Date: $(date)"

REWARD_MODEL="PST/transformer_reward_model_aime_1.5B_best.pt"
REWARD=0.9
MAX_TOKEN=4000
NSTEPS=4
CONF=0.72
STEP=0.9

LOG_DIR="PST/logs/aime25_${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then NUM_GPUS=1; fi

echo "========== AIME 2025 Evaluation =========="
pids=()
for ((i=0; i<NUM_GPUS; i++)); do
  CUDA_VISIBLE_DEVICES=$i $PYCMD -u PST/run_basic_overwrite_update.py \
    --model "Open-Reasoner-Zero/Open-Reasoner-Zero-1.5B" \
    --sae_layer 8 \
    --n_clusters 5 \
    --reward_model_path "$REWARD_MODEL" \
    --step_size $STEP \
    --num_steps $NSTEPS \
    --max_token $MAX_TOKEN \
    --reward_threshold $REWARD \
    --confidence_threshold $CONF \
    --dataset aime25 \
    --num_examples 30 \
    --print_response \
    --save_judge_reason \
    --judge_reason_dir "${LOG_DIR}/judge_reasons" \
    --shard_id $i \
    --num_shards $NUM_GPUS \
    > "${LOG_DIR}/log_worker_${i}.txt" 2>&1 &
  pids+=($!)
done

for pid in "${pids[@]}"; do wait $pid; done
echo "  Done. Collecting results..."

RESULTS_FILE="logs/aime25_results_${SLURM_JOB_ID}.txt"
echo "=== AIME 2025: t=$MAX_TOKEN n=$NSTEPS reward=$REWARD conf=$CONF step=$STEP (30 questions) ===" > "$RESULTS_FILE"
echo "t      | step   | conf   | nsteps | total | Base对 | Steer对 | Base对Steer错 | Base错Steer对" >> "$RESULTS_FILE"
$PYCMD collect_sweep_results.py \
  --log_dir "$LOG_DIR" \
  --output "$RESULTS_FILE" \
  --label "t${MAX_TOKEN}_s${STEP}_c${CONF}_n${NSTEPS}"

echo ""
echo "============================================================"
echo "AIME 2025 评测结果"
echo "============================================================"
cat "$RESULTS_FILE"
echo "============================================================"
echo "Date: $(date)"
