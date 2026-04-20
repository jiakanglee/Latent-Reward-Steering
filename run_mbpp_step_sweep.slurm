#!/bin/bash
#SBATCH --job-name=MBPP_Step_Sweep
#SBATCH --output=log2/mbpp_step_sweep_%j.out
#SBATCH --error=log2/mbpp_step_sweep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=ilab1
#SBATCH --gres=gpu:a4500:4

# MBPP 边例题 step_size sweep: 0.5~1.5 步长 0.1
# 题目: Initial对Steer错 + Initial错Steer对 + 两个都错 (共 204 题)
# 保持 reward=0.9 conf=0.72 num_steps=4 不变

export PYTHONNOUSERSITE=1
source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p log2
export PYTHONPATH=$PWD:$PYTHONPATH

if [ -x "$PWD/.venv/bin/python3" ]; then PYCMD="$PWD/.venv/bin/python3"; else PYCMD="python"; fi

echo "MBPP step_size sweep: 0.5~1.5 step=0.1 | 204 edge questions | reward=0.9 conf=0.72 n=4"
echo "Date: $(date)"

REWARD_MODEL="transformer_reward_model_mbpp_best.pt"
REWARD=0.9
MAX_TOKEN=4000
NSTEPS=4
CONF=0.72

# 204 道边例题 (Base对Steer错 + Base错Steer对 + 两个都错)
TARGET_MODE="--example_idx $(cat logs/mbpp_edge_questions_122981.txt)"

# step_size: 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5
STEP_LIST="0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5"

RESULTS_FILE="log2/mbpp_step_sweep_results_${SLURM_JOB_ID}.txt"
> "$RESULTS_FILE"
echo "=== MBPP step_size sweep: t=$MAX_TOKEN reward=$REWARD conf=$CONF n=$NSTEPS (204 questions) ===" >> "$RESULTS_FILE"
echo "t      | step   | conf   | nsteps | total | Base对 | Steer对 | Base对Steer错 | Base错Steer对" >> "$RESULTS_FILE"
echo "-------|--------|--------|--------|-------|--------|---------|---------------|---------------" >> "$RESULTS_FILE"

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then NUM_GPUS=1; fi

for STEP in $STEP_LIST; do
  SWEEP_DIR="log2/mbpp_step_sweep_s${STEP}_${SLURM_JOB_ID}"
  mkdir -p "$SWEEP_DIR"
  echo ""
  echo "========== step_size=$STEP reward=$REWARD conf=$CONF n=$NSTEPS =========="

  pids=()
  for ((i=0; i<NUM_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES=$i $PYCMD -u run_basic_overwrite.py \
      --model "Open-Reasoner-Zero/Open-Reasoner-Zero-7B" \
      --sae_layer 20 \
      --n_clusters 10 \
      --reward_model_path "$REWARD_MODEL" \
      --step_size $STEP \
      --num_steps $NSTEPS \
      --max_token $MAX_TOKEN \
      --reward_threshold $REWARD \
      --confidence_threshold $CONF \
      --dataset mbpp \
      --print_response \
      --save_judge_reason \
      --judge_reason_dir "${SWEEP_DIR}/judge_reasons" \
      --shard_id $i \
      --num_shards $NUM_GPUS \
      $TARGET_MODE \
      > "${SWEEP_DIR}/log_worker_${i}.txt" 2>&1 &
    pids+=($!)
  done

  for pid in "${pids[@]}"; do wait $pid; done
  echo "  Done. Collecting results..."

  $PYCMD collect_sweep_results.py \
    --log_dir "$SWEEP_DIR" \
    --output "$RESULTS_FILE" \
    --label "t${MAX_TOKEN}_s${STEP}_c${CONF}_n${NSTEPS}"
done

echo ""
echo "============================================================"
echo "MBPP step_size sweep  results"
echo "============================================================"
cat "$RESULTS_FILE"
echo "============================================================"
echo "Date: $(date)"
