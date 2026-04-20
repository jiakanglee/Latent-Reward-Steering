#!/bin/bash
#SBATCH --job-name=MBPP_Conf_Sweep
#SBATCH --output=log2/mbpp_conf_sweep_%j.out
#SBATCH --error=log2/mbpp_conf_sweep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=ilab1
#SBATCH --gres=gpu:a4500:4

# MBPP 边例题 confidence_threshold sweep: 0.68~0.72 步长 0.01
# step_size=0.1125, num_steps=4, reward=0.9 固定
# 题目: 204 道边例题 (mbpp_edge_questions_122981.txt)

export PYTHONNOUSERSITE=1
source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p log2
export PYTHONPATH=$PWD:$PYTHONPATH

if [ -x "$PWD/.venv/bin/python3" ]; then PYCMD="$PWD/.venv/bin/python3"; else PYCMD="python"; fi

echo "MBPP conf sweep: 0.68~0.72 step=0.01 | step_size=0.1125 n=4 reward=0.9 | 204 edge questions"
echo "Date: $(date)"

REWARD_MODEL="transformer_reward_model_mbpp_best.pt"
REWARD=0.9
MAX_TOKEN=4000
NSTEPS=4
STEP=0.1125

# 204 道边例题
TARGET_MODE="--example_idx $(cat logs/mbpp_edge_questions_122981.txt)"

# conf: 0.68 0.69 0.70 0.71 0.72
CONF_LIST="0.68 0.69 0.70 0.71 0.72"

RESULTS_FILE="log2/mbpp_conf_sweep_results_${SLURM_JOB_ID}.txt"
> "$RESULTS_FILE"
echo "=== MBPP conf sweep: t=$MAX_TOKEN s=$STEP reward=$REWARD n=$NSTEPS (204 questions) ===" >> "$RESULTS_FILE"
echo "t      | step   | conf   | nsteps | total | Base对 | Steer对 | Base对Steer错 | Base错Steer对" >> "$RESULTS_FILE"
echo "-------|--------|--------|--------|-------|--------|---------|---------------|---------------" >> "$RESULTS_FILE"

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then NUM_GPUS=1; fi

for CONF in $CONF_LIST; do
  SWEEP_DIR="log2/mbpp_conf_sweep_c${CONF}_${SLURM_JOB_ID}"
  mkdir -p "$SWEEP_DIR"
  echo ""
  echo "========== conf=$CONF step_size=$STEP reward=$REWARD n=$NSTEPS =========="

  pids=()
  for ((i=0; i<NUM_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES=$i $PYCMD -u steering/run_basic_overwrite.py \
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
echo "MBPP conf sweep results"
echo "============================================================"
cat "$RESULTS_FILE"
echo "============================================================"
echo "Date: $(date)"
