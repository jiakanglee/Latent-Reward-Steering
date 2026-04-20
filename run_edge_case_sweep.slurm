#!/bin/bash
#SBATCH --job-name=Edge_Case_Sweep
#SBATCH --output=log2/edge_sweep_%j.out
#SBATCH --error=log2/edge_sweep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:8

# Sweep：97 道边例题 | conf 0.68~0.75 步长0.01 | step_size 1.1125~1.15 共10值 | t2500 n4 reward=0.8

export PYTHONNOUSERSITE=1
source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p log2
export PYTHONPATH=$PWD:$PYTHONPATH

if [ -x "$PWD/.venv/bin/python3" ]; then PYCMD="$PWD/.venv/bin/python3"; else PYCMD="python"; fi

echo "Sweep: 97 edge-case | conf 0.68~0.75 step=0.01 | step_size 1.1125~1.15 x10 | t2500 n4 reward=0.8"
echo "Date: $(date)"

REWARD_MODEL="transformer_reward_model_best.pt"
REWARD=0.8
MAX_TOKEN=2500
NSTEPS=4
CONF_LIST="0.68 0.69 0.70 0.71 0.72 0.73 0.74 0.75"
# step_size: 1.1125~1.15 共10个等间距值
STEP_LIST=$(awk 'BEGIN{for(i=0;i<=9;i++)printf "%.4f ", 1.1125 + i*(1.15-1.1125)/9}')

# 97 道边例题库
TARGET_MODE="--example_idx 9 11 14 17 19 41 43 60 64 80 92 94 96 100 101 103 108 109 110 126 129 138 145 147 150 152 153 154 156 164 168 170 177 184 189 206 213 219 222 224 228 232 239 240 248 251 264 274 279 284 286 292 295 296 302 303 306 308 309 317 320 324 327 332 340 352 355 358 359 365 371 372 391 392 400 401 408 412 422 425 432 444 445 466 470 475 477 478 481 483 490 491 494 497 498"

RESULTS_FILE="log2/step_sweep_${SLURM_JOB_ID}.txt"
> "$RESULTS_FILE"
echo "=== conf x step_size sweep: t=$MAX_TOKEN n=$NSTEPS reward=$REWARD (97 questions) ===" >> "$RESULTS_FILE"
echo "t      | step   | conf   | nsteps | total | Base对 | Steer对 | Base对Steer错 | Base错Steer对" >> "$RESULTS_FILE"
echo "-------|--------|--------|--------|-------|--------|---------|---------------|---------------" >> "$RESULTS_FILE"

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then NUM_GPUS=1; fi

for CONF in $CONF_LIST; do
  for STEP in $STEP_LIST; do
    SWEEP_DIR="log2/step_sweep_t${MAX_TOKEN}_r${REWARD}_c${CONF}_s${STEP}_n${NSTEPS}"
    mkdir -p "$SWEEP_DIR"
    echo ""
    echo "========== conf=$CONF step_size=$STEP num_steps=$NSTEPS =========="
    
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
done

echo ""
echo "============================================================"
echo "Sweep 最终结果"
echo "============================================================"
cat "$RESULTS_FILE"
echo "============================================================"
echo "Date: $(date)"
