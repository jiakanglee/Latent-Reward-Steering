#!/bin/bash
#SBATCH --job-name=AIME_RM_MATH97
#SBATCH --output=log2/aime_rm_math97_%j.out
#SBATCH --error=log2/aime_rm_math97_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=rlab3
#SBATCH --gres=gpu:a4000:4

# AIME reward model (aime24+25 train) 在 MATH-500 的 97 道边例题上评测
# 对比：用 AIME RM 替代 MATH RM，看跨域效果

export PYTHONNOUSERSITE=1
source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p log2
export PYTHONPATH=$PWD:$PYTHONPATH

if [ -x "$PWD/.venv/bin/python3" ]; then PYCMD="$PWD/.venv/bin/python3"; else PYCMD="python"; fi

echo "AIME RM on MATH-500 97 edge cases | t2500 s1.15 c0.72 r0.9 n4"
echo "Date: $(date)"

REWARD_MODEL="transformer_reward_model_aime_best.pt"
REWARD_LIST="0.9"
MAX_TOKEN=2500
NSTEPS=4
CONF=0.72
STEP=1.15

# 97 道边例题库 (来自 edge_case_97_questions.json)
TARGET_MODE="--example_idx 9 11 14 17 19 41 43 60 64 80 92 94 96 100 101 103 108 109 110 126 129 138 145 147 150 152 153 154 156 164 168 170 177 184 189 206 213 219 222 224 228 232 239 240 248 251 264 274 279 284 286 292 295 296 302 303 306 308 309 317 320 324 327 332 340 352 355 358 359 365 371 372 391 392 400 401 408 412 422 425 432 444 445 466 470 475 477 478 481 483 490 491 494 497 498"

RESULTS_FILE="log2/aime_rm_math97_results_${SLURM_JOB_ID}.txt"
> "$RESULTS_FILE"
echo "=== AIME RM on MATH-500 97 edge: t=$MAX_TOKEN s=$STEP c=$CONF r=0.9 n=$NSTEPS ===" >> "$RESULTS_FILE"
echo "t      | step   | conf   | reward | nsteps | total | Base对 | Steer对 | Base对Steer错 | Base错Steer对" >> "$RESULTS_FILE"
echo "-------|--------|--------|--------|--------|-------|--------|---------|---------------|---------------" >> "$RESULTS_FILE"

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then NUM_GPUS=1; fi

for REWARD in $REWARD_LIST; do
  LOG_DIR="log2/aime_rm_math97_r${REWARD}_${SLURM_JOB_ID}"
  mkdir -p "$LOG_DIR"
  echo ""
  echo "========== reward=$REWARD =========="

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
      --dataset math500 \
      --print_response \
      --save_judge_reason \
      --judge_reason_dir "${LOG_DIR}/judge_reasons" \
      --shard_id $i \
      --num_shards $NUM_GPUS \
      $TARGET_MODE \
      > "${LOG_DIR}/log_worker_${i}.txt" 2>&1 &
    pids+=($!)
  done

  for pid in "${pids[@]}"; do wait $pid; done
  echo "  Done. Collecting results..."

  $PYCMD collect_sweep_results.py \
    --log_dir "$LOG_DIR" \
    --output "$RESULTS_FILE" \
    --label "t${MAX_TOKEN}_s${STEP}_c${CONF}_r${REWARD}_n${NSTEPS}"
done

echo ""
echo "============================================================"
echo "AIME RM on MATH-500 97 edge cases 评测结果"
echo "============================================================"
cat "$RESULTS_FILE"
echo "============================================================"
echo "Date: $(date)"
