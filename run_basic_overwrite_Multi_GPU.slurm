#!/bin/bash
#SBATCH --job-name=Transformer_Iter_Steer
#SBATCH --output=log2/iter_steer_%j.out
#SBATCH --error=log2/iter_steer_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16         # ⚠️ 建议增加CPU核数，每张卡分配4个核
#SBATCH --mem=64G                  # ⚠️ 建议增加内存，防止多进程OOM
#SBATCH --time=48:00:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=rlab1
#SBATCH --gres=gpu:a100:4         # 🔥 【关键修改】申请 4 张 A4000

# === 1. 环境配置 ===
export PYTHONNOUSERSITE=1
source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p log2
export PYTHONPATH=$PWD:$PYTHONPATH

if [ -x "$PWD/.venv/bin/python3" ]; then PYCMD="$PWD/.venv/bin/python3"; else PYCMD="python"; fi

echo "🚀 Starting Parallel Transformer Iterative Steering..."
echo "Date: $(date)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

# === 2. 关键参数设置 ===
REWARD_MODEL="transformer_reward_model_best.pt"
STEP_SIZE=2
NUM_STEPS=5

# 🔥 【模式选择】
# 原本的题目列表
#TARGET_MODE="--example_idx 17 68 101 104 147 150 180 189 213 214 233 251 299 324 358 396 419 14 19 41 108 109 129 145 153 224 372 391 408 416 439 483"
# baseline 会做错的题目，看 steer 能否修复
#TARGET_MODE="--example_idx 9 11 14 19 21 41 43 60 62 64 80 92 94 96 100 103 108 109 110 119 126 129 138 145 152 153 154 156 164 168 170 177 184 206 219 222 224 228 232 239 240 248 264 274 279 284 285 286 292 295 302 303 306 308 320 327 332 340 352 355 359 371 372 391 392 400 408 412 416 422 432 439 444 466 475 478 483 494 498"
TARGET_MODE="--num_examples 500"
# === 3. 参数网格 ===
REWARD=0.8
MAX_TOKEN=3000
CONF_LIST="0.73"

# 结果汇总文件
RESULTS_FILE="log2/threshold_sweep_${SLURM_JOB_ID}.txt"
> "$RESULTS_FILE"
echo "=== max_token=$MAX_TOKEN reward=$REWARD conf=0.73 (全部500题) ===" >> "$RESULTS_FILE"
echo "reward  | conf   | total | Base对 | Steer对 | Base对Steer错 | Base错Steer对" >> "$RESULTS_FILE"
echo "--------|--------|-------|--------|---------|---------------|---------------" >> "$RESULTS_FILE"

# 自动检测 GPU 数
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then NUM_GPUS=1; fi
echo "🔥 Detected $NUM_GPUS GPUs. Running all 500 questions (reward=$REWARD conf=0.73)..."

# === 4. conf 细扫 ===
for CONF in $CONF_LIST; do
    SWEEP_DIR="log2/sweep_t${MAX_TOKEN}_r${REWARD}_c${CONF}"
    mkdir -p "$SWEEP_DIR"
    echo ""
    echo "========== max_token=$MAX_TOKEN reward=$REWARD conf=$CONF =========="
    
    pids=()
    for ((i=0; i<NUM_GPUS; i++)); do
      CUDA_VISIBLE_DEVICES=$i $PYCMD -u steering/run_basic_overwrite.py \
        --model "Open-Reasoner-Zero/Open-Reasoner-Zero-7B" \
        --sae_layer 20 \
        --n_clusters 10 \
        --reward_model_path "$REWARD_MODEL" \
        --step_size $STEP_SIZE \
        --num_steps $NUM_STEPS \
        --max_token $MAX_TOKEN \
        --reward_threshold $REWARD \
        --confidence_threshold $CONF \
        --print_response \
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
      --label "r${REWARD}_c${CONF}"
done

# === 5. 最终结果汇总 ===
echo ""
echo "============================================================"
echo "📊 THRESHOLD SWEEP 最终结果汇总"
echo "============================================================"
cat "$RESULTS_FILE"
echo "============================================================"
echo "✅ 结果已保存到 $RESULTS_FILE"
echo "Date: $(date)"