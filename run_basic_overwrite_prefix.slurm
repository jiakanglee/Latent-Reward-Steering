#!/bin/bash
#SBATCH --job-name=RunPrefixSteer
#SBATCH --output=log2/run_prefix_steer_%j.out
#SBATCH --error=log2/run_prefix_steer_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=rlab2
#SBATCH --gres=gpu:a4000:1

# =========================================================
# 1. 环境配置
# =========================================================
set -e
set -u

export PYTHONNOUSERSITE=1

source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p log2
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

mkdir -p logs

if [ -x "$PWD/.venv/bin/python3" ]; then
    PYCMD="$PWD/.venv/bin/python3"
else
    PYCMD="python"
fi

echo "======================================================"
echo "🚀 Starting Prefix Steering Inference"
echo "Date      : $(date)"
echo "Node      : $(hostname)"
echo "Python    : $(which python)"
echo "PYCMD     : $PYCMD"
echo "Workdir   : $PWD"
echo "CUDA info :"
nvidia-smi
echo "======================================================"

# =========================================================
# 2. 路径与参数
# =========================================================
MODEL_NAME="Open-Reasoner-Zero/Open-Reasoner-Zero-7B"
REWARD_MODEL_PATH="/common/home/jl3614/Desktop/Python_project/thinking-llms-interp/transformer_reward_model_prefix_best.pt"

SAE_LAYER=20
N_CLUSTERS=10

STEP_SIZE=2
NUM_STEPS=1
DELTA_CLIP=0.2

REWARD_PREFIX_WINDOW=128
STEER_AFTER=1
STEER_EVERY=1

MAX_TOKEN=2000

# =========================================================
# 3. 题目设置
#    二选一：
#    A. 指定题号列表（推荐先跑你 Base 错题）
#    B. 跑前 N 个题
# =========================================================

# ---------- A. 指定题号 ----------
EXAMPLE_IDX="9 11 14 19 21"

# ---------- B. 跑前N个 ----------
# NUM_EXAMPLES=5

# =========================================================
# 4. 执行推理
# =========================================================

# ---- 方案A：指定题号 ----
$PYCMD run_basic_overwrite_prefix.py \
    --model "$MODEL_NAME" \
    --sae_layer $SAE_LAYER \
    --n_clusters $N_CLUSTERS \
    --reward_model_path "$REWARD_MODEL_PATH" \
    --step_size $STEP_SIZE \
    --num_steps $NUM_STEPS \
    --delta_clip $DELTA_CLIP \
    --reward_prefix_window $REWARD_PREFIX_WINDOW \
    --steer_after $STEER_AFTER \
    --steer_every $STEER_EVERY \
    --example_idx $EXAMPLE_IDX \
    --max_token $MAX_TOKEN \
    --monitor

# ---- 方案B：跑前N个题（不用时保持注释）----
# $PYCMD run_basic_overwrite_prefix.py \
#     --model "$MODEL_NAME" \
#     --sae_layer $SAE_LAYER \
#     --n_clusters $N_CLUSTERS \
#     --reward_model_path "$REWARD_MODEL_PATH" \
#     --step_size $STEP_SIZE \
#     --num_steps $NUM_STEPS \
#     --delta_clip $DELTA_CLIP \
#     --reward_prefix_window $REWARD_PREFIX_WINDOW \
#     --steer_after $STEER_AFTER \
#     --steer_every $STEER_EVERY \
#     --num_examples $NUM_EXAMPLES \
#     --max_token $MAX_TOKEN \
#     --monitor

EXIT_CODE=$?

echo "======================================================"
echo "✅ Inference finished"
echo "Exit code : $EXIT_CODE"
echo "Date      : $(date)"
echo "======================================================"

exit $EXIT_CODE