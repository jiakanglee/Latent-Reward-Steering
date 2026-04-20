#!/bin/bash
#SBATCH --job-name=Transformer_Iter_Steer
#SBATCH --output=log2/iter_steer_%j.out
#SBATCH --error=log2/iter_steer_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=rlab2
#SBATCH --gres=gpu:a4000:1

# === 1. 环境配置 ===
export PYTHONNOUSERSITE=1
source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p log2
export PYTHONPATH=$PWD:$PYTHONPATH

if [ -x "$PWD/.venv/bin/python3" ]; then PYCMD="$PWD/.venv/bin/python3"; else PYCMD="python"; fi

echo "🚀 Starting Transformer Iterative Steering..."
echo "Date: $(date)"

# === 2. 关键参数设置 ===
REWARD_MODEL="transformer_reward_model_best.pt"

# 🔴 Iterative Steering 参数 (代替了原来的 Strength)
# STEP_SIZE: 单次梯度上升的步长 (类似学习率)
# NUM_STEPS: 原地迭代优化的次数 (类似 Epochs)
# 总干预强度 ≈ STEP_SIZE * NUM_STEPS
STEP_SIZE=2.0
NUM_STEPS=5

# 测试样本数 (迭代模式比较慢，建议先测 20 个看效果)
NUM_EXAMPLES=500

# === 3. 运行脚本 ===
# 注意：文件名改为了 run_basic_overwrite.py
$PYCMD -u steering/run_basic_overwrite.py \
    --model "Open-Reasoner-Zero/Open-Reasoner-Zero-7B" \
    --sae_layer 20 \
    --n_clusters 10 \
    --reward_model_path "$REWARD_MODEL" \
    --step_size $STEP_SIZE \
    --num_steps $NUM_STEPS \
    --example_idx 11 14 33 36 60 64 80 92\
    --max_token 2000 \
    --print_response \
    # --monitor
    #--example_idx 12 15 34 37 61 65 81 93 97 101\
    #  --num_examples $NUM_EXAMPLES \

echo "✅ Experiment Finished at $(date)"