#!/bin/bash
#SBATCH --job-name=Train_Causal_RM
#SBATCH --output=log2/train_causal_%j.out
#SBATCH --error=log2/train_causal_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=rlab2
#SBATCH --gres=gpu:a4000:1

# === 1. 环境配置 ===
export PYTHONNOUSERSITE=1
source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

# 设置项目根目录
cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p log2
export PYTHONPATH=$PWD:$PYTHONPATH

# 检查 Python 解释器
if [ -x "$PWD/.venv/bin/python3" ]; then PYCMD="$PWD/.venv/bin/python3"; else PYCMD="python"; fi

echo "🚀 Starting Causal Transformer Reward Model Training..."
echo "Date: $(date)"
echo "Node: $(hostname)"

# === 2. 参数设置 ===
# 确保你的数据文件名正确
DATA_FILE="collected_sae_latents_10dim_2000.pt"

# 训练脚本的名字 (假设你存为了 train_causal_rm.py)
TRAIN_SCRIPT="train_7B_causal_mask.py"

EPOCHS=50
BATCH_SIZE=32
LR=0.0003

# === 3. 执行训练 ===
echo "📋 Configuration:"
echo "   Input Data:  $DATA_FILE"
echo "   Epochs:      $EPOCHS"
echo "   Batch Size:  $BATCH_SIZE"
echo "   LR:          $LR"

# 运行训练
# 现在的代码会自动保存 transformer_casual_mask_best_loss.pt 和 _best_acc.pt
$PYCMD -u $TRAIN_SCRIPT \
    --data_file "$DATA_FILE" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR

echo "✅ Training Finished at $(date)"