#!/bin/bash
#SBATCH --job-name=TrainPrefixRM
#SBATCH --output=log2/train_prefix_rm_%j.out
#SBATCH --error=log2/train_prefix_rm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=rlab1
#SBATCH --gres=gpu:a100:4

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

# 优先用项目里的虚拟环境；没有就退回系统 python
if [ -x "$PWD/.venv/bin/python3" ]; then
    PYCMD="$PWD/.venv/bin/python3"
else
    PYCMD="python"
fi

echo "======================================================"
echo "🚀 Starting Prefix Reward Model Training"
echo "Date      : $(date)"
echo "Node      : $(hostname)"
echo "Python    : $(which python)"
echo "PYCMD     : $PYCMD"
echo "Workdir   : $PWD"
echo "CUDA info :"
nvidia-smi
echo "======================================================"

# =========================================================
# 2. 训练参数
# =========================================================
DATA_FILE="/common/home/jl3614/Desktop/Python_project/thinking-llms-interp/collected_sae_latents_10dim_2000.pt"
SAVE_PATH="/common/home/jl3614/Desktop/Python_project/thinking-llms-interp/transformer_reward_model_prefix.pt"

EPOCHS=30
LR=5e-4
HIDDEN_DIM=128

# =========================================================
# 3. 开始训练
# =========================================================
$PYCMD train_reward_model_prefix.py \
    --data_file "$DATA_FILE" \
    --save_path "$SAVE_PATH" \
    --epochs $EPOCHS \
    --lr $LR \
    --hidden_dim $HIDDEN_DIM

EXIT_CODE=$?

echo "======================================================"
echo "✅ Training finished"
echo "Exit code : $EXIT_CODE"
echo "Date      : $(date)"
echo "======================================================"

exit $EXIT_CODE