#!/bin/bash
#SBATCH --job-name=Train_Probe_Qwen
#SBATCH --output=log2/train_probe_%j.out
#SBATCH --error=log2/train_probe_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=rlab6
#SBATCH --gres=gpu:a5000:2

export PYTHONNOUSERSITE=1
source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p log2
export PYTHONPATH=$PWD:$PYTHONPATH

if [ -x "$PWD/.venv/bin/python3" ]; then
  PYCMD="$PWD/.venv/bin/python3"
else
  UV_BIN=$(which uv 2>/dev/null || echo "/common/home/jl3614/miniconda/envs/stllms_env/bin/uv")
  PYCMD="${UV_BIN} run -- python"
fi

echo "🚀 Starting Probe Training..."

# === 关键配置 ===
# 1. 模型: 对应你 json 文件里的名字
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# 2. 层数: 选 10 (与 Steering Vectors 的通常位置对齐)
LAYER=10

echo "Configuration: Model=$MODEL, Layer=$LAYER"

# 3. 运行训练
$PYCMD -u hybrid/train_probes.py \
    --model "$MODEL" \
    --probe_layer $LAYER \
    --epochs 20 \
    --batch_size 32 \
    --lr 0.001 \
    --seed 42 \
    --num_samples 2000 # 样本多一点更准，反正 A100 跑得快

echo "✅ Training finished."
echo "Probe should be saved at: results/vars/probe_layer${LAYER}_${MODEL}.pt"