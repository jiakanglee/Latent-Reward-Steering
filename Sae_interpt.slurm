#!/bin/bash
#SBATCH --job-name=SAE_Interp
#SBATCH --output=log2/sae_interp_%j.out
#SBATCH --error=log2/sae_interp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
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

# 检查 Python 解释器
if [ -x "$PWD/.venv/bin/python3" ]; then PYCMD="$PWD/.venv/bin/python3"; else PYCMD="python"; fi

echo "🚀 Starting SAE 10-Dim Interpretability Analysis..."
echo "Date: $(date)"
echo "Node: $(hostname)"

# === 2. 参数设置 ===
MODEL_NAME="Open-Reasoner-Zero/Open-Reasoner-Zero-7B"
SAE_LAYER=20
N_CLUSTERS=10
NUM_SAMPLES=500
TOP_K=15

# === 3. 执行分析 ===
echo "📋 Configuration:"
echo "   Model: $MODEL_NAME"
echo "   SAE Layer: $SAE_LAYER"
echo "   Clusters (Dim): $N_CLUSTERS"
echo "   Samples: $NUM_SAMPLES"
echo "   Script: eval_sae_interp.py"
echo "----------------------------------------"

$PYCMD -u Sae_interpt.py \
    --model "$MODEL_NAME" \
    --sae_layer $SAE_LAYER \
    --n_clusters $N_CLUSTERS \
    --num_samples $NUM_SAMPLES \
    --top_k_examples $TOP_K

echo "----------------------------------------"
echo "✅ SAE Interpretability Analysis Finished at $(date)"