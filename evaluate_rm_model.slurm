#!/bin/bash
#SBATCH --job-name=Eval_1300
#SBATCH --output=log2/eval_2000_%j.out
#SBATCH --error=log2/eval_2000_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=rlab3
#SBATCH --gres=gpu:a4000:1

# === 1. 环境配置 ===
export PYTHONNOUSERSITE=1
source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p log2
export PYTHONPATH=$PWD:$PYTHONPATH

# 检查 Python
if [ -x "$PWD/.venv/bin/python3" ]; then PYCMD="$PWD/.venv/bin/python3"; else PYCMD="python"; fi

echo "🚀 Starting Evaluation on 1300 Token Dataset..."
echo "Date: $(date)"

# === 2. 核心参数 ===
# 你的新测试集
TEST_DATA="collected_sae_latents_10dim_2000.pt"

# 你刚才训练好的模型 (注意文件名要对上，如果你刚才保存的是 lstm_reward_model.pth 就用这个)
MODEL_PATH="lstm_reward_model.pth"

# === 3. 运行评测 ===
$PYCMD -u evaluate_rm_model.py \
    --test_data "$TEST_DATA" \
    --model_path "$MODEL_PATH"

echo "✅ Evaluation Finished at $(date)"