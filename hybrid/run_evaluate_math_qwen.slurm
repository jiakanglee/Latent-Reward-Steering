#!/bin/bash
#SBATCH --job-name=Eval_Math_Qwen_A100
#SBATCH --output=log2/evaluate_math_qwen_%j.out
#SBATCH --error=log2/evaluate_math_qwen_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=rlab6
#SBATCH --gres=gpu:a5000:4

export PYTHONNOUSERSITE=1
source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

# Ensure HF token env aliases
if [ -z "$HUGGINGFACE_HUB_TOKEN" ]; then
  if [ -f "$HOME/.huggingface/token" ]; then
    export HUGGINGFACE_HUB_TOKEN=$(cat "$HOME/.huggingface/token")
  fi
fi
if [ -n "$HUGGINGFACE_HUB_TOKEN" ]; then
  export HF_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN"
  export HUGGINGFACE_TOKEN="$HUGGINGFACE_HUB_TOKEN"
  export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
fi

echo "Job Node: $(hostname)"

# 进入项目根目录
cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p log2
export PYTHONPATH=$PWD:$PYTHONPATH

# 定义 Python 执行命令 (自动处理 uv 或 venv)
PROJECT_VENV_PY="$PWD/.venv/bin/python3"
if [ -x "$PROJECT_VENV_PY" ]; then
  PYCMD="$PROJECT_VENV_PY -u hybrid/evaluate_hybrid.py"
else
  UV_BIN=$(which uv 2>/dev/null || echo "/common/home/jl3614/miniconda/envs/stllms_env/bin/uv")
  PYCMD="${UV_BIN} run -- python -u hybrid/evaluate_hybrid.py"
fi

# 设置你要测试的模型和参数
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
BASE_MODEL="Qwen/Qwen2.5-Math-1.5B"
PROBE_LAYER=10 # 确保这与你 probe .pt 文件名的层数一致

# Run Combined Steering Verification
echo "Starting Combined Steering (Top-K=3) Verification..."

$PYCMD \
    --model "$MODEL" \
    --base_model "$BASE_MODEL" \
    --probe_layer $PROBE_LAYER \
    --top_k 3 \
    --n_batches 1 \
    --batch_size 1 \
    --max_new_tokens 300 \
    --seed 42

echo "Verification finished."