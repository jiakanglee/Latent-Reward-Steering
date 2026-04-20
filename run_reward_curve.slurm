#!/bin/bash
#SBATCH --job-name=Reward_Curve
#SBATCH --output=log2/reward_curve_%j.out
#SBATCH --error=log2/reward_curve_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=rlab1
#SBATCH --gres=gpu:4500_ada:3

# 24 边例题目：绘制 Base vs Steer 的 reward model 曲线（每10 token mean）

export PYTHONNOUSERSITE=1
source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p log2
export PYTHONPATH=$PWD:$PYTHONPATH

PYCMD="python"
[ -x "$PWD/.venv/bin/python3" ] && PYCMD="$PWD/.venv/bin/python3"

EDGE_IDS="21 25 46 82 94 147 152 153 156 189 219 233 284 285 292 295 320 349 355 359 408 412 483 494"
OUT_DIR="log2/reward_curves"

$PYCMD -u run_reward_curve.py \
  --example_idx $EDGE_IDS \
  --output_dir "$OUT_DIR" \
  --edge_json logs/edge_cases_24.json

echo "Done. Plots saved to $OUT_DIR"
