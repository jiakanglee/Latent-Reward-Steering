#!/bin/bash
#SBATCH --job-name=Sweep_Step
#SBATCH --output=PST/logs/sweep_step_aime24_%j.out
#SBATCH --error=PST/logs/sweep_step_aime24_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=ilab2
#SBATCH --gres=gpu:a4500:4

export PYTHONNOUSERSITE=1
source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p PST/logs
export PYTHONPATH=$PWD:$PYTHONPATH

if [ -x "$PWD/.venv/bin/python3" ]; then PYCMD="$PWD/.venv/bin/python3"; else PYCMD="python"; fi

echo "Sweep STEP=0.5~1.2 | CONF=0.72 | REWARD=0.9 | NSTEPS=4 | aime24 30题"
echo "Date: $(date)"

REWARD_MODEL="PST/transformer_reward_model_aime_1.5B_best.pt"
REWARD=0.9
MAX_TOKEN=4000
NSTEPS=4
CONF=0.72

LOG_BASE="PST/logs/sweep_step_aime24_${SLURM_JOB_ID}"
mkdir -p "$LOG_BASE"

STEPS=(0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2)

# 分2轮，每轮4个 STEP 各占1块 GPU
for round in 0 1; do
    pids=()
    for gpu in 0 1 2 3; do
        idx=$((round * 4 + gpu))
        STEP=${STEPS[$idx]}
        LOG_DIR="${LOG_BASE}/step_${STEP}"
        mkdir -p "${LOG_DIR}/judge_reasons"

        echo "=== Round $((round+1)) | GPU $gpu | STEP=$STEP ==="
        CUDA_VISIBLE_DEVICES=$gpu $PYCMD -u PST/run_basic_overwrite.py \
            --model "Open-Reasoner-Zero/Open-Reasoner-Zero-1.5B" \
            --sae_layer 8 \
            --n_clusters 5 \
            --reward_model_path "$REWARD_MODEL" \
            --step_size $STEP \
            --num_steps $NSTEPS \
            --max_token $MAX_TOKEN \
            --reward_threshold $REWARD \
            --confidence_threshold $CONF \
            --dataset aime24 \
            --num_examples 30 \
            --print_response \
            --save_judge_reason \
            --judge_reason_dir "${LOG_DIR}/judge_reasons" \
            --shard_id 0 \
            --num_shards 1 \
            > "${LOG_DIR}/log.txt" 2>&1 &
        pids+=($!)
    done

    echo "Waiting for round $((round+1)) (STEP=${STEPS[round*4]}~${STEPS[round*4+3]}) ..."
    for pid in "${pids[@]}"; do wait $pid; done
    echo "Round $((round+1)) done at $(date)"
    echo ""
done

# 汇总所有 STEP 的结果
RESULTS_FILE="PST/logs/sweep_step_aime24_results_${SLURM_JOB_ID}.txt"
echo "=== Sweep STEP aime24: CONF=$CONF REWARD=$REWARD NSTEPS=$NSTEPS ===" > "$RESULTS_FILE"
echo "STEP  | total | Base对 | Steer对 | Base对Steer错 | Base错Steer对" >> "$RESULTS_FILE"

for STEP in "${STEPS[@]}"; do
    LOG_DIR="${LOG_BASE}/step_${STEP}"
    $PYCMD collect_sweep_results.py \
        --log_dir "$LOG_DIR" \
        --output "$RESULTS_FILE" \
        --label "step_${STEP}"
done

echo ""
echo "============================================================"
echo "Sweep 结果汇总 (STEP 0.5~1.2)"
echo "============================================================"
cat "$RESULTS_FILE"
echo "============================================================"
echo "Date: $(date)"
