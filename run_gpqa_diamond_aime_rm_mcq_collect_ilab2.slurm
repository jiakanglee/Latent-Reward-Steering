#!/bin/bash
# GPQA-Diamond：与 run_gpqa_diamond_aime_rm.slurm 同 setting（AIME RM、t4000、ns=4、reward=0.9、conf=0.72、step=1.15）。
# 判题：run_basic_overwrite 中 gpqa → evaluate_gpqa（extract_choice_letter），非 LLM judge。
# 收集：judge_reasons/*.json 含 gold_letter、extracted_letter_base/steer；可选全文见 SAVE_JUDGE_WITH_TEXT。
# 结束后 collect_gpqa_mcqa_summary.py 写 gpqa_mcqa_summary.csv。
#
#SBATCH --job-name=GPQA_MCQ_AIME
#SBATCH --output=log2/gpqa_diamond_aime_mcq_%j.out
#SBATCH --error=log2/gpqa_diamond_aime_mcq_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=14G
#SBATCH --time=72:00:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=rlab6
#SBATCH --gres=gpu:a5000:2

export PYTHONNOUSERSITE=1
source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p log2
export PYTHONPATH=$PWD:$PYTHONPATH

if [ -x "$PWD/.venv/bin/python3" ]; then PYCMD="$PWD/.venv/bin/python3"; else PYCMD="python"; fi

REWARD_MODEL="transformer_reward_model_aime_best.pt"
REWARD=0.9
MAX_TOKEN=4000
NSTEPS=4
CONF=0.72
STEP=1.15

LOG_DIR="log2/gpqa_diamond_aime_mcq_${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR/judge_reasons"

# 设为 1 则在 JSON 中写入完整 response_base/response_steer（体积大）
SAVE_JUDGE_WITH_TEXT="${SAVE_JUDGE_WITH_TEXT:-0}"

NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then NUM_GPUS=4; fi

echo "GPQA-Diamond | rule MCQ extract | AIME RM | t=${MAX_TOKEN} n=${NSTEPS} r=${REWARD} c=${CONF} step=${STEP}"
echo "LOG_DIR=$LOG_DIR  SAVE_JUDGE_WITH_TEXT=$SAVE_JUDGE_WITH_TEXT"
echo "Date: $(date)"

EXTRA=()
if [ "$SAVE_JUDGE_WITH_TEXT" = "1" ]; then
  EXTRA+=(--save_judge_with_text)
fi

pids=()
for ((i=0; i<NUM_GPUS; i++)); do
  CUDA_VISIBLE_DEVICES=$i $PYCMD -u run_basic_overwrite.py \
    --model "Open-Reasoner-Zero/Open-Reasoner-Zero-7B" \
    --sae_layer 20 \
    --n_clusters 10 \
    --reward_model_path "$REWARD_MODEL" \
    --step_size "$STEP" \
    --num_steps "$NSTEPS" \
    --max_token "$MAX_TOKEN" \
    --reward_threshold "$REWARD" \
    --confidence_threshold "$CONF" \
    --dataset gpqa_diamond \
    --num_examples 198 \
    --no-monitor \
    --save_judge_reason \
    --judge_reason_dir "${LOG_DIR}/judge_reasons" \
    "${EXTRA[@]}" \
    --shard_id $i \
    --num_shards $NUM_GPUS \
    > "${LOG_DIR}/log_worker_${i}.txt" 2>&1 &
  pids+=($!)
done

for pid in "${pids[@]}"; do wait "$pid"; done

RESULTS_FILE="log2/gpqa_diamond_aime_mcq_results_${SLURM_JOB_ID}.txt"
echo "=== GPQA-Diamond MCQ extract + AIME RM: t=$MAX_TOKEN n=$NSTEPS reward=$REWARD conf=$CONF step=$STEP (198 questions) ===" > "$RESULTS_FILE"
$PYCMD collect_sweep_results.py \
  --log_dir "$LOG_DIR" \
  --output "$RESULTS_FILE" \
  --label "t${MAX_TOKEN}_s${STEP}_c${CONF}_n${NSTEPS}_mcq"

$PYCMD collect_gpqa_mcqa_summary.py "$LOG_DIR"

echo ""
cat "$RESULTS_FILE"
echo "---"
echo "CSV: ${LOG_DIR}/gpqa_mcqa_summary.csv"
echo "Date: $(date)"
