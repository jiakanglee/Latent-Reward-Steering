#!/usr/bin/env bash
# 将复现 steering 实验所需代码 + log2 中「可公开的小结果」打到一个目录，便于 git / 发版。
# 用法:
#   ./scripts/export_github_bundle.sh [目标目录，默认 ../thinking-llms-interp-release]
#
# 环境变量:
#   WITH_WEIGHTS=1   同时复制 AIME RM 与 ORZ-7B 用 SAE 权重（几 MB，可进 git；多模型 SAE 仍可能很大）
#   WITH_LOG2=1      复制 log2 中「轻量结果」；=0 则不要 log2
#
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-${REPO_ROOT}/../thinking-llms-interp-release}"
OUT="$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")"

WITH_LOG2="${WITH_LOG2:-1}"
WITH_WEIGHTS="${WITH_WEIGHTS:-0}"

mkdir -p "$OUT"

echo "==> REPO_ROOT=$REPO_ROOT"
echo "==> OUT=$OUT"

rsync -a --relative \
  "$REPO_ROOT/./utils" \
  "$REPO_ROOT/./run_basic_overwrite.py" \
  "$REPO_ROOT/./collect_sweep_results.py" \
  "$REPO_ROOT/./collect_gpqa_mcqa_summary.py" \
  "$REPO_ROOT/./run_math500_baseline_official.py" \
  "$REPO_ROOT/./run_gpqa_diamond_baseline_official.py" \
  "$REPO_ROOT/./run_reward_curve.py" \
  "$REPO_ROOT/./train_latent_classifier_7B.py" \
  "$REPO_ROOT/./generate_data_7B.py" \
  "$REPO_ROOT/./merge_collect_shards.py" \
  "$REPO_ROOT"/./export_math500_*.py \
  "$REPO_ROOT"/./aggregate_quad_steer_cells_for_sweep.py \
  "$REPO_ROOT"/./summarize_math500_*.py \
  "$REPO_ROOT"/./analyze_math500_*.py \
  "$REPO_ROOT/./log2/slurms" \
  "$OUT/"
shopt -s nullglob
runs=( "$REPO_ROOT"/run_*.sh )
shopt -u nullglob
if (( ${#runs[@]} )); then cp -a "${runs[@]}" "$OUT/"; fi

# 常一起用的分析脚本（根目录单文件）
shopt -s nullglob
for f in "$REPO_ROOT"/*.py; do
  b=$(basename "$f")
  case "$b" in
    run_basic_overwrite.py|collect_sweep_results.py) continue ;;
  esac
  if [[ "$b" == export_* ]] || [[ "$b" == plot_* ]] || [[ "$b" == aggregate_* ]] || [[ "$b" == summarize_* ]]; then
    cp -a "$f" "$OUT/" 2>/dev/null || true
  fi
done
shopt -u nullglob

if [[ "$WITH_WEIGHTS" == "1" ]]; then
  mkdir -p "$OUT/train-saes/results/vars/saes"
  if [[ -f "$REPO_ROOT/transformer_reward_model_aime_best.pt" ]]; then
    cp -a "$REPO_ROOT/transformer_reward_model_aime_best.pt" "$OUT/"
  fi
  for f in "$REPO_ROOT"/train-saes/results/vars/saes/sae_open-reasoner-zero-7b_layer20_clusters10.pt; do
    if [[ -f "$f" ]]; then
      cp -a "$f" "$OUT/train-saes/results/vars/saes/"
    fi
  done
  echo "    (已 WITH_WEIGHTS: 复制了 ORZ-7B 用 SAE 与 AIME RM，若需其他 SAE 请手拷 train-saes/results/vars/saes/)"
fi

MAN="${OUT}/BUNDLE_MANIFEST.txt"
{
  echo "thinking-llms-interp 发布包说明"
  echo "生成时间: $(date -Iseconds)"
  echo ""
  echo "一、跑分主链（与 log2 中 sweep 一致）"
  echo "  - run_basic_overwrite.py"
  echo "  - utils/ (load_model, SAE, llm_judge, rule_eval 等)"
  echo "  - collect_sweep_results.py, collect_gpqa_mcqa_summary.py"
  echo "  - log2/slurms/*.slurm（收录的 Slurm）+ HuggingFace 数据、本机 Python/conda"
  echo ""
  echo "二、不随包提供、需自行准备"
  echo "  - 7B 主模型由 HuggingFace 在运行时拉取 (Open-Reasoner-Zero-7B)"
  echo "  - 若未开 WITH_WEIGHTS: train-saes/.../sae_open-reasoner-zero-7b_layer20_clusters10.pt 与 transformer_reward_model_aime_best.pt"
  echo "  - 其他子项目: Vicky/, PST/, hybrid/ 为历史/平行实验，主 sweep 一般不用"
  echo ""
} > "$MAN"

if [[ "$WITH_LOG2" == "1" ]]; then
  mkdir -p "$OUT/log2"
  # 仅复制「轻量」结果文件；不包含 log_worker_*.txt / judge/*.json（体积大）
  if [[ -d "$REPO_ROOT/log2" ]]; then
    rsync -a --prune-empty-dirs \
      --include='*/' \
      --include='**/sweep_collect_append.txt' \
      --include='**/run_meta.json' \
      --include='**/final_report_grep.txt' \
      --include='**/*.png' \
      --include='**/gpqa_mcqa_summary.csv' \
      --include='**/*results*.txt' \
      --exclude='*' \
      "$REPO_ROOT/log2/" "$OUT/log2/"
    if [[ -d "$REPO_ROOT/log2/slurm_sweep_min" ]]; then
      mkdir -p "$OUT/log2/slurm_sweep_min"
      find "$REPO_ROOT/log2/slurm_sweep_min" -maxdepth 1 -type f -size -512k -exec cp -a {} "$OUT/log2/slurm_sweep_min/" \; 2>/dev/null || true
    fi
  fi
  {
    echo "三、log2 收入本包规则"
    echo "  已尽量只含: sweep_collect_append, run_meta, final_report_grep, png, *results*, gpqa csv, 小 slurm 日志"
    echo "  未含: log_worker_*.txt, judge 大 JSON（请用 collect 脚本在本地重算或另存网盘）"
  } >> "$MAN"
fi

# 粗计大小
echo "==> 包大小:"; du -sh "$OUT"
echo "==> 清单: $MAN"
cat "$MAN"
echo "完成: $OUT"
