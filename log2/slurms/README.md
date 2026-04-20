# 收录的 Slurm（写入 `log2/`）

| 脚本 | 用途 |
|------|------|
| `generate_data_7B.slurm` | 采集轨迹 → `.pt`（默认文件名见脚本内注释） |
| `train_latent_classifier_7B.slurm` | 训练 Transformer reward model |
| `run_basic_overwrite.slurm` | 迭代 steering 主实验（`run_basic_overwrite.py`） |
| `run_gpqa_diamond_aime_rm_mcq_collect_ilab2.slurm` | GPQA-Diamond + AIME RM + MCQ 规则判题汇总 |
| `run_dump_gpqa_decode_sample.slurm` | 单题导出 GPQA 解码文本（调试 extract） |

环境：设置 `CONDA_INIT_SH`（可选）指向 `conda.sh`，或依赖默认的 `~/miniconda3`、`~/anaconda3`。`CONDA_ENV` 默认 `stllms_env`。
