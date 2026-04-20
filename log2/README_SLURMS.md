# log2 目录说明

- **默认**：`log2/` 下除本说明与 **`slurms/`** 外的内容均被 `.gitignore` 忽略（跑分日志、CSV、超大 JSON 等）。
- **`slurms/`**：收录与主流程一致的 Slurm 作业模板（产出写入本仓库下的 `log2/`）。
- **提交作业**：建议在仓库根目录执行  
  `sbatch log2/slurms/<脚本>.slurm`  
  脚本开头会 `cd` 到仓库根，以便 `#SBATCH --output=log2/...` 路径正确。

主代码入口（根目录）：`generate_data_7B.py`、`train_latent_classifier_7B.py`、`run_basic_overwrite.py`。
