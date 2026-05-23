# Steering（主实验）

- **入口**：`run_basic_overwrite.py` — 迭代 latent steering。
- **AIME 一条龙**：先在根目录备好 `transformer_reward_model_aime_best.pt`，再  
  `sbatch steering/run_aime24.slurm` 或 `sbatch steering/run_aime25.slurm`。  
  大规模扫参见 `run_aime24_steer_sweep_ilab2.slurm`、`run_aime25_steer_sweep_ilab2.slurm` 等。

仓库根目录另有历史 `run_*.slurm` 副本时，以本目录版本为准。
