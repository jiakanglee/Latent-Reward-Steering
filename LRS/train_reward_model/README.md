# Train reward model

- **入口**：`train_latent_classifier_7B.py` — Transformer reward model。
- **AIME 示例**：`run_train_aime_classifier.slurm`（仓库根目录 `sbatch train_reward_model/run_train_aime_classifier.slurm`）。读入 `collected_sae_latents_10dim_4000_aime24_aime25.pt`，写出 `transformer_reward_model_aime_best.pt`。
