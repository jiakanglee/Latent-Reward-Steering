# Base Models Know How to Reason, Thinking Models Learn When

Code for the paper [Base Models Know How to Reason, Thinking Models Learn When](https://arxiv.org/abs/2510.07364).

**Website:** [thinking-llms-interp.com](https://thinking-llms-interp.com/)

## Setup

### Requirements

- Python 3.10+
- `uv` installed (`pip install uv` or see the [uv docs](https://docs.astral.sh/uv/getting-started/installation/))

### Install

```bash
git clone https://github.com/cvenhoff/cot-interp.git
cd cot-interp
uv sync
```

## Workflow (latent reward steering)

Run everything from the **repository root**. Slurm scripts assume that directory, set `PYTHONPATH`, and write under `log2/`.

End-to-end **AIME** pipeline (matches the checked-in Slurm under `collect_data/`, `train_reward_model/`, `steering/`):

| Stage | Artifact |
|-------|----------|
| Collect | `collected_sae_latents_10dim_4000_aime24_aime25.pt` |
| Train RM | `transformer_reward_model_aime_best.pt` (written next to `--save_path` with `_best` suffix) |
| Steer | Uses `transformer_reward_model_aime_best.pt`; SAE layer **20**, clusters **10**, ORZ-7B |

Defaults in the scripts: **`aime24_aime25`** rollouts, **`max_token=4000`**, training **`epochs=30`**, **`lr=5e-4`**. Steering jobs (e.g. `run_aime24.slurm`) use **`max_token=4000`**, **`num_steps=4`**, **`reward_threshold=0.9`**, **`confidence_threshold=0.72`**, **`step_size=1.15`**.

### 1. Collect data (AIME 2024 + AIME 2025 latents)

Same call as `collect_data/run_collect_aime.slurm`:

```bash
python collect_data/generate_data_7B.py \
  --dataset aime24_aime25 \
  --max_token 4000 \
  --output_file collected_sae_latents_10dim_4000_aime24_aime25.pt
```

Cluster:

```bash
sbatch collect_data/run_collect_aime.slurm
```

(Optional: `--load_in_8bit` if you hit OOM on a small GPU.)

### 2. Train reward model (on that `.pt`)

Same as `train_reward_model/run_train_aime_classifier.slurm`:

```bash
python train_reward_model/train_latent_classifier_7B.py \
  --data_file collected_sae_latents_10dim_4000_aime24_aime25.pt \
  --save_path transformer_reward_model_aime.pt \
  --epochs 30 \
  --lr 0.0005 \
  --hidden_dim 128
```

This produces **`transformer_reward_model_aime_best.pt`** in the repo root.

```bash
sbatch train_reward_model/run_train_aime_classifier.slurm
```

### 3. Main experiment (AIME steering)

Place or symlink **`transformer_reward_model_aime_best.pt`** in the repo root (the Slurm below expects that filename). Then run the full benchmark, e.g. **AIME 2024** (30 problems, 2× GPU shard in the template):

```bash
sbatch steering/run_aime24.slurm
```

For **AIME 2025**, use `steering/run_aime25.slurm`. Larger hyperparameter sweeps live in `steering/run_aime24_steer_sweep_ilab2.slurm`, `steering/run_aime25_steer_sweep_ilab2.slurm`, etc.; all point at `transformer_reward_model_aime_best.pt`.

Minimal manual invocation (same knob bundle as `run_aime24.slurm`; adjust `--dataset` / `--num_examples` for AIME25):

```bash
python steering/run_basic_overwrite.py \
  --model Open-Reasoner-Zero/Open-Reasoner-Zero-7B \
  --sae_layer 20 --n_clusters 10 \
  --reward_model_path transformer_reward_model_aime_best.pt \
  --step_size 1.15 --num_steps 4 --max_token 4000 \
  --reward_threshold 0.9 --confidence_threshold 0.72 \
  --dataset aime24 --num_examples 30 \
  --print_response --save_judge_reason
```

See `collect_data/README.md`, `train_reward_model/README.md`, and `steering/README.md` for paths and extra Slurm jobs.

### Other code in this repo

The upstream cot-interp style experiments (SAE taxonomy, hybrid models, MMLU response generation, etc.) still live under `generate-responses/`, `train-saes/`, `train-vectors/`, `hybrid/`, etc. Use their `run.sh` / `uv run` entry points if you reproduce the original paper pipeline.

## Citation

If you find this work useful, please cite:

```bibtex
@misc{venhoff2025basemodelsknowreason,
      title={Base Models Know How to Reason, Thinking Models Learn When},
      author={Constantin Venhoff and Iván Arcuschin and Philip Torr and Arthur Conmy and Neel Nanda},
      year={2025},
      eprint={2510.07364},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.07364},
}
```
