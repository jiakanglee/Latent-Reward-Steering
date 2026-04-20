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

Run everything from the **repository root** (`PYTHONPATH` should include the repo root; Slurm scripts `cd` there automatically). The pipeline is:

**collect SAE latents → train Transformer reward model → run iterative steering.**

### 1. Collect data

Roll out the model with the SAE pipeline and save latent traces (`.pt`) for supervised reward training.

```bash
python collect_data/generate_data_7B.py \
  --dataset aime24_aime25 \
  --num_examples 200 \
  --max_token 4000 \
  --sae_layer 20 \
  --n_clusters 10
```

Cluster jobs (same defaults, override with env vars in the script):

```bash
sbatch collect_data/generate_data_7B.slurm
```

See `collect_data/README.md` for details and additional Slurm helpers.

### 2. Train reward model

Train the Transformer classifier on the collected tensors (adjust `--data_file` to match the file produced above).

```bash
python train_reward_model/train_latent_classifier_7B.py \
  --data_file collected_sae_latents_10dim_2000.pt \
  --save_path transformer_reward_model.pt \
  --epochs 50 \
  --hidden_dim 128
```

```bash
sbatch train_reward_model/train_latent_classifier_7B.slurm
```

See `train_reward_model/README.md`.

### 3. Main steering experiment

Run iterative latent steering with a trained reward checkpoint and cluster SAE config aligned with collection.

```bash
python steering/run_basic_overwrite.py \
  --model Open-Reasoner-Zero/Open-Reasoner-Zero-7B \
  --sae_layer 20 \
  --n_clusters 10 \
  --reward_model_path transformer_reward_model.pt \
  --step_size 2.0 \
  --num_steps 5 \
  --max_token 2000
```

```bash
sbatch steering/run_gpqa_diamond_aime_rm_mcq_collect_ilab2.slurm   # example: GPQA + MCQ sweep
```

See `steering/README.md` and the `steering/*.slurm` scripts for cluster-specific settings.

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
