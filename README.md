# Latent Reward Steering (LRS)

**Latent Reward Steering: Adaptive Inference-Time Framework that Implicitly Promotes Cognitive Behaviors in Reasoning LLMs**

> Anonymous ACL submission

---

## Overview

![LRS Framework](method-1.png)

Strong reasoning in LLMs depends not only on model knowledge but also on **when and how cognitive behaviors are deployed** during generation. Existing methods for cognitive-behavior control — whether prompt-based or representation-level steering — commit to predefined behaviors and fixed intervention directions that may not match the local reasoning state.

**Latent Reward Steering (LRS)** takes a different approach: instead of specifying *which* cognitive behaviors to inject, LRS trains a **latent reward model** from successful and unsuccessful reasoning traces to estimate the quality of intermediate SAE latent states. During inference, reward gradients provide state-specific correction directions, while a **reward–confidence gate** restricts intervention to steps flagged as unreliable — leaving healthy reasoning steps untouched.

---

## Main Results

All results are reported under **greedy decoding, zero-shot**. Values in parentheses denote absolute gains of LRS BASIC / LRS over the Base model. LRS BASIC is ungated; full LRS uses reward–confidence gating.

### Open-Reasoner-7B

| Dataset | Base | LRS BASIC | LRS | CoT | Few-shot （5-shot) |
|---------|------|-----------|-----|-----|----------|
| MATH-500 | 79.4 | 83.0 (+3.6) | **83.8 (+4.4)** | 81.4 | 81.0 |
| AIME 2024 | 16.6 | 16.6 (+0.0) | **26.6 (+10.0)** | 13.3 | 13.3 |
| AIME 2025 | 16.6 | 20.0 (+3.4) | **26.6 (+10.0)** | 13.3 | 10.0 |
| GPQA-Diamond | 32.3 | 30.8 (−1.5) | **39.4 (+7.1)** | 35.9 | 38.4 |
| AMC23 | 50.0 | 45.0 (−5.0) | 60.0 (+10.0) | 55.0 | **65.0** |
| IneqMath | 46.0 | 52.0 (+6.0) | **60.0 (+14.0)** | 48.0 | 48.0 |

### Open-Reasoner-1.5B

| Dataset | Base | LRS BASIC | LRS | CoT | Few-shot |
|---------|------|-----------|-----|-----|----------|
| MATH-500 | 59.2 | 59.0 (−0.2) | **60.8 (+1.6)** | 58.6 | 57.2 |
| AIME 2024 | 3.3 | 0.0 (−3.3) | **13.3 (+10.0)** | 6.7 | 6.7 |
| AIME 2025 | 3.3 | 0.0 (−3.3) | **6.6 (+3.3)** | 0.0 | 3.3 |
| GPQA-Diamond | 18.2 | 15.7 (−2.5) | **22.8 (+4.6)** | 17.2 | 17.2 |
| AMC23 | 30.0 | 32.5 (+2.5) | **37.5 (+7.5)** | 32.5 | 30.0 |
| IneqMath | 29.0 | 28.0 (−1.0) | **34.0 (+5.0)** | 30.0 | 28.0 |

LRS improves over standard decoding on **all six benchmarks** for both model sizes without changing model weights. The ungated variant LRS BASIC is less stable, confirming that selective intervention via the reward–confidence gate is essential.

---

## Method

LRS operates in three stages:

1. **Latent Trace Construction** — Run a frozen reasoning LLM + pretrained SAE to collect sparse latent sequences, labeled by final-answer correctness.
2. **Latent Reward Learning** — Train a lightweight Transformer reward model on the latent traces to predict trajectory quality without any behavior annotations.
3. **Online Selective Latent Repair** — At inference time, a reward–confidence gate identifies fragile latent states; reward gradients then optimize those states via normalized gradient ascent, and the latent difference is decoded back into the hidden activation as a residual correction.

The **reward–confidence gate** triggers steering when:
- The reward score `r_t < τ_r` (low-quality state), **or**
- `r_t ≥ τ_r` but the previous-token confidence `c_{t-1} < τ_c` (borderline quality, uncertain next step)

Default thresholds: `τ_r = 0.9`, `τ_c = 0.72`. Steering is applied at SAE layer 20 in a 10-dimensional latent space.

---

## Setup

### Requirements

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (`pip install uv` or see the uv docs)
- CUDA-compatible GPU (experiments run on RTX A4500 / A5000 / A6000)

### Install

```bash
git clone https://github.com/jiakanglee/Latent-Reward-Steering.git
cd Latent-Reward-Steering
uv sync
```

`uv sync` reads `pyproject.toml` and `uv.lock` to install all dependencies into a local `.venv`. To run scripts, either activate the environment or prefix with `uv run`:

```bash
# Option 1: activate
source .venv/bin/activate
python LRS/Steering/run_basic_overwrite.py ...

# Option 2: uv run (no activation needed)
uv run python LRS/Steering/run_basic_overwrite.py ...
```

---

## Workflow (Latent Reward Steering Pipeline)

Run everything from the **repository root**. Slurm scripts assume that directory, set `PYTHONPATH`, and write logs under `log2/`.

All LRS scripts live under `LRS/`:

```
LRS/
├── collect_data/          # Stage 1: collect SAE latent traces
├── train_reward_model/    # Stage 2: train the latent reward model
└── Steering/              # Stage 3: run inference-time steering
```

### Stage 1 — Collect Latent Traces (AIME 2024 + 2025)

```bash
python LRS/collect_data/generate_data_7B.py \
  --dataset aime24_aime25 \
  --max_token 4000 \
  --output_file collected_sae_latents_10dim_4000_aime24_aime25.pt
```

Cluster (Slurm):

```bash
sbatch LRS/collect_data/run_collect_aime.slurm
```

Output: `collected_sae_latents_10dim_4000_aime24_aime25.pt`

### Stage 2 — Train the Latent Reward Model

```bash
python LRS/train_reward_model/train_latent_classifier_7B.py \
  --data_file collected_sae_latents_10dim_4000_aime24_aime25.pt \
  --save_path transformer_reward_model_aime.pt \
  --epochs 30 \
  --lr 0.0005 \
  --hidden_dim 128
```

Cluster (Slurm):

```bash
sbatch LRS/train_reward_model/run_train_aime_classifier.slurm
```

Output: `transformer_reward_model_aime_best.pt` (best checkpoint saved next to `--save_path`)

### Stage 3 — Run Inference-Time Steering

Place `transformer_reward_model_aime_best.pt` in the repo root, then:

**AIME 2024:**
```bash
sbatch LRS/Steering/run_aime24.slurm
```

**AIME 2025:**
```bash
sbatch LRS/Steering/run_aime25.slurm
```

**Manual invocation** (same hyperparameters as `run_aime24.slurm`):

```bash
python LRS/Steering/run_basic_overwrite.py \
  --model Open-Reasoner-Zero/Open-Reasoner-Zero-7B \
  --sae_layer 20 --n_clusters 10 \
  --reward_model_path transformer_reward_model_aime_best.pt \
  --step_size 1.15 --num_steps 4 --max_token 4000 \
  --reward_threshold 0.9 --confidence_threshold 0.72 \
  --dataset aime24 --num_examples 30 \
  --print_response --save_judge_reason
```

Large-scale hyperparameter sweeps: `LRS/Steering/run_aime24_steer_sweep_ilab2.slurm`, `run_aime25_steer_sweep_ilab2.slurm`, etc.

### Hyperparameter Configurations

| Dataset | Model | K | α | Reward τ | Confidence τ | Device |
|---------|-------|---|---|----------|--------------|--------|
| MATH-500 | ORZ-7B | 1 | 1.400 | 0.8 | 0.69 | RTX A4500 |
| AIME24 | ORZ-7B | 2 | 0.295 | 0.9 | 0.72 | RTX A4500 |
| AIME25 | ORZ-7B | 3 | 1.320 | 0.9 | 0.72 | RTX A4500 |
| GPQA-Diamond | ORZ-7B | 4 | 1.150 | 0.9 | 0.72 | RTX A5000 |
| AMC23 | ORZ-7B | 1 | 1.400 | 0.9 | 0.72 | RTX A4500 |
| IneqMath | ORZ-7B | 2 | 0.700 | 0.9 | 0.72 | RTX A5000 |
| MATH-500 | ORZ-1.5B | 1 | 0.100 | 0.9 | 0.72 | RTX A6000 |
| AIME24 | ORZ-1.5B | 4 | 0.900 | 0.9 | 0.72 | RTX A4500 |
| AIME25 | ORZ-1.5B | 2 | 0.400 | 0.9 | 0.72 | RTX A6000 |
| GPQA-Diamond | ORZ-1.5B | 3 | 1.000 | 0.9 | 0.72 | RTX A5000 |
| AMC23 | ORZ-1.5B | 4 | 0.300 | 0.9 | 0.72 | RTX A6000 |
| IneqMath | ORZ-1.5B | 2 | 1.100 | 0.9 | 0.72 | RTX A6000 |

*K = number of latent optimization steps; α = step size.*

---

## Repository Structure

```
.
├── LRS/                          # Main LRS pipeline (this paper)
│   ├── collect_data/             # Latent trace collection
│   ├── train_reward_model/       # Reward model training
│   ├── Steering/                 # Inference-time steering scripts
│   └── Interpretability/         # SAE interpretability analysis
├── hybrid/                       # Hybrid model experiments
├── train-saes/                   # SAE training
├── train-vectors/                # Steering vector training
├── visualize-saes/               # SAE visualization
├── utils/                        # Shared utilities
├── method.pdf                    # Framework diagram
├── pyproject.toml
└── environment.yaml
```

---

## Inference Efficiency

LRS incurs a modest overhead with no change to model weights:

| Metric | Base | LRS |
|--------|------|-----|
| Avg. generated tokens | 2595 | 2596 |
| Avg. wall-clock / problem (s) | 115.3 | 156.2 |
| Slowdown ratio | 1.00× | **1.35×** |
| Steered tokens (%) | — | 27.9% |
| Avg. steering triggers / problem | — | 725 |

The reward–confidence gate skips **72.1%** of tokens, focusing intervention only on fragile states.

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{lrs2025,
  title  = {Latent Reward Steering: Adaptive Inference-Time Framework that
             Implicitly Promotes Cognitive Behaviors in Reasoning LLMs},
  author = {Anonymous},
  year   = {2025},
  note   = {ACL submission}
}
```

---

## Acknowledgements

This project builds on the interpretability codebase from:

> Constantin Venhoff, Iván Arcuschin, Philip Torr, Arthur Conmy, and Neel Nanda.
> *Base Models Know How to Reason, Thinking Models Learn When.*
> arXiv:2510.07364, 2025.
