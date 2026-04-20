# Clustering Training and Evaluation Workflow

This folder contains scripts for training clustering models and evaluating them using OpenAI's batch API for cost-effective large-scale evaluation.

## Overview

The workflow has been split into three main stages:
1. **Training**: Train clustering models without evaluation
2. **Title Generation**: Generate cluster descriptions/titles using batch API
3. **Evaluation**: Evaluate clustering performance using batch API

## Scripts

### 1. `train_clustering.py`
Trains clustering models for different cluster sizes without performing evaluation. Also creates empty JSON result files with the correct structure.

**Purpose**: Fit clustering models, save them for later evaluation, and create JSON structure files needed by subsequent scripts.

### 2. `generate_titles_trained_clustering.py`
Generates multiple sets of human-readable titles and descriptions for clusters using OpenAI's batch API.

**Purpose**: Create semantic descriptions of what each cluster represents. Generates multiple different category sets (repetitions) to enable robust evaluation with different cluster interpretations.

### 3. `evaluate_trained_clustering.py`
Evaluates clustering performance (accuracy, completeness, semantic orthogonality) using OpenAI's batch API.

**Purpose**: Assess the quality of clustering results.

## Execution Order

### Step 1: Train Clustering Models
```bash
python train_clustering.py --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" --layer 12
```

**Options:**
- `--model`: Model to analyze (default: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
- `--layer`: Layer to analyze (default: 12)
- `--n_examples`: Number of examples to use (default: 500)
- `--clusters`: Comma-separated cluster sizes to train (default: "5,10,15,20,25,30,35,40,45,50")
- `--clustering_methods`: Methods to use (default: ["gmm", "pca_gmm", "spherical_kmeans", "pca_kmeans", "agglomerative", "pca_agglomerative", "sae_topk"])
- `--load_in_8bit`: Load model in 8-bit mode

### Step 2: Generate Cluster Titles

#### Submit title generation jobs:
```bash
python generate_titles_trained_clustering.py --command submit --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" --layer 12
```

#### Wait for batches to complete (check status):
```bash
python generate_titles_trained_clustering.py --command process --check_status
```

#### Process completed batches:
```bash
python generate_titles_trained_clustering.py --command process
```

**Options:**
- `--command`: Either "submit" or "process"
- `--evaluator_model`: Model for generating descriptions (default: "gpt-4o")
- `--description_examples`: Number of examples per cluster for descriptions (default: 200)
- `--repetitions`: Number of different category sets to generate (default: 3)
- `--check_status`: Check if all batches are completed before processing
- `--batch_file`: Specific batch file to process (optional)

### Step 3: Evaluate Clustering Performance

#### Submit evaluation jobs:
```bash
python evaluate_trained_clustering.py --command submit --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" --layer 12
```

#### Wait for batches to complete (check status):
```bash
python evaluate_trained_clustering.py --command process --check_status
```

#### Process completed batches:
```bash
python evaluate_trained_clustering.py --command process
```

**Options:**
- `--command`: Either "submit" or "process"
- `--evaluator_model`: Model for evaluations (default: "gpt-4o")
- `--n_autograder_examples`: Examples per cluster for accuracy testing (default: 100)
- `--repetitions`: Number of evaluation repetitions (default: 3)
- `--re_compute_cluster_labels`: Recompute cluster assignments
- `--check_status`: Check if all batches are completed before processing
- `--batch_file`: Specific batch file to process (optional)

## Batch Management

### Checking Batch Status
Both title generation and evaluation scripts support `--check_status` to verify all batches are completed before processing:

```bash
# Check title generation status
python generate_titles_trained_clustering.py --command process --check_status

# Check evaluation status  
python evaluate_trained_clustering.py --command process --check_status
```

### Batch Information Files
The scripts create JSON files to track batch submissions:
- `batch_info_titles_{model_id}_layer{layer}.json` - Title generation batches
- `batch_info_eval_{model_id}_layer{layer}.json` - Evaluation batches

### Processing Specific Batches
You can process specific batch files:
```bash
python generate_titles_trained_clustering.py --command process --batch_file custom_batch_file.json
python evaluate_trained_clustering.py --command process --batch_file custom_batch_file.json
```

## Output Files

### Training Results
- `results/vars/{method}_results_{model_id}_layer{layer}.json` - JSON files with empty structure for each cluster size (created by training step)
- Saved clustering models in `results/vars/{method}/` directories

### Final Results
After all steps, the same JSON files contain:
- Cluster descriptions/titles (from step 2)
- Evaluation metrics including accuracy, completeness, and semantic orthogonality (from step 3)
- Statistical analysis across multiple repetitions

## Cost Considerations

Using OpenAI's batch API provides:
- 50% cost reduction compared to standard API
- Efficient processing of large numbers of prompts
- 24-hour completion window for batch jobs

The batch approach is especially beneficial for:
- Multiple clustering methods
- Multiple cluster sizes
- Multiple evaluation repetitions
- Large numbers of examples

## Error Handling

- Scripts handle batch failures gracefully
- Individual batch errors don't stop the entire process
- Status checking prevents processing incomplete batches
- Detailed error logging for debugging

## Example Complete Workflow

```bash
# 1. Train all clustering models
python train_clustering.py --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" --layer 12

# 2. Generate cluster titles
python generate_titles_trained_clustering.py --command submit --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" --layer 12
# Wait 1-24 hours for batch completion
python generate_titles_trained_clustering.py --command process --check_status

# 3. Evaluate clustering performance
python evaluate_trained_clustering.py --command submit --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" --layer 12
# Wait 1-24 hours for batch completion  
python evaluate_trained_clustering.py --command process --check_status
```