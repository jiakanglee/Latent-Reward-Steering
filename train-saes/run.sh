#!/usr/bin/env bash
CLUSTERS="5 10 15 20 25 30 35 40 45 50" # 5 10 15 20 25 30 35 40 45 50
N_EXAMPLES=100000  # all responses

# CLUSTERING_METHODS="gmm pca_gmm spherical_kmeans pca_kmeans agglomerative pca_agglomerative sae_topk"
CLUSTERING_METHODS="sae_topk"

MODELS="deepseek-ai/DeepSeek-R1-Distill-Llama-8B deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B deepseek-ai/DeepSeek-R1-Distill-Qwen-14B qwen/QwQ-32B deepseek-ai/DeepSeek-R1-Distill-Qwen-32B deepseek-ai/DeepSeek-R1-Distill-Llama-70B Open-Reasoner-Zero/Open-Reasoner-Zero-0.5B Open-Reasoner-Zero/Open-Reasoner-Zero-1.5B Open-Reasoner-Zero/Open-Reasoner-Zero-7B Open-Reasoner-Zero/Open-Reasoner-Zero-32B"

REPETITIONS=3

# Rationale for selecting `--layers` per model:
# - We pick 6 layer indices per model, roughly evenly spaced through depth, to keep runs comparable and avoid training/evaluation over every layer.
# - The implicit pattern used below is a constant step size per model, approximately
# step ≈ (total_layers / 8), rounded to an integer, with indices chosen to stay in-range.
get_layers() {
    local model=$1
    case "$model" in
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B") echo "6 10 14 18 22 26" ;; # Total layers: 32
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B") echo "4 8 12 16 20 24" ;; # Total layers: 28
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B") echo "8 14 20 26 32 38" ;; # Total layers: 48
        "qwen/QwQ-32B") echo "9 18 27 36 45 54" ;; # Total layers: 64
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B") echo "9 18 27 36 45 54" ;; # Total layers: 64
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B") echo "11 22 33 44 55 66" ;; # Total layers: 80
        "Open-Reasoner-Zero/Open-Reasoner-Zero-0.5B") echo "5 8 11 14 17 20" ;; # Total layers: 24
        "Open-Reasoner-Zero/Open-Reasoner-Zero-1.5B") echo "4 8 12 16 20 24" ;; # Total layers: 28
        "Open-Reasoner-Zero/Open-Reasoner-Zero-7B") echo "4 8 12 16 20 24" ;; # Total layers: 28
        "Open-Reasoner-Zero/Open-Reasoner-Zero-32B") echo "9 18 27 36 45 54" ;; # Total layers: 64
        *) echo "" ;;
    esac
}

# Generate activations for all models and layers
for MODEL in $MODELS; do
    LAYERS_TO_PROCESS=$(get_layers "$MODEL")
    if [ -n "$LAYERS_TO_PROCESS" ]; then
        python generate_activations.py --model "$MODEL" --layers $LAYERS_TO_PROCESS --n_examples $N_EXAMPLES
    fi
done

# Train all clustering methods for all models and layers
for MODEL in $MODELS; do
    for LAYER in $(get_layers $MODEL); do
        python train_clustering.py --model $MODEL --layer $LAYER --clusters $CLUSTERS --n_examples $N_EXAMPLES --clustering_methods $CLUSTERING_METHODS
    done
done

# Generate titles for all clustering methods for all models and layers
# Uses OpenAI's batch API by default. Change to --command direct if you want to generate titles directly, and ommit the next loop that calls the command "process"
for MODEL in $MODELS; do
    for LAYER in $(get_layers $MODEL); do
        python generate_titles_trained_clustering.py --model $MODEL --layer $LAYER --clusters $CLUSTERS --n_examples $N_EXAMPLES --clustering_methods $CLUSTERING_METHODS --repetitions $REPETITIONS --command submit
    done
done

# Wait for titles to be generated
for MODEL in $MODELS; do
    for LAYER in $(get_layers $MODEL); do
        python generate_titles_trained_clustering.py --model $MODEL --layer $LAYER --clusters $CLUSTERS --n_examples $N_EXAMPLES --clustering_methods $CLUSTERING_METHODS --repetitions $REPETITIONS --command process --wait-batch-completion
    done
done

# Evaluate all clustering methods for all models and layers
# Uses OpenAI's batch API by default. Change to --command direct if you want to generate titles directly, and ommit the next loop that calls the command "process"
for MODEL in $MODELS; do
    for LAYER in $(get_layers $MODEL); do
        # Extra flags to disable re-computing some of the evaluation metrics, use as needed: --no-accuracy --no-completeness --no-orth --no-sem-orth
        python evaluate_trained_clustering.py --model $MODEL --layer $LAYER --clusters $CLUSTERS --n_examples $N_EXAMPLES --clustering_methods $CLUSTERING_METHODS --repetitions $REPETITIONS --command submit --accuracy_target_cluster_percentage 0.2
    done
done

# Wait for evaluation to complete
for MODEL in $MODELS; do
    for LAYER in $(get_layers $MODEL); do
        python evaluate_trained_clustering.py --model $MODEL --layer $LAYER --clusters $CLUSTERS --n_examples $N_EXAMPLES --clustering_methods $CLUSTERING_METHODS --repetitions $REPETITIONS --command process --wait-batch-completion
    done
done

# Visualize all clustering methods for all models and layers
for MODEL in $MODELS; do
    for LAYER in $(get_layers $MODEL); do
        python visualize_results.py --model $MODEL --layer $LAYER --clusters 5 10 15 20 25 30 35 40 45 50 --clustering_methods $CLUSTERING_METHODS
        python visualize_comparison.py --model $MODEL --layer $LAYER
    done
    python visualize_clusters.py --model $MODEL
done
python visualize_clusters.py --model all