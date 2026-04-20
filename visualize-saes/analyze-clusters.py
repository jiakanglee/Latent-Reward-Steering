# %%
import json

# %%
layers = [6, 10, 14, 18, 22, 26]
n_cluster_range = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

final_scores_by_layer_n_clusters = {}  # (layer, n_clusters) -> final_score

for layer in layers:
    try:
        with open(f'../train-saes/results/vars/sae_topk_results_deepseek-r1-distill-llama-8b_layer{layer}.json') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Results file not found for layer {layer}")
        continue

    # Use new format: results_by_cluster_size instead of detailed_results
    results_by_cluster_size = results['results_by_cluster_size']

    for n_clusters in n_cluster_range:
        if str(n_clusters) not in results_by_cluster_size:
            print(f"Warning: No data for layer {layer}, {n_clusters} clusters")
            continue

        cluster_results = results_by_cluster_size[str(n_clusters)]
        avg_final_score = cluster_results['avg_final_score']
        final_scores_by_layer_n_clusters[(layer, n_clusters)] = avg_final_score

print("=== Clusters sorted by Average Final Score ===")
sorted_clusters = sorted(final_scores_by_layer_n_clusters.items(), key=lambda x: x[1], reverse=False)
for (layer, n_clusters), score in sorted_clusters:
    print(f"Layer {layer}, {n_clusters} clusters: {score:.4f}")

# Find the overall best configuration
if final_scores_by_layer_n_clusters:
    best_config = max(final_scores_by_layer_n_clusters.items(), key=lambda x: x[1])
    best_layer, best_n_clusters = best_config[0]
    best_score = best_config[1]
    print(f"\n=== Best Configuration ===")
    print(f"Layer {best_layer}, {best_n_clusters} clusters: {best_score:.4f}")

# %%