# %%
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json

# %%
def print_and_flush(message):
    """Prints a message and flushes stdout."""
    print(message)
    sys.stdout.flush()

parser = argparse.ArgumentParser(description="K-means clustering and autograding of neural activations")
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                    help="Model to analyze")
parser.add_argument("--layer", type=int, default=12,
                    help="Layer to analyze")
args, _ = parser.parse_known_args()

# %% Get model identifier for file naming
model_id = args.model.split('/')[-1].lower()

# %%

def visualize_method_comparison(model_id, layer, all_results):
    """
    Create a comparative visualization of different clustering methods as a bar chart.
    
    Parameters:
    -----------
    model_id : str
        Model identifier
    layer : int
        Layer number
    all_results : dict
        Dictionary mapping method names to their results
    """
    # Create directory for figures if it doesn't exist
    os.makedirs('results/figures', exist_ok=True)
    
    # Method naming dictionary for display names
    method_names = {
        "pca_agglomerative": "PCA + Agglomerative", 
        "agglomerative": "Agglomerative",
        "pca_gmm": "PCA + GMM",
        "gmm": "GMM",
        "spherical_kmeans": "K-means",
        "pca_kmeans": "PCA + K-means",
        "sae_topk": "SAE TopK",
    }
    
    # Font sizes for plot elements
    font_sizes = {
        "title": 18,
        "axes": 16,
        "x-ticks": 12,
        "y-ticks": 14,
        "legend": 14,
        "bar_values": 10
    }
    
    # Extract metrics for each method
    methods = []
    display_names = []
    orthogonality_scores = []
    assignment_rates = []
    accuracy_scores = []
    cluster_counts = []
    
    for method, results in all_results.items():
        try:
            methods.append(method)
            display_names.append(method_names.get(method, method.replace('_', ' ').title()))
            
            # Extract metrics from new format - use best_cluster section
            best_cluster = results['best_cluster']
            
            orthogonality_scores.append(best_cluster['orthogonality'])
            assignment_rates.append(best_cluster['avg_completeness'])
            
            # Use avg_precision as accuracy (best available metric)
            accuracy_scores.append(best_cluster['avg_precision'])
            cluster_counts.append(int(best_cluster['size']))
                
        except Exception as e:
            print_and_flush(f"Error processing results for {method}: {e}")
            import traceback
            print(traceback.format_exc())
            # Remove the method if we couldn't extract its metrics
            if method in methods:
                idx = methods.index(method)
                methods.pop(idx)
                display_names.pop(idx)
    
    if not methods:
        print_and_flush("No valid methods with results to visualize")
        return
    
    # Create bar chart
    x = np.arange(len(methods))
    width = 0.25
    
    fig, ax = plt.figure(figsize=(14, 8)), plt.axes()
    
    # Plot bars with nice colors
    bars1 = ax.bar(x - width, orthogonality_scores, width, label='Orthogonality', color='skyblue')
    bars2 = ax.bar(x, assignment_rates, width, label='Completeness', color='salmon')
    bars3 = ax.bar(x + width, accuracy_scores, width, label='Precision', color='lightgreen', edgecolor='black', linewidth=1)
    
    # Add labels and title
    ax.set_ylabel('Score', fontsize=font_sizes["axes"])
    ax.set_title(f'Comparison of Clustering Methods ({model_id.upper()}, Layer {layer})', fontsize=font_sizes["title"])
    
    # Create x-tick labels with cluster counts
    display_names_with_counts = [f"{name}\n(k={count})" for name, count in zip(display_names, cluster_counts)]
    ax.set_xticks(x)
    ax.set_xticklabels(display_names_with_counts, fontsize=font_sizes["x-ticks"])
    ax.legend(fontsize=font_sizes["legend"])
    
    # Apply font size to tick labels
    ax.tick_params(axis='y', labelsize=font_sizes["y-ticks"])
    
    # Ensure y-axis starts at 0 and goes to slightly above 1
    ax.set_ylim(0, 1.1)
    
    # Add value labels on top of bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=font_sizes["bar_values"])
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    plt.tight_layout()
    plt.savefig(f'results/figures/method_comparison_{model_id}_layer{layer}.pdf')
    # plt.show()
    
    print_and_flush(f"Saved comparison visualization to results/figures/method_comparison_{model_id}_layer{layer}.pdf")

# %%

def print_methods_comparison_summary(all_results):
    print_and_flush("\n" + "="*50)
    print_and_flush("CURRENT RUN METHODS COMPARISON")
    print_and_flush("="*50)
    print_and_flush(f"{'Method':<15} {'Optimal K':<10} {'Accuracy':<12} {'F1 Score':<12} {'Orthogonality':<15}")
    print_and_flush(f"{'-'*15} {'-'*10} {'-'*12} {'-'*12} {'-'*15}")
    
    for method, results in all_results.items():
        # Get the metrics from new format - use best_cluster section
        best_cluster = results['best_cluster']
        n_clusters = int(best_cluster['size'])
        accuracy = best_cluster['avg_accuracy']
        f1 = best_cluster['avg_f1']
        orthogonality = best_cluster['orthogonality']
            
        print_and_flush(f"{method.capitalize():<15} {n_clusters:<10} "
              f"{accuracy:<12.4f} "
              f"{f1:<12.4f} {orthogonality:<15.4f}")

# %%

def load_all_results(model_id, layer):
    """
    Load all available clustering results for a specific model and layer.
    
    Parameters:
    -----------
    model_id : str
        Model identifier
    layer : int
        Layer number
        
    Returns:
    --------
    dict
        Dictionary mapping clustering methods to their results
    """
    results_dir = 'results/vars'
    all_results = {}
    
    # Create directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Find all result files for this model and layer
    for filename in os.listdir(results_dir):
        # Filter for result files matching this model and layer
        if f"_{model_id}_layer{layer}.json" in filename:
            method = filename.split('_results_')[0]
            file_path = os.path.join(results_dir, filename)
            
            try:
                with open(file_path, 'r') as f:
                    results = json.load(f)
                
                # Store in all_results dictionary
                all_results[method] = results
                print_and_flush(f"Loaded results for {method}")
            except Exception as e:
                print_and_flush(f"Error loading {file_path}: {e}")
    
    if not all_results:
        print_and_flush(f"No existing results found for model {model_id} layer {layer}")
    else:
        print_and_flush(f"Loaded {len(all_results)} result sets for visualization")
    
    return all_results

# %% Load ALL available results and create comprehensive visualization
print_and_flush("\n" + "="*50)
print_and_flush("LOADING ALL AVAILABLE RESULTS FOR COMPREHENSIVE COMPARISON")
print_and_flush("="*50)

all_results = load_all_results(model_id, args.layer)

if len(all_results) > 1:
    # Create comparison visualization
    visualize_method_comparison(model_id, args.layer, all_results)

print_methods_comparison_summary(all_results)