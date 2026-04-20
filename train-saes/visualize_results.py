# %%
import sys
import os
import matplotlib.pyplot as plt
import argparse
import json
import numpy as np

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
parser.add_argument("--min_clusters", type=int, default=4,
                    help="Minimum number of clusters")
parser.add_argument("--max_clusters", type=int, default=20,
                    help="Maximum number of clusters")
parser.add_argument("--clusters", type=int, nargs='+', default=[5,10,15,20,25,30,35,40,45,50],
                    help="Comma-separated list of cluster sizes to visualize")
parser.add_argument("--clustering_methods", type=str, nargs='+', 
                    default=["gmm", "pca_gmm", "spherical_kmeans", "pca_kmeans", "agglomerative", "pca_agglomerative", "sae_topk"],
                    help="Clustering methods to use")
args, _ = parser.parse_known_args()

# %% Get model identifier for file naming
model_id = args.model.split('/')[-1].lower()

CLUSTERING_METHODS = {
    'agglomerative',
    'pca_agglomerative',
    'gmm',
    'pca_gmm',
    'spherical_kmeans',
    'pca_kmeans',
    'sae_topk'
}

clustering_methods = [method for method in args.clustering_methods if method in CLUSTERING_METHODS]

# %%

def visualize_results(results_json_path, args):
    """
    Create a comprehensive visualization of clustering results from the results JSON.
    Shows median values with uncertainty bands (min/max) across repetitions.
    
    Parameters:
    -----------
    results_json_path : str
        Path to the JSON file containing the clustering results
    args : argparse.Namespace
        Command line arguments containing cluster configuration
    """
    # Load results from JSON
    with open(results_json_path, 'r') as f:
        results = json.load(f)
    
    # Extract basic info from new format
    model_id = results['model_id']
    layer = results['layer']
    method = results['clustering_method']
    results_by_cluster_size = results['results_by_cluster_size']
    
    # Extract cluster range from available data
    available_clusters = sorted([int(k) for k in results_by_cluster_size.keys()])
    print(f"Cluster range available: {available_clusters}")
    
    # Filter to desired range from args.clusters
    cluster_range_to_keep = args.clusters
    cluster_range = [c for c in cluster_range_to_keep if c in available_clusters]
    
    if not cluster_range:
        print("No data available for the desired cluster range")
        return
    
    # Initialize lists for metrics
    metrics = {
        'final_scores': {'mean': [], 'min': [], 'max': []},
        'f1_scores': {'mean': [], 'min': [], 'max': []},
        'accuracy_scores': {'mean': [], 'min': [], 'max': []},
        'confidence_scores': {'mean': [], 'min': [], 'max': []},
        'orthogonality_scores': {'mean': [], 'min': [], 'max': []},
        'semantic_orthogonality_scores': {'mean': [], 'min': [], 'max': []},
        'recall_scores': {'mean': [], 'min': [], 'max': []}
    }
    
    # Extract data for each cluster count
    for n_clusters in cluster_range:
        print(f"Loading cluster results for {n_clusters}")
        cluster_results = results_by_cluster_size[str(n_clusters)]
        all_repetitions = cluster_results['all_results']
        
        # Extract metrics from each repetition
        rep_final_scores = [rep.get('final_score', 0) for rep in all_repetitions]
        rep_f1_scores = [rep.get('avg_f1', 0) for rep in all_repetitions]
        rep_accuracy_scores = [rep.get('avg_accuracy', 0) for rep in all_repetitions]
        rep_confidence_scores = [rep.get('avg_confidence', 0) for rep in all_repetitions]  # This metric is called 'Completeness' in the plots
        rep_orthogonality_scores = [rep.get('orthogonality', 0) for rep in all_repetitions]
        rep_semantic_orthogonality_scores = [rep.get('semantic_orthogonality_score', 0) for rep in all_repetitions]
        rep_recall_scores = [rep.get('avg_recall', 0) for rep in all_repetitions]
        
        # Calculate statistics across repetitions
        for metric_name, values in [
            ('final_scores', rep_final_scores),
            ('f1_scores', rep_f1_scores),
            ('accuracy_scores', rep_accuracy_scores),
            ('confidence_scores', rep_confidence_scores),
            ('orthogonality_scores', rep_orthogonality_scores),
            ('semantic_orthogonality_scores', rep_semantic_orthogonality_scores),
            ('recall_scores', rep_recall_scores)
        ]:
            metrics[metric_name]['mean'].append(np.mean(values))
            metrics[metric_name]['min'].append(np.min(values))
            metrics[metric_name]['max'].append(np.max(values))
    
    # Find optimal cluster count based on median final score
    optimal_n_clusters = int(results['best_cluster']['size'])
    
    # Create figure with 3x2 subplots
    fig, axs = plt.subplots(4, 2, figsize=(15, 24))
    
    # Define x-coordinates for vertical lines
    vertical_lines_x = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    
    def plot_with_uncertainty(ax, x, mean, min_vals, max_vals, color, xlabel, ylabel, title):
        """Helper function to plot line with uncertainty band"""
        # Plot uncertainty band
        ax.fill_between(x, min_vals, max_vals, alpha=0.2, color=color, label='Min-Max Range')
        # Plot mean line
        ax.plot(x, mean, 'o-', color=color, linewidth=2, label='Mean')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0.4, 0.9)  # Set y-axis range to 0.4 - 0.9
        ax.axvline(x=optimal_n_clusters, color='gray', linestyle='--', alpha=0.7, label='Optimal')

        # Add vertical grid lines
        for x_line in vertical_lines_x:
            ax.axvline(x=x_line, color='red', linestyle='--', alpha=0.15)

        ax.legend()
    
    # Final Score (combined metric) - Top Left
    plot_with_uncertainty(
        axs[0, 0], cluster_range, 
        metrics['final_scores']['mean'], 
        metrics['final_scores']['min'], 
        metrics['final_scores']['max'], 
        'blue', 'Number of Clusters', 'Final Score', 
        'Final Score vs. Number of Clusters'
    )
    
    # F1 Score - Top Right
    plot_with_uncertainty(
        axs[0, 1], cluster_range,
        metrics['f1_scores']['mean'],
        metrics['f1_scores']['min'],
        metrics['f1_scores']['max'],
        'red', 'Number of Clusters', 'Average F1 Score',
        'Average F1 Score vs. Number of Clusters'
    )
    
    # Semantic Orthogonality - Middle Left
    plot_with_uncertainty(
        axs[1, 0], cluster_range,
        metrics['semantic_orthogonality_scores']['mean'],
        metrics['semantic_orthogonality_scores']['min'],
        metrics['semantic_orthogonality_scores']['max'],
        'brown', 'Number of Clusters', 'Semantic Orthogonality',
        'Semantic Orthogonality vs. Number of Clusters'
    )
    
    # Completeness - Middle Right
    plot_with_uncertainty(
        axs[1, 1], cluster_range,
        metrics['confidence_scores']['mean'],
        metrics['confidence_scores']['min'],
        metrics['confidence_scores']['max'],
        'purple', 'Number of Clusters', 'Completeness',
        'Completeness vs. Number of Clusters'
    )
    
    # Centroid Orthogonality - Bottom Left
    plot_with_uncertainty(
        axs[2, 0], cluster_range,
        metrics['orthogonality_scores']['mean'],
        metrics['orthogonality_scores']['min'],
        metrics['orthogonality_scores']['max'],
        'orange', 'Number of Clusters', 'Orthogonality',
        'Centroid Orthogonality vs. Number of Clusters'
    )
    
    # Accuracy - Bottom Right
    plot_with_uncertainty(
        axs[2, 1], cluster_range,
        metrics['accuracy_scores']['mean'],
        metrics['accuracy_scores']['min'],
        metrics['accuracy_scores']['max'],
        'green', 'Number of Clusters', 'Accuracy',
        'Autograder Accuracy vs. Number of Clusters'
    )
    
    # Average Recall - Bottom Right
    plot_with_uncertainty(
        axs[3, 0], cluster_range,
        metrics['recall_scores']['mean'],
        metrics['recall_scores']['min'],
        metrics['recall_scores']['max'],
        'cyan', 'Number of Clusters', 'Average Recall',
        'Average Recall vs. Number of Clusters'
    )
    
    # Hide the unused subplot
    axs[3, 1].axis('off')
    
    # Add overall title
    method_name = method.capitalize()
    plt.suptitle(f'{method_name} Clustering Metrics Summary (Model: {model_id}, Layer: {layer})\nLines show mean across repetitions, bands show min-max range', fontsize=14)
    
    # Save figure
    plt.tight_layout(rect=(0, 0, 1, 0.96))  # Adjust for suptitle
    os.makedirs('results/figures', exist_ok=True)
    model_short_name = args.model.split('/')[-1].lower()
    save_path = f'results/figures/{method}_summary_{model_short_name}_layer{layer}.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print_and_flush(f"Saved {method_name} summary visualization to {save_path}")
    
    # plt.show()

# %%

def print_concise_summary(results_data, clustering_method):
    """
    Print a concise summary of the clustering results.
    
    Parameters:
    -----------
    results_data : dict
        Dictionary containing the clustering results
    clustering_method : str
        Name of the clustering method
    """
    # Extract data for printing from new format
    model_id = results_data['model_id']
    layer = results_data['layer']
    results_by_cluster_size = results_data['results_by_cluster_size']
    best_cluster = results_data.get('best_cluster', {})
    
    # Get available cluster range and optimal from the results
    available_clusters = sorted([int(k) for k in results_by_cluster_size.keys()])
    
    # Get optimal cluster size from best_cluster or calculate it
    optimal_n_clusters = best_cluster.get('size')
    if optimal_n_clusters is not None:
        optimal_n_clusters = int(optimal_n_clusters)
    else:
        # Calculate optimal based on mean final scores
        final_scores = []
        for n_clusters in available_clusters:
            cluster_results = results_by_cluster_size[str(n_clusters)]
            all_repetitions = cluster_results['all_results']
            rep_final_scores = [rep['final_score'] for rep in all_repetitions]
            final_scores.append(np.mean(rep_final_scores))
        optimal_n_clusters = available_clusters[np.argmax(final_scores)]

    # Print concise summary of experiment
    print_and_flush("\n" + "="*50)
    model_display = model_id.upper() if model_id else "UNKNOWN"
    print_and_flush(f"{clustering_method.upper()} CLUSTERING SUMMARY - {model_display} Layer {layer}")
    print_and_flush("="*50)

    print_and_flush(f"- Tested cluster range: {min(available_clusters)} to {max(available_clusters)}")
    print_and_flush(f"- Optimal number of clusters: {optimal_n_clusters}")

    # Print results for all cluster sizes
    print_and_flush("\nMetrics for all cluster sizes:")
    print_and_flush(f"{'Clusters':<10} {'Accuracy':<12} {'Avg F1':<12} {'Orthogonality':<15}")
    print_and_flush(f"{'-'*10} {'-'*12} {'-'*12} {'-'*15}")

    for n_clusters in available_clusters:
        cluster_results = results_by_cluster_size[str(n_clusters)]
        all_repetitions = cluster_results['all_results']
        
        # Calculate means across repetitions
        accuracy = np.mean([rep['avg_accuracy'] for rep in all_repetitions])
        avg_f1 = np.mean([rep['avg_f1'] for rep in all_repetitions])
        orthogonality = np.mean([rep['orthogonality'] for rep in all_repetitions])
        
        # Highlight the optimal cluster size
        prefix = "* " if n_clusters == optimal_n_clusters else "  "
        
        print_and_flush(f"{prefix}{n_clusters:<8} {accuracy:<12.4f} "
                f"{avg_f1:<12.4f} {orthogonality:<15.4f}")

    # Print top clusters in optimal clustering
    print_and_flush("\nTop clusters with optimal clustering (K={}):\n".format(optimal_n_clusters))
    print_and_flush(f"{'Cluster ID':<10} {'Title':<40} {'Size':<8} {'Precision':<12} {'Recall':<12} {'F1':<8}")
    print_and_flush(f"{'-'*10} {'-'*40} {'-'*8} {'-'*12} {'-'*12} {'-'*8}")

    # Get details from the optimal clustering - take the best repetition
    optimal_cluster_results = results_by_cluster_size[str(optimal_n_clusters)]
    all_repetitions = optimal_cluster_results['all_results']
    
    # Find the best repetition based on final score
    best_repetition = max(all_repetitions, key=lambda x: x['final_score'])
    optimal_detailed_results = best_repetition['detailed_results']

    # Sort clusters by size
    sorted_clusters = sorted(
        optimal_detailed_results.items(),
        key=lambda x: x[1].get('size', 0) if isinstance(x[1], dict) else 0,
        reverse=True
    )

    for cluster_id, cluster_data in sorted_clusters:
        if not isinstance(cluster_data, dict):
            continue
            
        title_short = cluster_data.get('title', 'Unknown')[:37] + "..." if len(cluster_data.get('title', '')) > 40 else cluster_data.get('title', 'Unknown')
        size = cluster_data.get('size', 0)
        precision = cluster_data.get('precision', 0)
        recall = cluster_data.get('recall', 0)
        f1 = cluster_data.get('f1', 0)
        
        print_and_flush(f"{cluster_id:<10} {title_short:<40} {size:<8} "
                f"{precision:<12.4f} {recall:<12.4f} {f1:<.4f}")

# %%

for clustering_method in clustering_methods:
    results_json_path = f'results/vars/{clustering_method}_results_{model_id}_layer{args.layer}.json'
    if not os.path.exists(results_json_path):
        print_and_flush(f"Results file {results_json_path} does not exist")
        continue
    
    with open(results_json_path, 'r') as f:
        results_data = json.load(f)
    print_and_flush(f"Loaded {clustering_method} results from {results_json_path}")

    # Visualize results
    visualize_results(results_json_path, args)

# %%