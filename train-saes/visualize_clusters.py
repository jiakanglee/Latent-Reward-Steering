# %%
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import argparse
    
import random

parser = argparse.ArgumentParser(description="Visualize SAE grid search results")
parser.add_argument("--model", type=str, default="all",
                    help="Model identifier")
parser.add_argument("--print-stats", action="store_true", default=False,
                    help="Print all statistics to stdout instead of generating plots")
parser.add_argument("--max-repetitions", type=int, default=None,
                    help="Maximum number of repetitions to use (randomly samples if more available)")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for sampling repetitions (default: 42)")

args, _ = parser.parse_known_args()


def sample_repetitions(all_results, max_reps, seed, identifier=""):
    """
    Sample repetitions if we have more than max_reps.

    Parameters:
    -----------
    all_results : list
        List of repetition results
    max_reps : int or None
        Maximum number of repetitions to keep (None = keep all)
    seed : int
        Random seed for reproducibility
    identifier : str
        Identifier string for deterministic sampling (e.g., "model_layer_clusters")

    Returns:
    --------
    list
        Sampled list of repetition results
    """
    if max_reps is None or len(all_results) <= max_reps:
        return all_results

    # Use identifier to make sampling deterministic across calls with same data
    rng = random.Random(seed + hash(identifier) % (2**31))
    indices = list(range(len(all_results)))
    rng.shuffle(indices)
    selected_indices = sorted(indices[:max_reps])

    return [all_results[i] for i in selected_indices]


def compute_statistics_from_results(all_results):
    """
    Compute statistics from a list of repetition results.

    Parameters:
    -----------
    all_results : list
        List of repetition result dicts

    Returns:
    --------
    dict
        Statistics dict with mean, std_dev, sem, ci_95 for each metric
    """
    from scipy import stats as scipy_stats

    metrics_to_stat = [
        'avg_accuracy', 'avg_f1', 'avg_precision', 'avg_recall', 'orthogonality',
        'semantic_orthogonality_score', 'avg_confidence', 'final_score'
    ]

    statistics = {}
    for metric in metrics_to_stat:
        values = [res[metric] for res in all_results if metric in res]
        if not values:
            continue

        mean = np.mean(values)
        n = len(values)
        if n > 1:
            std_dev = np.std(values, ddof=1)
            sem = scipy_stats.sem(values)
            conf = 0.95
            t_score = scipy_stats.t.ppf((1 + conf) / 2, df=n - 1)
            ci_95 = (mean - t_score * sem, mean + t_score * sem)
        else:
            std_dev = 0
            sem = 0
            ci_95 = (mean, mean)

        statistics[metric] = {
            'mean': mean,
            'std_dev': std_dev,
            'sem': sem,
            'ci_95': ci_95
        }

    return statistics

def load_sae_grid_search_results(model_id, method="sae_topk", max_reps=None, seed=42):
    """
    Load all SAE grid search results for a specific model across all layers and cluster sizes.

    Parameters:
    -----------
    model_id : str
        Model identifier (e.g., "deepseek-r1-distill-qwen-1.5b")
    method : str
        Clustering method to load (default: "sae_topk")
    max_reps : int or None
        Maximum number of repetitions to use (None = use all)
    seed : int
        Random seed for sampling repetitions

    Returns:
    --------
    DataFrame
        DataFrame containing metrics for each layer/n_clusters configuration
    """
    results_dir = 'results/vars'
    results_data = []

    # Find all result files for this model and method
    for filename in os.listdir(results_dir):
        # Filter for result files matching this model and method
        if f"{method}_results_{model_id}_layer" in filename and filename.endswith(".json"):
            try:
                # Extract layer number
                layer_str = filename.split(f"{method}_results_{model_id}_layer")[1]
                layer = int(layer_str.split(".json")[0])

                file_path = os.path.join(results_dir, filename)
                with open(file_path, 'r') as f:
                    results = json.load(f)

                # New structure: results_by_cluster_size contains cluster sizes as keys
                for n_clusters_str, cluster_data in results["results_by_cluster_size"].items():
                    n_clusters = int(n_clusters_str)

                    # Extract all repetitions for this cluster size
                    all_results = cluster_data.get("all_results", [])

                    if not all_results:
                        print(f"Warning: No results found for layer {layer}, clusters {n_clusters}")
                        continue

                    # Sample repetitions if max_reps is set
                    identifier = f"{model_id}_{layer}_{n_clusters}"
                    sampled_results = sample_repetitions(all_results, max_reps, seed, identifier)

                    # Recompute statistics from sampled results
                    if max_reps is not None and len(all_results) > max_reps:
                        statistics = compute_statistics_from_results(sampled_results)
                    else:
                        statistics = cluster_data.get("statistics", {})
                        if not statistics:
                            statistics = compute_statistics_from_results(sampled_results)

                    # Calculate average metrics across sampled repetitions
                    avg_orthogonality = statistics.get("orthogonality", {}).get("mean", 0)
                    avg_accuracy = statistics.get("avg_accuracy", {}).get("mean", 0)
                    avg_f1 = statistics.get("avg_f1", {}).get("mean", 0)
                    avg_completeness = statistics.get("avg_confidence", {}).get("mean", 0)
                    avg_semantic_orthogonality = statistics.get("semantic_orthogonality_score", {}).get("mean", 0)
                    avg_final_score = statistics.get("final_score", {}).get("mean", 0)

                    # Check for dead latents by examining detailed results from first sampled repetition
                    has_dead_latents = False
                    active_clusters = n_clusters  # Default assumption

                    if sampled_results and "detailed_results" in sampled_results[0]:
                        detailed = sampled_results[0]["detailed_results"]
                        active_clusters = 0

                        # Count clusters with size > 0
                        for cluster_id_str, cluster_info in detailed.items():
                            if cluster_info.get("size", 0) > 0:
                                active_clusters += 1

                        has_dead_latents = active_clusters < n_clusters

                    metrics = {
                        "layer": layer,
                        "n_clusters": n_clusters,
                        "orthogonality": avg_orthogonality,
                        "semantic_orthogonality": avg_semantic_orthogonality,
                        "accuracy": avg_accuracy,
                        "f1": avg_f1,
                        "completeness": avg_completeness,
                        "final_score": avg_final_score,
                        "has_dead_latents": has_dead_latents,
                        "active_clusters": active_clusters
                    }
                    results_data.append(metrics)

                print(f"Loaded results for layer {layer}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    if not results_data:
        print(f"No grid search results found for model {model_id} with method {method}")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(results_data)
    return df

def visualize_grid_search(results_df, model_id, output_dir="results/figures"):
    """
    Visualize the SAE grid search results with heatmaps.
    
    Parameters:
    -----------
    results_df : DataFrame
        DataFrame containing metrics for each layer/n_clusters configuration
    model_id : str
        Model identifier for the plot title
    output_dir : str
        Directory to save the visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define metrics to visualize
    metrics = ["avg_f1", "avg_confidence", "semantic_orthogonality_score", "final_score"]
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    # Define custom colormap that goes from red (weak) to white to blue (strong)
    cmap = LinearSegmentedColormap.from_list(
        'custom_diverging',
        [(0.8, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.0, 0.8)],
        N=256
    )
    
    # Pivot the DataFrame for each metric
    for i, metric in enumerate(metrics):
        # Create a pivot table
        pivot = results_df.pivot_table(
            index='layer', 
            columns='n_clusters', 
            values=metric,
            aggfunc='mean'  # In case there are duplicates
        )
        
        # Sort indices to make sure they're in ascending order
        pivot = pivot.sort_index(axis=0).sort_index(axis=1)
        
        # Create heatmap with colors for all cells but annotations only for best/worst
        ax = axes[i]
        
        # First create heatmap with all colors but no annotations
        hm = sns.heatmap(
            pivot,
            annot=False,  # No annotations initially
            cmap=cmap,
            center=0.5,
            vmin=0,
            vmax=1,
            cbar=False,  # No individual colorbars
            ax=ax
        )
        
        # Find top 3 and bottom 3 configurations
        flat_pivot = pivot.values.flatten()
        top_indices = np.argsort(flat_pivot)[-3:][::-1]  # Indices of top 3 values (highest first)
        bottom_indices = np.argsort(flat_pivot)[:3]  # Indices of bottom 3 values (lowest first)
        
        # Convert flat indices to 2D coordinates
        top_positions = [np.unravel_index(idx, pivot.shape) for idx in top_indices]
        bottom_positions = [np.unravel_index(idx, pivot.shape) for idx in bottom_indices]
        
        # Colors for top 3 (green gradient) and bottom 3 (red gradient)
        top_colors = ['darkgreen', 'green', 'forestgreen']  
        bottom_colors = ['darkred', 'red', 'firebrick']
        
        # Add annotations for top 3 and bottom 3
        for i, (pos, color) in enumerate(zip(top_positions, top_colors)):
            text = f"{pivot.iloc[pos[0], pos[1]]:.2f}"
            ax.text(pos[1] + 0.5, pos[0] + 0.5, text,
                   ha="center", va="center",
                   fontsize=20, weight="bold", color="black")
            ax.add_patch(plt.Rectangle((pos[1], pos[0]), 1, 1, fill=False, 
                                      edgecolor=color, lw=5, linestyle='-'))
        
        for i, (pos, color) in enumerate(zip(bottom_positions, bottom_colors)):
            text = f"{pivot.iloc[pos[0], pos[1]]:.2f}"
            ax.text(pos[1] + 0.5, pos[0] + 0.5, text,
                   ha="center", va="center",
                   fontsize=20, weight="bold", color="black")
            ax.add_patch(plt.Rectangle((pos[1], pos[0]), 1, 1, fill=False, 
                                      edgecolor=color, lw=5, linestyle='-'))
        
        # Grey out cells with dead latents
        for row in range(pivot.shape[0]):
            for col in range(pivot.shape[1]):
                if (row < has_dead_latents_pivot.shape[0] and 
                    col < has_dead_latents_pivot.shape[1] and
                    pd.notna(has_dead_latents_pivot.iloc[row, col]) and 
                    has_dead_latents_pivot.iloc[row, col]):
                    # Use hatching pattern instead of gray fill
                    ax.add_patch(plt.Rectangle((col, row), 1, 1, 
                                              fill=True, 
                                              color='#000000',  # Black
                                              alpha=0.2,
                                              hatch='///', 
                                              edgecolor='#000000',  # Black
                                              linewidth=0.5))
        
        # Set labels
        ax.set_ylabel("Number of Clusters", fontsize=20) if i == 0 else ax.set_ylabel("")
        ax.set_xlabel("Layer", fontsize=12)
        
        # Add text annotations about best/worst config
        ax.text(
            0.02, 0.02, 
            f"Best: Layer {max_pos[0]}, Clusters {max_pos[1]} ({max_val:.2f})\nWorst: Layer {min_pos[0]}, Clusters {min_pos[1]} ({min_val:.2f})",
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8)
        )
    
    # Add overall title
    plt.suptitle(f"SAE Grid Search Results for {model_id.upper()}", fontsize=20)
    
    # Adjust tight_layout to reduce white borders
    plt.tight_layout(rect=[-0.05, 0, 1.05, 0.97])
    
    # Create summary table of best configurations per metric
    print("\nBest configurations per metric:")
    
    summary_data = []
    for metric in metrics:
        pivot = results_df.pivot_table(
            index='layer', 
            columns='n_clusters', 
            values=metric,
            aggfunc='mean'
        )
        
        max_pos = np.unravel_index(pivot.values.argmax(), pivot.shape)
        max_layer = pivot.index[max_pos[0]]
        max_clusters = pivot.columns[max_pos[1]]
        max_val = pivot.values[max_pos]
        
        summary_data.append({
            "metric": metric,
            "best_layer": max_layer,
            "best_clusters": max_clusters,
            "value": max_val
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df)
    
    # Find best overall configuration using final_score from JSON
    best_config = results_df.loc[results_df['final_score'].idxmax()]
    
    print("\nBest overall configuration (based on final_score):")
    print(f"Layer: {int(best_config['layer'])}, Clusters: {int(best_config['n_clusters'])}")
    print(f"Metrics: Orthogonality={best_config['orthogonality']:.2f}, F1={best_config['f1']:.2f}, " +
          f"Accuracy={best_config['accuracy']:.2f}, Completeness={best_config['completeness']:.2f}")
    print(f"Final Score: {best_config['final_score']:.3f}")
    
    return fig

def visualize_combined_grid_search(results_df, model_id, output_dir="results/figures"):
    """
    Visualize the SAE grid search results with a single heatmap of final scores.
    Grey out configurations with dead latents.
    
    Parameters:
    -----------
    results_df : DataFrame
        DataFrame containing metrics for each layer/n_clusters configuration
    model_id : str
        Model identifier for the plot title
    output_dir : str
        Directory to save the visualization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define metrics to combine
    metrics = ["f1", "completeness", "semantic_orthogonality"]
    
    # Normalize each metric to 0-1 range
    for metric in metrics:
        min_val = results_df[metric].min()
        max_val = results_df[metric].max()
        results_df[f"{metric}_norm"] = (results_df[metric] - min_val) / (max_val - min_val)
        print(f"Metric: {metric}, Min: {min_val}, Max: {max_val}, Norm Min: {results_df[f'{metric}_norm'].min()}, Norm Max: {results_df[f'{metric}_norm'].max()}")
    
    # Calculate combined score (equal weight to all metrics)
    results_df['normalized_final_score'] = (results_df['f1_norm'] + 
                                   results_df['completeness_norm'] + 
                                   results_df['semantic_orthogonality_norm']) / len(metrics)
    
    # Create figure - taller than wide
    plt.figure(figsize=(10, 14))
    
    # Define custom colormap that goes from red (weak) to white to blue (strong)
    # Using paler colors for better readability of scores
    cmap = LinearSegmentedColormap.from_list(
        'custom_diverging',
        [(0.9, 0.5, 0.5), (1.0, 1.0, 1.0), (0.5, 0.5, 0.9)],
        N=256
    )
    
    # Create a pivot table for the final score - flip axes by putting n_clusters as index and layer as columns
    pivot = results_df.pivot_table(
        index='n_clusters', 
        columns='layer', 
        values='normalized_final_score',
        aggfunc='mean'
    )
    
    # Create a mask for cells with dead latents
    has_dead_latents_pivot = results_df.pivot_table(
        index='n_clusters',
        columns='layer',
        values='has_dead_latents',
        aggfunc=lambda x: any(x)
    )
    
    # Create a pivot for active clusters (for annotations)
    active_clusters_pivot = results_df.pivot_table(
        index='n_clusters',
        columns='layer',
        values='active_clusters',
        aggfunc='mean'
    )
    
    # Sort indices to make sure they're in ascending order
    pivot = pivot.sort_index(axis=1).sort_index(axis=0, ascending=False)
    has_dead_latents_pivot = has_dead_latents_pivot.sort_index(axis=1).sort_index(axis=0, ascending=False)
    active_clusters_pivot = active_clusters_pivot.sort_index(axis=1).sort_index(axis=0, ascending=False)
    
    # Create heatmap
    ax = plt.gca()
    hm = sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0.5,
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Normalized Final Score'},
        annot_kws={"size": 14, "color": "black"}
    )
    
    # Increase colorbar label font size
    cbar = hm.collections[0].colorbar
    cbar.ax.set_ylabel('Normalized Final Score', fontsize=16)
    
    # Grey out cells with dead latents without adding text
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            if i < has_dead_latents_pivot.shape[0] and j < has_dead_latents_pivot.shape[1]:
                if pd.notna(has_dead_latents_pivot.iloc[i, j]) and has_dead_latents_pivot.iloc[i, j]:
                    # Add grey overlay for cells with dead latents
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color='grey', alpha=0.3))
    
    # Find the best and worst configurations
    max_val = pivot.max().max()
    min_val = pivot.min().min()
    max_pos = np.unravel_index(pivot.values.argmax(), pivot.shape)
    min_pos = np.unravel_index(pivot.values.argmin(), pivot.shape)
    
    # Convert positions to actual layer and n_clusters values
    max_n_clusters = pivot.index[max_pos[0]]
    max_layer = pivot.columns[max_pos[1]]
    min_n_clusters = pivot.index[min_pos[0]]
    min_layer = pivot.columns[min_pos[1]]
    
    # Mark best and worst cells with more visible colored outlines
    ax.add_patch(plt.Rectangle((min_pos[1], min_pos[0]), 1, 1, fill=False, edgecolor='darkred', lw=4))
    ax.add_patch(plt.Rectangle((max_pos[1], max_pos[0]), 1, 1, fill=False, edgecolor='darkgreen', lw=4))
    
    # Set title and labels with larger font sizes - flipped axis labels
    ax.set_ylabel("Number of Clusters", fontsize=18)
    ax.set_xlabel("Layer", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    plt.suptitle(f"SAE Grid Search Results for {model_id.upper()}", fontsize=22)
    
    # Use tight_layout with adjusted left and right margins to reduce white space
    plt.tight_layout(rect=[-0.05, 0, 1.05, 0.95])
    
    # Save figure
    save_path = os.path.join(output_dir, f"sae_combined_grid_search_{model_id}.pdf")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        
    # Count configurations with dead latents
    dead_latent_count = results_df['has_dead_latents'].sum()
    print(f"Number of configurations with dead latents: {dead_latent_count} / {len(results_df)}")
    
    return plt.gcf()

# Usage example
def main(model_id, max_reps=None, seed=42):
    """
    Main function to load and visualize SAE grid search results.

    Parameters:
    -----------
    model_id : str
        Model identifier (e.g., "deepseek-r1-distill-qwen-1.5b")
    max_reps : int or None
        Maximum number of repetitions to use
    seed : int
        Random seed for sampling repetitions
    """
    # Load grid search results
    results_df = load_sae_grid_search_results(model_id, max_reps=max_reps, seed=seed)
    print(f"Column names: {results_df.columns}")
    
    if results_df is not None:
        # Print overview of available data
        print(f"\nFound {len(results_df)} configurations across {results_df['layer'].nunique()} layers " +
              f"and {results_df['n_clusters'].nunique()} cluster sizes")
        
        # Visualize combined grid search results
        visualize_combined_grid_search(results_df, model_id)
        
        # Also visualize individual metrics if desired
        # visualize_grid_search(results_df, model_id)
    else:
        print("No results to visualize.")

def get_all_model_ids():
    """
    Extract all unique model IDs from the result files in the directory.
    
    Returns:
    --------
    list
        List of unique model IDs
    """
    results_dir = 'results/vars'
    model_ids = set()
    
    for filename in os.listdir(results_dir):
        if "sae_topk_results_" in filename and filename.endswith(".json"):
            # Extract model ID from filename
            parts = filename.split("sae_topk_results_")[1].split("_layer")
            if parts and len(parts) > 0:
                model_id = parts[0]
                model_ids.add(model_id)
    
    return sorted(list(model_ids))

def visualize_all_models(output_dir="results/figures", max_reps=None, seed=42):
    """
    Load and visualize results for DeepSeek distill, QwQ, and Open Reasoner Zero models.

    Layout:
        Row 1: Deepseek 1.5B - Deepseek Llama 8B - Deepseek Qwen 14B - Deepseek Qwen 32B - QwQ 32B
        Row 2: ORZ 0.5B - ORZ 1.5B - ORZ 7B - ORZ 32B - empty

    Parameters:
    -----------
    output_dir : str
        Directory to save the visualization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Fixed order: DeepSeek distill models, QwQ, then ORZ models
    model_ids = [
        # Row 1: DeepSeek distill models + QwQ
        "deepseek-r1-distill-qwen-1.5b",
        "deepseek-r1-distill-llama-8b",
        "deepseek-r1-distill-qwen-14b",
        "deepseek-r1-distill-qwen-32b",
        "qwq-32b",
        # Row 2: ORZ models
        "open-reasoner-zero-0.5b",
        "open-reasoner-zero-1.5b",
        "open-reasoner-zero-7b",
        "open-reasoner-zero-32b",
    ]

    print(f"Loading {len(model_ids)} models: {', '.join(model_ids)}")
    if max_reps is not None:
        print(f"Using max {max_reps} repetitions (seed={seed})")

    # Load results for all models (preserve order)
    model_results = {}
    for model_id in model_ids:
        results_df = load_sae_grid_search_results(model_id, max_reps=max_reps, seed=seed)
        if results_df is not None:
            model_results[model_id] = results_df
            print(f"Loaded results for {model_id}")
        else:
            print(f"No results found for {model_id}")

    if not model_results:
        print("No results loaded for any model.")
        return

    # Define custom colormap
    cmap = LinearSegmentedColormap.from_list(
        'custom_diverging',
        [(0.9, 0.5, 0.5), (1.0, 1.0, 1.0), (0.5, 0.5, 0.9)],
        N=256
    )

    # Create multi-panel figure with fixed 2x5 layout
    # Layout: Row 1: Deepseek 1.5B, Deepseek Llama 8B, Deepseek Qwen 14B, Deepseek Qwen 32B, QwQ 32B
    #         Row 2: ORZ 0.5B, ORZ 1.5B, ORZ 7B, ORZ 32B (centered)
    n_cols = 5
    n_rows = 2
    fig = plt.figure(figsize=(8 * n_cols, 11 * n_rows))

    # Set font sizes globally for better readability in a paper
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 32,
        'axes.labelsize': 28,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 24,
        'figure.titlesize': 36
    })

    # Create a GridSpec layout with 10 columns (2 per plot) + colorbar to allow centering bottom row
    # Each plot spans 2 columns; bottom row is offset by 1 column to center 4 plots in 5-plot width
    gs = fig.add_gridspec(n_rows, 10 + 1, width_ratios=10 * [1] + [0.1], height_ratios=n_rows * [1], top=0.88)

    # Create axes for each model in row-major order
    # Row 1: 5 models at cols 0-1, 2-3, 4-5, 6-7, 8-9
    # Row 2: 4 models at cols 1-2, 3-4, 5-6, 7-8 (offset by 1 to center)
    axes = []
    # Row 1: 5 DeepSeek/QwQ models
    for i in range(5):
        axes.append(fig.add_subplot(gs[0, i*2:(i+1)*2]))
    # Row 2: 4 ORZ models (centered by offsetting by 1 column)
    for i in range(4):
        axes.append(fig.add_subplot(gs[1, 1+i*2:1+(i+1)*2]))
    
    # Create a separate axis for the colorbar spanning all rows
    cbar_ax = fig.add_subplot(gs[:, -1])
    
    # Create visualizations for each model
    for j, (model_id, results_df) in enumerate(model_results.items()):
        metrics = ["f1", "completeness", "semantic_orthogonality"]
        for metric in metrics:
            if metric not in results_df.columns:
                raise ValueError(f"Metric {metric} not found in results for {model_id}. Columns: {results_df.columns}")
            
            min_val = results_df[metric].min()
            max_val = results_df[metric].max()
            results_df[f"{metric}_norm"] = (results_df[metric] - min_val) / (max_val - min_val)
            print(f"Metric: {metric}, Min: {min_val}, Max: {max_val}, Norm Min: {results_df[f'{metric}_norm'].min()}, Norm Max: {results_df[f'{metric}_norm'].max()}")

        results_df['normalized_final_score'] = (results_df['f1_norm'] + 
                                   results_df['completeness_norm'] + 
                                   results_df['semantic_orthogonality_norm']) / len(metrics)
        
        # Create pivot table
        pivot = results_df.pivot_table(
            index='n_clusters', 
            columns='layer', 
            values='normalized_final_score',
            aggfunc='mean'
        )
        
        # Create mask for cells with dead latents
        has_dead_latents_pivot = results_df.pivot_table(
            index='n_clusters',
            columns='layer',
            values='has_dead_latents',
            aggfunc=lambda x: any(x)
        )
        
        # Sort indices
        pivot = pivot.sort_index(axis=1).sort_index(axis=0, ascending=False)
        has_dead_latents_pivot = has_dead_latents_pivot.sort_index(axis=1).sort_index(axis=0, ascending=False)
        
        # Create heatmap with colors for all cells but annotations only for top/bottom
        ax = axes[j]
        
        # First create heatmap with all colors but no annotations
        hm = sns.heatmap(
            pivot,
            annot=False,  # No annotations initially
            cmap=cmap,
            center=0.5,
            vmin=0,
            vmax=1,
            cbar=False,  # No individual colorbars
            ax=ax
        )
        
        # Find top 3 and bottom 3 configurations
        flat_pivot = pivot.values.flatten()
        top_indices = np.argsort(flat_pivot)[-3:][::-1]  # Indices of top 3 values (highest first)
        bottom_indices = np.argsort(flat_pivot)[:3]  # Indices of bottom 3 values (lowest first)
        
        # Convert flat indices to 2D coordinates
        top_positions = [np.unravel_index(idx, pivot.shape) for idx in top_indices]
        bottom_positions = [np.unravel_index(idx, pivot.shape) for idx in bottom_indices]
        
        # Colors for top 3 (green gradient) and bottom 3 (red gradient)
        top_colors = ['darkgreen', 'green', 'forestgreen']  
        bottom_colors = ['darkred', 'red', 'firebrick']
        
        # Add annotations for top 3 and bottom 3
        for i, (pos, color) in enumerate(zip(top_positions, top_colors)):
            text = f"{pivot.iloc[pos[0], pos[1]]:.2f}"
            ax.text(pos[1] + 0.5, pos[0] + 0.5, text,
                   ha="center", va="center",
                   fontsize=22, weight="bold", color="black")
            ax.add_patch(plt.Rectangle((pos[1], pos[0]), 1, 1, fill=False, 
                                      edgecolor=color, lw=5, linestyle='-'))
        
        for i, (pos, color) in enumerate(zip(bottom_positions, bottom_colors)):
            text = f"{pivot.iloc[pos[0], pos[1]]:.2f}"
            ax.text(pos[1] + 0.5, pos[0] + 0.5, text,
                   ha="center", va="center",
                   fontsize=22, weight="bold", color="black")
            ax.add_patch(plt.Rectangle((pos[1], pos[0]), 1, 1, fill=False, 
                                      edgecolor=color, lw=5, linestyle='-'))
        
        # Grey out cells with dead latents
        for row in range(pivot.shape[0]):
            for col in range(pivot.shape[1]):
                if (row < has_dead_latents_pivot.shape[0] and 
                    col < has_dead_latents_pivot.shape[1] and
                    pd.notna(has_dead_latents_pivot.iloc[row, col]) and 
                    has_dead_latents_pivot.iloc[row, col]):
                    # Use hatching pattern instead of gray fill
                    ax.add_patch(plt.Rectangle((col, row), 1, 1, 
                                              fill=True, 
                                              color='#000000',  # Black
                                              alpha=0.2,
                                              hatch='///', 
                                              edgecolor='#000000',  # Black
                                              linewidth=0.5))
        
        # Set labels - show y-label only on leftmost column (j=0 for row 1, j=5 for row 2)
        if j in [0, 5]:
            ax.set_ylabel("Number of Clusters", fontsize=28)
        else:
            ax.set_ylabel("")
        ax.set_xlabel("Layer", fontsize=28)
        ax.tick_params(axis='both', which='major', labelsize=24)

        # Set title with readable display names
        display_names = {
            "deepseek-r1-distill-qwen-1.5b": "Deepseek 1.5B",
            "deepseek-r1-distill-llama-8b": "Deepseek Llama 8B",
            "deepseek-r1-distill-qwen-14b": "Deepseek Qwen 14B",
            "deepseek-r1-distill-qwen-32b": "Deepseek Qwen 32B",
            "qwq-32b": "QwQ 32B",
            "open-reasoner-zero-0.5b": "ORZ 0.5B",
            "open-reasoner-zero-1.5b": "ORZ 1.5B",
            "open-reasoner-zero-7b": "ORZ 7B",
            "open-reasoner-zero-32b": "ORZ 32B",
        }
        display_model_id = display_names.get(model_id, model_id.upper())
        ax.set_title(display_model_id, fontsize=32, pad=10)
    
    # Add a single colorbar
    norm = plt.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Normalized Final Score', fontsize=28)
    cbar.ax.tick_params(labelsize=24)
    
    # Adjust layout to reduce white borders on left and right
    plt.tight_layout(rect=[-0.05, 0, 1.05, 0.98])  
    
    # Save figure with reduced padding
    save_path = os.path.join(output_dir, "sae_grid_search_all_models.pdf")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)

    return fig


def load_full_results(model_id, method="sae_topk", max_reps=None, seed=42):
    """
    Load the full JSON results for a model including all repetitions and statistics.

    Parameters:
    -----------
    model_id : str
        Model identifier
    method : str
        Clustering method (default: "sae_topk")
    max_reps : int or None
        Maximum number of repetitions to use (None = use all)
    seed : int
        Random seed for sampling repetitions

    Returns:
    --------
    dict
        Dictionary mapping layer -> full results data (with sampled repetitions if max_reps set)
    """
    results_dir = 'results/vars'
    results_by_layer = {}

    for filename in os.listdir(results_dir):
        if f"{method}_results_{model_id}_layer" in filename and filename.endswith(".json"):
            try:
                layer_str = filename.split(f"{method}_results_{model_id}_layer")[1]
                layer = int(layer_str.split(".json")[0])

                file_path = os.path.join(results_dir, filename)
                with open(file_path, 'r') as f:
                    results = json.load(f)

                # Sample repetitions if max_reps is set
                if max_reps is not None:
                    for n_clusters, cluster_data in results.get("results_by_cluster_size", {}).items():
                        all_results = cluster_data.get("all_results", [])
                        identifier = f"{model_id}_{layer}_{n_clusters}"
                        sampled_results = sample_repetitions(all_results, max_reps, seed, identifier)
                        cluster_data["all_results"] = sampled_results
                        # Recompute statistics from sampled results
                        cluster_data["statistics"] = compute_statistics_from_results(sampled_results)

                results_by_layer[layer] = results
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    return results_by_layer


def print_all_stats(model_id=None, max_reps=None, seed=42):
    """
    Print all statistics for one or all models to stdout.

    Parameters:
    -----------
    model_id : str or None
        Model identifier, or None to print all models
    max_reps : int or None
        Maximum number of repetitions to use (None = use all)
    seed : int
        Random seed for sampling repetitions
    """
    if model_id is None or model_id == "all":
        model_ids = get_all_model_ids()
    else:
        model_ids = [model_id]

    # Metrics to display
    metrics = ['avg_accuracy', 'avg_f1', 'avg_precision', 'avg_recall',
               'orthogonality', 'semantic_orthogonality_score', 'avg_confidence', 'final_score']

    if max_reps is not None:
        print(f"Using max {max_reps} repetitions (seed={seed})")
        print()

    for model_id in model_ids:
        print("=" * 80)
        print(f"MODEL: {model_id.upper()}")
        print("=" * 80)

        results_by_layer = load_full_results(model_id, max_reps=max_reps, seed=seed)

        if not results_by_layer:
            print(f"  No results found for {model_id}")
            continue

        for layer in sorted(results_by_layer.keys()):
            layer_data = results_by_layer[layer]
            print(f"\n  LAYER {layer}")
            print("  " + "-" * 76)

            results_by_cluster_size = layer_data.get("results_by_cluster_size", {})

            for n_clusters in sorted(results_by_cluster_size.keys(), key=int):
                cluster_data = results_by_cluster_size[n_clusters]
                all_results = cluster_data.get("all_results", [])
                statistics = cluster_data.get("statistics", {})

                print(f"\n    Clusters: {n_clusters} ({len(all_results)} repetitions)")
                print("    " + "-" * 72)

                # Print per-repetition results
                print("\n    Per-repetition results:")
                header = "      Rep  " + "  ".join([f"{m[:12]:>12}" for m in metrics])
                print(header)
                print("      " + "-" * (len(header) - 6))

                for rep_idx, rep_result in enumerate(all_results):
                    values = []
                    for metric in metrics:
                        val = rep_result.get(metric, None)
                        if val is not None:
                            values.append(f"{val:12.4f}")
                        else:
                            values.append(f"{'N/A':>12}")
                    print(f"      {rep_idx + 1:3d}  " + "  ".join(values))

                # Print aggregated statistics
                if statistics:
                    print("\n    Aggregated statistics:")
                    print("      " + "-" * 68)
                    print(f"      {'Metric':<30} {'Mean':>10} {'Std':>10} {'SEM':>10} {'CI_95':>20}")
                    print("      " + "-" * 68)

                    for metric in metrics:
                        if metric in statistics:
                            stats = statistics[metric]
                            mean = stats.get('mean', 0)
                            std = stats.get('std_dev', 0)
                            sem = stats.get('sem', 0)
                            ci_95 = stats.get('ci_95', [0, 0])
                            ci_str = f"[{ci_95[0]:.3f}, {ci_95[1]:.3f}]"
                            print(f"      {metric:<30} {mean:10.4f} {std:10.4f} {sem:10.4f} {ci_str:>20}")

                print()

        print("\n")


if __name__ == "__main__":
    max_reps = args.max_repetitions
    seed = args.seed

    # Generate plots (unless --print-stats only)
    if not args.print_stats:
        if args.model == "all":
            visualize_all_models(max_reps=max_reps, seed=seed)
        else:
            main(args.model.split('/')[-1].lower(), max_reps=max_reps, seed=seed)

    # Always print stats
    print_all_stats(args.model, max_reps=max_reps, seed=seed)
# %%
