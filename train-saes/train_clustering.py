# %%
import numpy as np
import torch
import argparse
import json
import os
from tqdm import tqdm
from utils import utils
import gc
from utils.utils import print_and_flush
from utils.clustering_methods import CLUSTERING_METHODS

# %%

parser = argparse.ArgumentParser(
    description="K-means clustering and autograding of neural activations"
)
parser.add_argument(
    "--model",
    type=str,
    default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    help="Model to analyze",
)
parser.add_argument("--layer", type=int, default=12, help="Layer to analyze")
parser.add_argument(
    "--n_examples", type=int, default=500, help="Number of examples to analyze"
)
parser.add_argument("--clusters", type=int, nargs='+', default=None,
                    help="Specific cluster sizes to process")
parser.add_argument(
    "--load_in_8bit",
    action="store_true",
    default=False,
    help="Load model in 8-bit mode",
)
parser.add_argument(
    "--n_autograder_examples",
    type=int,
    default=100,
    help="Number of examples from each cluster to use for autograding",
)
parser.add_argument(
    "--description_examples",
    type=int,
    default=200,
    help="Number of examples to use for generating cluster descriptions",
)
parser.add_argument(
    "--clustering_methods",
    type=str,
    nargs="+",
    default=[
        "gmm",
        "pca_gmm",
        "spherical_kmeans",
        "pca_kmeans",
        "agglomerative",
        "pca_agglomerative",
        "sae_topk",
    ],
    help="Clustering methods to use",
)
parser.add_argument(
    "--clustering_pilot_size",
    type=int,
    default=50_000,
    help="Number of samples to use for pilot fitting with GMM",
)
parser.add_argument(
    "--clustering_pilot_n_init",
    type=int,
    default=10,
    help="Number of initializations for pilot fitting with GMM",
)
parser.add_argument(
    "--clustering_pilot_max_iter",
    type=int,
    default=100,
    help="Maximum iterations for pilot fitting with GMM",
)
parser.add_argument(
    "--clustering_full_n_init",
    type=int,
    default=1,
    help="Number of initializations for full fitting with GMM",
)
parser.add_argument(
    "--clustering_full_max_iter",
    type=int,
    default=100,
    help="Maximum iterations for full fitting with GMM",
)
args, _ = parser.parse_known_args()

# %%


def run_clustering_experiment(
    clustering_method, clustering_func, activations, args,
):
    """
    Run a clustering experiment using the specified clustering method.
    This version only performs clustering training without evaluation.

    Parameters:
    -----------
    clustering_method : str
        Name of the clustering method
    clustering_func : function
        Function that implements the clustering algorithm
    activations : numpy.ndarray
        Normalized activation vectors
    args : argparse.Namespace
        Command line arguments

    Returns:
    --------
    dict
        Results of the clustering training
    """
    print_and_flush(f"\nRunning {clustering_method.upper()} clustering training...")

    # Define cluster range to test
    assert args.clusters is not None, "Clusters must be specified"
    cluster_range = [int(c) for c in args.clusters]

    print_and_flush(f"Training models for {len(cluster_range)} different cluster counts...")

    # Process each cluster count
    training_results_by_cluster_size = {}
    for n_clusters in tqdm(
        cluster_range, desc=f"{clustering_method.capitalize()} progress"
    ):
        print_and_flush(f"Training model for {n_clusters} clusters...")

        # Perform clustering training only
        cluster_labels, cluster_centers = clustering_func(activations, n_clusters, args)

        # Store basic training information
        training_results_by_cluster_size[n_clusters] = {
            "n_clusters": n_clusters,
            "cluster_centers_shape": cluster_centers.shape,
            "cluster_labels_shape": cluster_labels.shape,
            "trained": True
        }

        print_and_flush(f"Completed training for {n_clusters} clusters")

    print_and_flush(f"Completed training for {clustering_method.upper()}")
    return training_results_by_cluster_size


def create_empty_results_json(clustering_method, model_id, layer, training_results):
    """
    Create a JSON file with empty results structure for title generation script to use.
    
    Parameters:
    -----------
    clustering_method : str
        Name of the clustering method
    model_id : str
        Model identifier
    layer : int
        Layer number
    training_results : dict
        Training results by cluster size
    """
    # Create the results directory if it doesn't exist
    results_dir = f"results/vars"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create the JSON file path
    model_short_name = model_id.split('/')[-1].lower()
    results_json_path = f"{results_dir}/{clustering_method}_results_{model_short_name}_layer{layer}.json"
    
    # Create the basic structure with empty results for each cluster size
    results_data = {
        "clustering_method": clustering_method,
        "model_id": model_id,
        "layer": layer,
        "results_by_cluster_size": {}
    }
    
    # Add empty results for each cluster size that was trained
    for cluster_size, training_info in training_results.items():
        results_data["results_by_cluster_size"][str(cluster_size)] = {
            "all_results": [],  # Empty list to be filled by title generation script
            "avg_final_score": 0.0,
            "statistics": {}
        }
    
    # Save the JSON file
    with open(results_json_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print_and_flush(f"Created empty results JSON at {results_json_path}")


# %% Load model and process activations
print_and_flush("Loading model and processing activations...")
model, tokenizer = utils.load_model(
    model_name=args.model, load_in_8bit=args.load_in_8bit
)

# %% Get model identifier for file naming
model_id = args.model.split("/")[-1].lower()

# %% Process saved responses
all_activations, all_texts = utils.process_saved_responses(
    args.model, args.n_examples, model, tokenizer, args.layer
)

del model, tokenizer
torch.cuda.empty_cache()
gc.collect()

# %% Filter clustering methods based on args
clustering_methods = [
    method for method in args.clustering_methods if method in CLUSTERING_METHODS
]

# Run each clustering method
current_results = {}
for method in clustering_methods:
    try:
        clustering_func = CLUSTERING_METHODS[method]
        results = run_clustering_experiment(
            method, clustering_func, all_activations, args
        )
        current_results[method] = results
        
        # Create JSON file with empty results structure for title generation script
        create_empty_results_json(method, args.model, args.layer, results)
        
        print_and_flush(f"Successfully completed training for {method}")
    except Exception as e:
        print_and_flush(f"Error running {method}: {e}")
        import traceback

        print(traceback.format_exc())
