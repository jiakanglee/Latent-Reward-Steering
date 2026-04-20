import argparse
import json
import os
import glob
from tqdm import tqdm
import traceback

from utils.clustering import (
    load_trained_clustering_data,
    predict_clusters,
    generate_representative_examples,
)
from utils import utils
from utils.utils import print_and_flush

def main(args):
    """
    Main function to add representative examples to SAE clustering results.
    """
    # Find all sae_topk result files
    results_dir = "results/vars"
    files = glob.glob(os.path.join(results_dir, "sae_topk_results_*.json"))
    print_and_flush(f"Found {len(files)} SAE result files to process.")

    # Group files by model_id to avoid reloading models
    model_files = {}
    for f in files:
        if "llama-70b" in f:
            continue
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
            model_id = data.get('model_id')
            if model_id:
                if model_id not in model_files:
                    model_files[model_id] = []
                model_files[model_id].append(f)
            else:
                print_and_flush(f"Warning: 'model_id' not found in {f}. Skipping.")
        except json.JSONDecodeError:
            print_and_flush(f"Warning: Could not decode JSON from {f}. Skipping.")
        except Exception as e:
            print_and_flush(f"An error occurred while processing {f}: {e}")


    for model_id, file_list in model_files.items():
        print_and_flush(f"Processing model: {model_id}")
        
        # Cache activations by layer
        activations_cache = {}

        for file_path in file_list:
            print_and_flush(f"  Processing file: {file_path}")
            try:
                with open(file_path, 'r') as f:
                    results_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print_and_flush(f"  Could not read {file_path}. Error: {e}. Skipping.")
                continue
            
            layer = results_data.get('layer')
            if layer is None:
                print_and_flush(f"  'layer' not found in {file_path}. Skipping.")
                continue

            # Get activations and texts, use cache
            if layer not in activations_cache:
                print_and_flush(f"    Loading activations for layer {layer}...")
                try:
                    all_activations, all_texts = utils.process_saved_responses(
                        model_id,
                        args.n_examples,
                        None,
                        None,
                        layer
                    )
                    activations_cache[layer] = (all_activations, all_texts)
                except Exception as e:
                    print_and_flush(f"    Failed to process responses for layer {layer}: {e}")
                    print(traceback.format_exc())
                    continue
            else:
                print_and_flush(f"    Using cached activations for layer {layer}")
                all_activations, all_texts = activations_cache[layer]

            # process cluster sizes
            if "results_by_cluster_size" not in results_data:
                print_and_flush(f"  'results_by_cluster_size' not found in {file_path}. Skipping.")
                continue

            for n_clusters_str, cluster_size_results in tqdm(results_data["results_by_cluster_size"].items(), desc="  Cluster sizes", leave=False):
                try:
                    n_clusters = int(n_clusters_str)
                    model_name = model_id.split('/')[-1].lower()

                    clustering_data = load_trained_clustering_data(model_name, layer, n_clusters, 'sae_topk')
                    cluster_centers = clustering_data['cluster_centers']
                    
                    cluster_labels = predict_clusters(all_activations, clustering_data, model_name, layer, n_clusters)
                    
                    representative_examples = generate_representative_examples(
                        cluster_centers, all_texts, cluster_labels, all_activations,
                        clustering_data=clustering_data, model_id=model_name, layer=layer, n_clusters=n_clusters
                    )

                    # Update JSON data with a new 'examples' key for the cluster size
                    examples_by_cluster = {}
                    for cluster_idx, examples in representative_examples.items():
                        examples_by_cluster[str(cluster_idx)] = examples[:15]
                    cluster_size_results['examples'] = examples_by_cluster

                except Exception as e:
                    print_and_flush(f"    Error processing {n_clusters_str} clusters: {e}")
                    print(traceback.format_exc())

            # Save updated data
            try:
                with open(file_path, 'w') as f:
                    json.dump(results_data, f, indent=2)
                print_and_flush(f"  Updated {file_path}")
            except Exception as e:
                print_and_flush(f"  Failed to write updated results to {file_path}: {e}")
                print(traceback.format_exc())

    print_and_flush("Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add representative examples to SAE results files.")
    parser.add_argument("--n_examples", type=int, default=100000, help="Number of examples to analyze for activations")
    parser.add_argument("--load_in_8bit", action="store_true", default=False, help="Load model in 8-bit mode")
    args = parser.parse_args()
    main(args)
