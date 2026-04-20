#!/usr/bin/env python3

import argparse
import json
import os
from utils.utils import print_and_flush
from utils.clustering_batched import check_batch_status
from utils.clustering import (
    load_trained_clustering_data, predict_clusters,
    generate_representative_examples,
    generate_cluster_descriptions
)
from utils.clustering_batched import (
    generate_cluster_descriptions_batch, process_cluster_descriptions_batch
)
from utils import utils
import numpy as np
import gc
import torch
import time
import random


def is_valid_cluster_description(title, description):
    """Check if a cluster description is valid (not a placeholder)."""
    return title != "Unnamed Cluster" and description != "No description available"


def get_missing_clusters(existing_results, cluster_size, rep_idx, n_clusters):
    """
    Determine which clusters need descriptions for a given cluster size and repetition.

    Returns:
    --------
    list of int : cluster indices that need descriptions
    """
    cluster_results = existing_results.get("results_by_cluster_size", {}).get(str(cluster_size), {})
    all_results = cluster_results.get("all_results", [])

    # If we don't have this repetition yet, all clusters need generation
    if rep_idx >= len(all_results):
        return list(range(n_clusters))

    rep_data = all_results[rep_idx]
    categories = rep_data.get("categories", [])

    # If no categories, all clusters need generation
    if not categories:
        return list(range(n_clusters))

    # Check which clusters have invalid/missing descriptions
    missing_clusters = []
    existing_clusters_dict = {int(cat[0]): (cat[1], cat[2]) for cat in categories}

    for cluster_idx in range(n_clusters):
        if cluster_idx not in existing_clusters_dict:
            # Cluster not in results at all
            missing_clusters.append(cluster_idx)
        else:
            title, description = existing_clusters_dict[cluster_idx]
            if not is_valid_cluster_description(title, description):
                # Cluster has placeholder description
                missing_clusters.append(cluster_idx)

    return missing_clusters


# %%

parser = argparse.ArgumentParser(description="Generate cluster titles using batch API")
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                    help="Model to analyze")
parser.add_argument("--layer", type=int, default=12,
                    help="Layer to analyze")
parser.add_argument("--n_examples", type=int, default=500,
                    help="Number of examples to analyze")
parser.add_argument("--clustering_methods", type=str, nargs='+', 
                    default=["gmm", "pca_gmm", "spherical_kmeans", "pca_kmeans", "agglomerative", "pca_agglomerative", "sae_topk"],
                    help="Clustering methods to evaluate")
parser.add_argument("--load_in_8bit", action="store_true", default=False,
                    help="Load model in 8-bit mode")
parser.add_argument("--description_examples", type=int, default=200,
                    help="Number of examples to use for generating cluster descriptions")
parser.add_argument("--evaluator_model", type=str, default="o4-mini",
                    help="Model to use for generating descriptions")
parser.add_argument("--command", type=str, choices=["submit", "process", "direct"], required=True,
                    help="Command to run: submit batch jobs, process results, or generate directly")
parser.add_argument("--batch_file", type=str, default=None,
                    help="JSON file containing batch information (for process command)")
parser.add_argument("--wait-batch-completion", action="store_true", default=False,
                    help="If set, wait for all batches to complete, checking every minute. Otherwise, check once and exit if not complete.")
parser.add_argument("--repetitions", type=int, default=3,
                    help="Number of repetitions for generating different category sets")
parser.add_argument("--clusters", type=int, nargs='+', default=None,
                    help="Specific cluster sizes to process (if None, process all available cluster sizes)")
parser.add_argument("--n_trace_examples", type=int, default=3,
                    help="Number of full reasoning trace examples to include in prompts")
parser.add_argument("--n_categories_examples", type=int, default=5,
                    help="Number of category examples to include in prompts")

args, _ = parser.parse_known_args()


def create_empty_results_json(clustering_method, model_id, layer, clusters):
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
    clusters : list
        List of cluster sizes to create empty results for
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
    
    # Add empty results for each cluster size
    for cluster_size in clusters:
        results_data["results_by_cluster_size"][str(cluster_size)] = {
            "all_results": [],  # Empty list to be filled by title generation script
            "avg_final_score": 0.0,
            "statistics": {}
        }
    
    # Save the JSON file
    with open(results_json_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print_and_flush(f"Created empty results JSON at {results_json_path}")
    return results_data


def _prepare_cluster_examples(n_clusters, representative_examples, description_examples, cluster_indices=None):
    """
    Prepares a list of examples for each cluster, sampled for diversity.

    Parameters:
    -----------
    cluster_indices : list of int, optional
        If provided, only prepare examples for these cluster indices.
        If None, prepare for all clusters.
    """
    if cluster_indices is None:
        cluster_indices = range(n_clusters)

    cluster_examples_list = []
    for cluster_idx in cluster_indices:
        print_and_flush(f"Preparing examples for cluster {cluster_idx}:")

        if len(representative_examples[cluster_idx]) == 0:
            print_and_flush(f"WARNING: Skipping empty cluster {cluster_idx}")
            continue

        cluster_examples = representative_examples[cluster_idx]
        total_examples = len(cluster_examples)

        if total_examples <= description_examples:
            examples = cluster_examples
        else:
            # Pick half from the top, half randomly from the rest
            n_top = description_examples // 2
            n_random = description_examples - n_top

            top_examples = cluster_examples[:n_top]
            print_and_flush(f"Top examples:")
            for example in top_examples:
                print_and_flush(f"  {example}")

            remaining_examples = cluster_examples[n_top:]

            if len(remaining_examples) < n_random:
                random_examples = remaining_examples
            else:
                random_examples = random.sample(remaining_examples, n_random)

            print_and_flush(f"Random examples:")
            for example in random_examples:
                print_and_flush(f"  {example}")

            examples = top_examples + random_examples

        # Shuffle examples for diversity
        random.shuffle(examples)
        cluster_examples_list.append((cluster_idx, examples))
    return cluster_examples_list


def submit_description_batches():
    """Submit batch jobs for generating cluster descriptions."""
    print_and_flush("=== SUBMITTING CLUSTER DESCRIPTION BATCHES ===")
    
    # Get model identifier for file naming
    model_id = args.model.split('/')[-1].lower()
    
    # Load model and process activations
    print_and_flush("Loading model and processing activations...")
    model, tokenizer = utils.load_model(
        model_name=args.model,
        load_in_8bit=args.load_in_8bit
    )

    # Process saved responses
    all_activations, all_texts = utils.process_saved_responses(
        args.model, 
        args.n_examples,
        model,
        tokenizer,
        args.layer
    )

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    batch_info = {}
    
    # Process each clustering method
    for method in args.clustering_methods:
        print_and_flush(f"\n=== Processing {method.upper()} ===")
        
        # Load existing results to get cluster sizes
        results_json_path = f'results/vars/{method}_results_{model_id}_layer{args.layer}.json'
        if not os.path.exists(results_json_path):
            print_and_flush(f"No existing results found for {method} at {results_json_path}.")
            # Create empty results JSON if clusters are specified
            if args.clusters is not None:
                print_and_flush(f"Creating empty results file for {method} with cluster sizes {args.clusters}")
                existing_results = create_empty_results_json(method, args.model, args.layer, args.clusters)
            else:
                print_and_flush(f"No cluster sizes specified. Skipping {method}.")
                continue
        else:
            with open(results_json_path, 'r') as f:
                existing_results = json.load(f)
        
        clusters = list(existing_results.get("results_by_cluster_size", {}).keys())
        if not clusters:
            print_and_flush(f"No clustering results found for {method}. Skipping.")
            continue
            
        # Filter cluster sizes if specified
        if args.clusters is not None:
            requested_sizes = [str(size) for size in args.clusters]
            clusters = [size for size in clusters if size in requested_sizes]
            if not clusters:
                print_and_flush(f"None of the requested cluster sizes {args.clusters} found for {method}. Skipping.")
                continue
            
        method_batches = {}
        
        # Process each cluster size
        for cluster_size in clusters:
            n_clusters = int(cluster_size)
            print_and_flush(f"Submitting batch for {method} with {n_clusters} clusters...")
            
            try:
                # Load the trained clustering model
                clustering_data = load_trained_clustering_data(model_id, args.layer, n_clusters, method)
                cluster_centers = clustering_data['cluster_centers']
                
                # Predict cluster labels for current activations
                if method == 'sae_topk':
                    cluster_labels = predict_clusters(all_activations, clustering_data, model_id, args.layer, n_clusters)
                else:
                    cluster_labels = predict_clusters(all_activations, clustering_data)
                
                # Generate representative examples
                representative_examples = generate_representative_examples(
                    cluster_centers, all_texts, cluster_labels, all_activations, 
                    clustering_data=clustering_data, model_id=model_id, layer=args.layer, n_clusters=n_clusters
                )
                
                # Submit batches for multiple repetitions (different category sets)
                cluster_size_batches = {}

                # Load existing results to check which clusters need generation
                results_json_path = f'results/vars/{method}_results_{model_id}_layer{args.layer}.json'
                if os.path.exists(results_json_path):
                    with open(results_json_path, 'r') as f:
                        existing_results_for_check = json.load(f)
                else:
                    existing_results_for_check = existing_results

                for rep_idx in range(args.repetitions):
                    # Check which clusters need generation for this repetition
                    missing_clusters = get_missing_clusters(
                        existing_results_for_check, cluster_size, rep_idx, n_clusters
                    )

                    if not missing_clusters:
                        print_and_flush(f"  Repetition {rep_idx + 1}/{args.repetitions}: All clusters already have valid descriptions. Skipping.")
                        continue

                    print_and_flush(f"  Submitting repetition {rep_idx + 1}/{args.repetitions}...")
                    print_and_flush(f"    Generating descriptions for {len(missing_clusters)} clusters: {missing_clusters}")

                    # Prepare examples only for missing clusters
                    cluster_examples_list = _prepare_cluster_examples(
                        n_clusters, representative_examples, args.description_examples,
                        cluster_indices=missing_clusters
                    )

                    if not cluster_examples_list:
                        print_and_flush(f"    No valid examples to submit for repetition {rep_idx + 1}. Skipping.")
                        continue

                    # Submit batch for this repetition
                    batch_id, cluster_indices = generate_cluster_descriptions_batch(
                        args.model, cluster_examples_list, model=args.evaluator_model,
                        n_trace_examples=args.n_trace_examples,
                        n_categories_examples=args.n_categories_examples
                    )

                    cluster_size_batches[f"rep_{rep_idx}"] = {
                        "batch_id": batch_id,
                        "cluster_indices": cluster_indices,
                        "n_clusters": n_clusters,
                        "method": method,
                        "missing_clusters": missing_clusters  # Track which clusters this batch is for
                    }

                    print_and_flush(f"  Submitted batch {batch_id} for repetition {rep_idx + 1}")

                if cluster_size_batches:
                    method_batches[cluster_size] = cluster_size_batches
                else:
                    print_and_flush(f"No batches needed for {method} with {n_clusters} clusters - all descriptions already exist")
                
                print_and_flush(f"Submitted all repetition batches for {method} with {n_clusters} clusters")
                
            except Exception as e:
                print_and_flush(f"Error processing {method} with {n_clusters} clusters: {e}")
                import traceback
                traceback.print_exc()
                continue

        if method_batches:
            batch_info[method] = method_batches
        else:
            print_and_flush(f"No batches submitted for {method} - all descriptions already exist")

    # Save batch information only if there are batches
    if batch_info:
        batch_info_file = f"batch_info_titles_{model_id}_layer{args.layer}.json"
        with open(batch_info_file, 'w') as f:
            json.dump(batch_info, f, indent=2)

        print_and_flush(f"\nBatch information saved to {batch_info_file}")
        print_and_flush("All batches submitted successfully!")
    else:
        print_and_flush("\nNo batches needed - all cluster descriptions already exist!")


def process_description_batches():
    """Process completed batch jobs and update results with cluster descriptions."""
    print_and_flush("=== PROCESSING CLUSTER DESCRIPTION BATCHES ===")
    
    # Get model identifier for file naming
    model_id = args.model.split('/')[-1].lower()
    
    # Load batch information
    if args.batch_file:
        batch_info_file = args.batch_file
    else:
        batch_info_file = f"batch_info_titles_{model_id}_layer{args.layer}.json"
    
    if not os.path.exists(batch_info_file):
        print_and_flush(f"Batch information file {batch_info_file} not found!")
        return
    
    with open(batch_info_file, 'r') as f:
        batch_info = json.load(f)
    
    # Check status of all batches first
    print_and_flush("Checking batch status...")
    while True:
        all_completed = True
        for method, method_batches in batch_info.items():
            for cluster_size, cluster_data in method_batches.items():
                for rep_key, rep_data in cluster_data.items():
                    if not rep_key.startswith("rep_"):
                        continue
                    batch_id = rep_data["batch_id"]
                    status = check_batch_status(batch_id)
                    print_and_flush(f"{method} {cluster_size} {rep_key}: {batch_id} -> {status}")
                    if status not in ["completed", "expired", "cancelled"]:
                        all_completed = False
        
        if all_completed:
            print_and_flush("All batches are completed. Processing...")
            break
        
        if args.wait_batch_completion:
            print_and_flush("Not all batches are completed. Waiting for 1 minute before re-checking.")
            time.sleep(60)
        else:
            print_and_flush("Not all batches are completed. Exiting. Use --wait-batch-completion to wait.")
            return

    # Process each method's batches
    for method, method_batches in batch_info.items():
        print_and_flush(f"\n=== Processing {method.upper()} ===")
        
        # Load existing results
        results_json_path = f'results/vars/{method}_results_{model_id}_layer{args.layer}.json'
        if not os.path.exists(results_json_path):
            print_and_flush(f"No existing results found for {method} at {results_json_path}.")
            # Create empty results JSON if clusters are specified
            if args.clusters is not None:
                print_and_flush(f"Creating empty results file for {method} with cluster sizes {args.clusters}")
                existing_results = create_empty_results_json(method, args.model, args.layer, args.clusters)
            else:
                print_and_flush(f"No cluster sizes specified. Skipping {method}.")
                continue
        else:
            with open(results_json_path, 'r') as f:
                existing_results = json.load(f)
        
        # Filter cluster sizes if specified
        if args.clusters is not None:
            requested_sizes = [str(size) for size in args.clusters]
            method_batches = {size: data for size, data in method_batches.items() if size in requested_sizes}
            if not method_batches:
                print_and_flush(f"None of the requested cluster sizes {args.clusters} found for {method}. Skipping.")
                continue
        
        # Process each cluster size
        for cluster_size, cluster_data in method_batches.items():
            n_clusters = None

            print_and_flush(f"Processing batches for {cluster_size} clusters...")

            # Get existing cluster results to preserve valid descriptions
            cluster_results = existing_results.get("results_by_cluster_size", {}).get(cluster_size, {})
            all_results = cluster_results.get("all_results", [])

            # Ensure we have enough repetitions in the results
            while len(all_results) < args.repetitions:
                all_results.append({"categories": []})

            # Process each repetition
            for rep_idx in range(args.repetitions):
                rep_key = f"rep_{rep_idx}"
                if rep_key not in cluster_data:
                    print_and_flush(f"  Repetition {rep_idx + 1}: No batch submitted (already had valid descriptions). Skipping.")
                    continue

                rep_data = cluster_data[rep_key]
                batch_id = rep_data["batch_id"]
                cluster_indices = rep_data["cluster_indices"]
                missing_clusters = rep_data.get("missing_clusters", cluster_indices)
                if n_clusters is None:
                    n_clusters = rep_data["n_clusters"]

                print_and_flush(f"  Processing repetition {rep_idx + 1} batch {batch_id}...")
                print_and_flush(f"    This batch covers clusters: {missing_clusters}")

                # Check batch status before processing
                status = check_batch_status(batch_id)
                if status not in ["completed", "expired", "cancelled"]:
                    print_and_flush(f"  Batch {batch_id} not completed (status: {status}). Skipping.")
                    continue

                # Process batch results
                new_categories = process_cluster_descriptions_batch(batch_id, cluster_indices)

                if new_categories:
                    print_and_flush(f"  Generated descriptions for {len(new_categories)} clusters:")
                    for cluster_id, title, description in new_categories:
                        print_and_flush(f"    Cluster {cluster_id}: {title}")
                        print_and_flush(f"      {description}")

                    # Merge new categories with existing ones for this repetition
                    existing_categories = all_results[rep_idx].get("categories", [])

                    # Create dict of existing categories
                    existing_dict = {int(cat[0]): cat for cat in existing_categories}

                    # Update with new categories (only for the clusters in this batch)
                    new_dict = {int(cat[0]): cat for cat in new_categories}
                    existing_dict.update(new_dict)

                    # Convert back to list, sorted by cluster ID
                    merged_categories = [existing_dict[i] for i in sorted(existing_dict.keys())]

                    all_results[rep_idx]["categories"] = merged_categories
                    print_and_flush(f"  Successfully updated repetition {rep_idx + 1} with {len(new_categories)} new descriptions")
                else:
                    print_and_flush(f"  No categories generated for batch {batch_id}. It may have failed or had no content.")

            # Update existing results
            if "results_by_cluster_size" not in existing_results:
                existing_results["results_by_cluster_size"] = {}

            cluster_results["all_results"] = all_results
            existing_results["results_by_cluster_size"][cluster_size] = cluster_results

            print_and_flush(f"Completed processing for {cluster_size} clusters")
        
        # Save updated results
        with open(results_json_path, 'w') as f:
            json.dump(existing_results, f, indent=2)
        print_and_flush(f"Updated results saved to {results_json_path}")
    
    print_and_flush("All batch results processed successfully!")


def generate_descriptions_direct():
    """Generate cluster descriptions directly without using batch API."""
    print_and_flush("=== GENERATING CLUSTER DESCRIPTIONS DIRECTLY ===")
    
    # Get model identifier for file naming
    model_id = args.model.split('/')[-1].lower()
    
    # Load model and process activations
    print_and_flush("Loading model and processing activations...")
    model, tokenizer = utils.load_model(
        model_name=args.model,
        load_in_8bit=args.load_in_8bit
    )

    # Process saved responses
    all_activations, all_texts = utils.process_saved_responses(
        args.model, 
        args.n_examples,
        model,
        tokenizer,
        args.layer
    )

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    # Process each clustering method
    for method in args.clustering_methods:
        print_and_flush(f"\n=== Processing {method.upper()} ===")
        
        # Load existing results to get cluster sizes
        results_json_path = f'results/vars/{method}_results_{model_id}_layer{args.layer}.json'
        if not os.path.exists(results_json_path):
            print_and_flush(f"No existing results found for {method} at {results_json_path}.")
            # Create empty results JSON if clusters are specified
            if args.clusters is not None:
                print_and_flush(f"Creating empty results file for {method} with cluster sizes {args.clusters}")
                existing_results = create_empty_results_json(method, args.model, args.layer, args.clusters)
            else:
                print_and_flush(f"No cluster sizes specified. Skipping {method}.")
                continue
        else:
            with open(results_json_path, 'r') as f:
                existing_results = json.load(f)
        
        clusters = list(existing_results.get("results_by_cluster_size", {}).keys())
        if not clusters:
            print_and_flush(f"No clustering results found for {method}. Skipping.")
            continue
            
        # Filter cluster sizes if specified
        if args.clusters is not None:
            requested_sizes = [str(size) for size in args.clusters]
            clusters = [size for size in clusters if size in requested_sizes]
            if not clusters:
                print_and_flush(f"None of the requested cluster sizes {args.clusters} found for {method}. Skipping.")
                continue
        
        # Process each cluster size
        for cluster_size in clusters:
            n_clusters = int(cluster_size)
            print_and_flush(f"Generating descriptions for {method} with {n_clusters} clusters...")
            
            try:
                # Load the trained clustering model
                clustering_data = load_trained_clustering_data(model_id, args.layer, n_clusters, method)
                cluster_centers = clustering_data['cluster_centers']
                
                # Predict cluster labels for current activations
                if method == 'sae_topk':
                    cluster_labels = predict_clusters(all_activations, clustering_data, model_id, args.layer, n_clusters)
                else:
                    cluster_labels = predict_clusters(all_activations, clustering_data)
                
                # Generate representative examples
                representative_examples = generate_representative_examples(
                    cluster_centers, all_texts, cluster_labels, all_activations, 
                    clustering_data=clustering_data, model_id=model_id, layer=args.layer, n_clusters=n_clusters
                )
                
                # Get existing cluster results to preserve valid descriptions
                cluster_results = existing_results.get("results_by_cluster_size", {}).get(cluster_size, {})
                all_results = cluster_results.get("all_results", [])

                # Ensure we have enough repetitions in the results
                while len(all_results) < args.repetitions:
                    all_results.append({"categories": []})

                # Generate descriptions for multiple repetitions
                for rep_idx in range(args.repetitions):
                    # Check which clusters need generation for this repetition
                    missing_clusters = get_missing_clusters(
                        existing_results, cluster_size, rep_idx, n_clusters
                    )

                    if not missing_clusters:
                        print_and_flush(f"  Repetition {rep_idx + 1}/{args.repetitions}: All clusters already have valid descriptions. Skipping.")
                        continue

                    print_and_flush(f"  Generating repetition {rep_idx + 1}/{args.repetitions}...")
                    print_and_flush(f"    Generating descriptions for {len(missing_clusters)} clusters: {missing_clusters}")

                    # Prepare examples only for missing clusters
                    cluster_examples_list = _prepare_cluster_examples(
                        n_clusters, representative_examples, args.description_examples,
                        cluster_indices=missing_clusters
                    )

                    if not cluster_examples_list:
                        print_and_flush(f"    No valid examples to generate for repetition {rep_idx + 1}. Skipping.")
                        continue

                    # Generate cluster descriptions directly
                    new_categories = generate_cluster_descriptions(
                        args.model,
                        cluster_examples_list,
                        args.evaluator_model,
                        n_trace_examples=args.n_trace_examples,
                        n_categories_examples=args.n_categories_examples
                    )

                    print_and_flush(f"  Generated descriptions for {len(new_categories)} clusters:")
                    for cluster_id, title, description in new_categories:
                        print_and_flush(f"    Cluster {cluster_id}: {title}")
                        print_and_flush(f"      {description}")

                    # Merge new categories with existing ones for this repetition
                    existing_categories = all_results[rep_idx].get("categories", [])

                    # Create dict of existing categories
                    existing_dict = {int(cat[0]): cat for cat in existing_categories}

                    # Update with new categories (only for the clusters we just generated)
                    new_dict = {int(cat[0]): cat for cat in new_categories}
                    existing_dict.update(new_dict)

                    # Convert back to list, sorted by cluster ID
                    merged_categories = [existing_dict[i] for i in sorted(existing_dict.keys())]

                    all_results[rep_idx]["categories"] = merged_categories
                    print_and_flush(f"  Successfully completed repetition {rep_idx + 1} with {len(new_categories)} new descriptions")

                # Update existing results
                if "results_by_cluster_size" not in existing_results:
                    existing_results["results_by_cluster_size"] = {}

                cluster_results["all_results"] = all_results
                existing_results["results_by_cluster_size"][cluster_size] = cluster_results

                print_and_flush(f"Completed processing for {cluster_size} clusters")
                
            except Exception as e:
                print_and_flush(f"Error processing {method} with {n_clusters} clusters: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save updated results
        with open(results_json_path, 'w') as f:
            json.dump(existing_results, f, indent=2)
        print_and_flush(f"Updated results saved to {results_json_path}")
    
    print_and_flush("All cluster descriptions generated successfully!")


def main():
    if args.command == "submit":
        submit_description_batches()
    elif args.command == "process":
        process_description_batches()
    elif args.command == "direct":
        generate_descriptions_direct()


if __name__ == "__main__":
    main()
