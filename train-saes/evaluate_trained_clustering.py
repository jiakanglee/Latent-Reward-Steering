# %%
import numpy as np
import argparse
import json
import os
from tqdm import tqdm
import torch
import gc
import time
from utils.utils import print_and_flush
from utils.clustering_batched import check_batch_status
from utils.clustering import (
    SUPPORTED_CLUSTERING_METHODS,
    load_trained_clustering_data, predict_clusters, save_clustering_results, compute_centroid_orthogonality,
    evaluate_clustering_scoring_metrics
)
from utils.clustering_batched import (
    accuracy_autograder_batch, process_accuracy_batch,
    completeness_autograder_batch, process_completeness_batch,
    compute_semantic_orthogonality_batch, process_semantic_orthogonality_batch
)
from utils import utils
import random

# %%

parser = argparse.ArgumentParser(description="Evaluate saved clustering models using batch API")
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                    help="Model to analyze")
parser.add_argument("--layer", type=int, default=12,
                    help="Layer to analyze")
parser.add_argument("--n_examples", type=int, default=500,
                    help="Number of examples to analyze")
parser.add_argument("--clustering_methods", type=str, nargs='+', 
                    default=list(SUPPORTED_CLUSTERING_METHODS),
                    help="Clustering methods to evaluate")
parser.add_argument("--load_in_8bit", action="store_true", default=False,
                    help="Load model in 8-bit mode")
parser.add_argument("--n_autograder_examples", type=int, default=200,
                    help="Number of examples from each cluster to use for autograding")
parser.add_argument("--n_completeness_examples", type=int, default=200,
                    help="Number of examples to use for completeness evaluation")
parser.add_argument("--description_examples", type=int, default=200,
                    help="Number of examples to use for generating cluster descriptions")
parser.add_argument("--evaluator_model", type=str, default="gpt-4.1-mini",
                    help="Model to use for evaluations")
parser.add_argument("--command", type=str, choices=["submit", "process", "direct"], required=True,
                    help="Command to run: submit batch jobs, process results, or evaluate directly")
parser.add_argument("--batch_file", type=str, default=None,
                    help="JSON file containing batch information (for process command)")
parser.add_argument("--wait-batch-completion", action="store_true", default=False,
                    help="If set, wait for all batches to complete, checking every minute. Otherwise, check once and exit if not complete.")
parser.add_argument("--repetitions", type=int, default=3,
                    help="Number of repetitions for evaluation")
parser.add_argument("--clusters", type=int, nargs='+', default=None,
                    help="Specific cluster sizes to process (if None, process all available cluster sizes)")
parser.add_argument("--no-accuracy", action="store_true", default=False, help="Disable accuracy evaluation and use existing results.")
parser.add_argument("--no-completeness", action="store_true", default=False, help="Disable completeness evaluation and use existing results.")
parser.add_argument("--no-sem-orth", action="store_true", default=False, help="Disable semantic orthogonality evaluation and use existing results.")
parser.add_argument("--no-orth", action="store_true", default=False, help="Disable centroid orthogonality evaluation and use existing results.")
parser.add_argument("--accuracy_target_cluster_percentage", type=float, default=0.2, help="Percentage of examples to take from target cluster for accuracy evaluation (default: 0.2)")
args, _ = parser.parse_known_args()

# %% Get model identifier for file naming
model_id = args.model.split('/')[-1].lower()

clustering_methods = [method for method in args.clustering_methods if method in SUPPORTED_CLUSTERING_METHODS]

def submit_evaluation_batches():
    """Submit batch jobs for evaluating clustering methods."""
    print_and_flush("=== SUBMITTING CLUSTERING EVALUATION BATCHES ===")
    
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
    
    # Filter clustering methods based on args
    clustering_methods = [method for method in args.clustering_methods if method in SUPPORTED_CLUSTERING_METHODS]
    
    # Process each clustering method
    for method in clustering_methods:
        print_and_flush(f"\n=== Processing {method.upper()} ===")
        
        # Load existing results to get cluster sizes and categories
        results_json_path = f'results/vars/{method}_results_{model_id}_layer{args.layer}.json'
        if not os.path.exists(results_json_path):
            print_and_flush(f"No existing results found for {method} at {results_json_path}. Skipping.")
            continue
            
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
            print_and_flush(f"Submitting batches for {method} with {n_clusters} clusters...")
            
            try:
                # Load the trained clustering model
                clustering_data = load_trained_clustering_data(model_id, args.layer, n_clusters, method)
                cluster_centers = clustering_data['cluster_centers']
                
                # Predict cluster labels for current activations (always recompute for consistency)
                if method == 'sae_topk':
                    cluster_labels = predict_clusters(all_activations, clustering_data, model_id, args.layer, n_clusters)
                else:
                    cluster_labels = predict_clusters(all_activations, clustering_data)
                
                # Get all categories from existing results (one set per repetition)
                cluster_results = existing_results["results_by_cluster_size"].get(cluster_size, {})
                all_categories = []
                if "all_results" in cluster_results and len(cluster_results["all_results"]) > 0:
                    for repetition_data in cluster_results["all_results"]:
                        if "categories" in repetition_data:
                            all_categories.append(repetition_data["categories"])
                
                if not all_categories:
                    print_and_flush(f"No categories found for {method} with {n_clusters} clusters. Skipping evaluation.")
                    continue
                
                # Ensure we have enough category sets for the number of repetitions
                if len(all_categories) < args.repetitions:
                    print_and_flush(f"Only {len(all_categories)} category sets found, but {args.repetitions} repetitions requested. Using available categories.")
                    # Repeat the last category set if we don't have enough
                    while len(all_categories) < args.repetitions:
                        all_categories.append(all_categories[-1])
                
                cluster_size_batches = {}
                
                # Submit batches for multiple repetitions, each with its own categories
                for rep_idx in range(args.repetitions):
                    categories = all_categories[rep_idx]  # Use categories specific to this repetition
                    print_and_flush(f"  Submitting repetition {rep_idx + 1}/{args.repetitions} with {len(categories)} categories...")
                    
                    rep_batches = {}
                    
                    # Submit accuracy evaluation batch
                    if not args.no_accuracy:
                        str_cluster_labels = [str(label) for label in cluster_labels]
                        acc_batch_id, acc_metadata = accuracy_autograder_batch(
                            all_texts, categories, str_cluster_labels, args.n_autograder_examples,
                            model=args.evaluator_model, target_cluster_percentage=args.accuracy_target_cluster_percentage
                        )
                        rep_batches["accuracy"] = {
                            "batch_id": acc_batch_id,
                            "metadata": acc_metadata
                        }
                    else:
                        rep_batches["accuracy"] = {"batch_id": None, "metadata": {}}
                    
                    # Submit completeness evaluation batch  
                    # Sample texts for completeness evaluation if needed
                    if not args.no_completeness:
                        if len(all_texts) > args.n_completeness_examples:
                            sample_indices = random.sample(range(len(all_texts)), args.n_completeness_examples)
                            completeness_texts = [all_texts[i] for i in sample_indices]
                            completeness_labels = [str_cluster_labels[i] for i in sample_indices]
                        else:
                            completeness_texts = all_texts
                            completeness_labels = str_cluster_labels
                        
                        comp_batch_id, comp_metadata = completeness_autograder_batch(
                            completeness_texts, completeness_labels, categories, 
                            model=args.evaluator_model
                        )
                        rep_batches["completeness"] = {
                            "batch_id": comp_batch_id,
                            "metadata": comp_metadata
                        }
                    else:
                        rep_batches["completeness"] = {"batch_id": None, "metadata": {}}
                    
                    # Submit semantic orthogonality batch
                    if not args.no_sem_orth:
                        sem_batch_id, sem_metadata = compute_semantic_orthogonality_batch(
                            categories, model=args.evaluator_model
                        )
                        rep_batches["semantic_orthogonality"] = {
                            "batch_id": sem_batch_id,
                            "metadata": sem_metadata
                        }
                    else:
                        rep_batches["semantic_orthogonality"] = {"batch_id": None, "metadata": {}}
                    
                    # Store categories used for this repetition
                    rep_batches["categories"] = categories
                    cluster_size_batches[f"rep_{rep_idx}"] = rep_batches
                
                # Store other data needed for processing
                cluster_size_batches["cluster_centers"] = cluster_centers.tolist()
                cluster_size_batches["n_clusters"] = n_clusters
                cluster_size_batches["method"] = method
                
                method_batches[cluster_size] = cluster_size_batches
                
                print_and_flush(f"Submitted all batches for {method} with {n_clusters} clusters")
                
            except Exception as e:
                print_and_flush(f"Error processing {method} with {n_clusters} clusters: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        batch_info[method] = method_batches
    
    # Save batch information
    batch_info_file = f"batch_info_eval_{model_id}_layer{args.layer}.json"
    with open(batch_info_file, 'w') as f:
        json.dump(batch_info, f, indent=2)
    
    print_and_flush(f"\nBatch information saved to {batch_info_file}")
    print_and_flush("All evaluation batches submitted successfully!")


def process_evaluation_batches():
    """Process completed batch jobs and update results with evaluation metrics."""
    print_and_flush("=== PROCESSING CLUSTERING EVALUATION BATCHES ===")
    
    # Load batch information
    if args.batch_file:
        batch_info_file = args.batch_file
    else:
        batch_info_file = f"batch_info_eval_{model_id}_layer{args.layer}.json"
    
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
                    for eval_type, batch_data in rep_data.items():
                        if isinstance(batch_data, dict) and "batch_id" in batch_data:
                            batch_id = batch_data["batch_id"]
                            if not batch_id:
                                print_and_flush(f"{method} {cluster_size} {rep_key} {eval_type}: No batch ID found")
                                continue
                            status = check_batch_status(batch_id)
                            print_and_flush(f"{method} {cluster_size} {rep_key} {eval_type}: {batch_id} -> {status}")
                            if status not in ["completed", "expired", "cancelled"]:
                                all_completed = False
                                break
        
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
        
        # Load existing results for this method
        results_json_path = f'results/vars/{method}_results_{model_id}_layer{args.layer}.json'
        if os.path.exists(results_json_path):
            with open(results_json_path, 'r') as f:
                existing_results_data = json.load(f)
        else:
            existing_results_data = {}

        # Filter cluster sizes if specified
        if args.clusters is not None:
            requested_sizes = [str(size) for size in args.clusters]
            method_batches = {size: data for size, data in method_batches.items() if size in requested_sizes}
            if not method_batches:
                print_and_flush(f"None of the requested cluster sizes {args.clusters} found for {method}. Skipping.")
                continue
        
        eval_results_by_cluster_size = {}
        
        # Process each cluster size
        for cluster_size, cluster_data in method_batches.items():
            n_clusters = cluster_data["n_clusters"]
            cluster_centers = np.array(cluster_data["cluster_centers"])
            
            print_and_flush(f"Processing {n_clusters} clusters...")
            
            all_results = []
            
            # Process each repetition
            for rep_idx in range(args.repetitions):
                rep_key = f"rep_{rep_idx}"
                if rep_key not in cluster_data:
                    print_and_flush(f"Missing repetition {rep_idx} for {method} {cluster_size}. Skipping.")
                    continue
                
                rep_data = cluster_data[rep_key]
                
                # Get existing repetition results if available
                existing_rep_result = {}
                if 'results_by_cluster_size' in existing_results_data and \
                   cluster_size in existing_results_data['results_by_cluster_size'] and \
                   'all_results' in existing_results_data['results_by_cluster_size'][cluster_size] and \
                   rep_idx < len(existing_results_data['results_by_cluster_size'][cluster_size]['all_results']):
                    existing_rep_result = existing_results_data['results_by_cluster_size'][cluster_size]['all_results'][rep_idx]
                
                # Get categories specific to this repetition
                categories = rep_data.get("categories", [])
                if not categories:
                    print_and_flush(f"No categories found for repetition {rep_idx} in {method} {cluster_size}. Skipping.")
                    continue
                
                rep_results = existing_rep_result.copy()
                
                # Process accuracy batch
                if not args.no_accuracy:
                    if "accuracy" in rep_data and rep_data["accuracy"]["batch_id"]:
                        acc_batch_id = rep_data["accuracy"]["batch_id"]
                        acc_metadata = rep_data["accuracy"]["metadata"]
                        
                        status = check_batch_status(acc_batch_id)
                        if status in ["completed", "expired", "cancelled"]:
                            accuracy_results = process_accuracy_batch(acc_batch_id, acc_metadata)
                            if "avg" in accuracy_results:
                                rep_results["avg_accuracy"] = accuracy_results["avg"]["accuracy"]
                                rep_results["avg_f1"] = accuracy_results["avg"]["f1"]
                                rep_results["avg_precision"] = accuracy_results["avg"]["precision"]
                                rep_results["avg_recall"] = accuracy_results["avg"]["recall"]
                                rep_results["accuracy_results_by_cluster"] = {k: v for k, v in accuracy_results.items() if k != "avg"}
                            else:
                                print_and_flush(f"Accuracy batch {acc_batch_id} processed with no 'avg' results.")
                                continue
                        else:
                            print_and_flush(f"Accuracy batch {acc_batch_id} not completed (status: {status})")
                            continue
                elif 'avg_accuracy' not in rep_results and not args.no_accuracy:
                    print_and_flush(f"WARNING: Accuracy results not found for {method} {cluster_size} rep {rep_idx} and --no-accuracy is set.")

                # Process completeness batch
                if not args.no_completeness:
                    if "completeness" in rep_data and rep_data["completeness"]["batch_id"]:
                        comp_batch_id = rep_data["completeness"]["batch_id"]
                        comp_metadata = rep_data["completeness"]["metadata"]
                        
                        status = check_batch_status(comp_batch_id)
                        if status in ["completed", "expired", "cancelled"]:
                            completeness_results = process_completeness_batch(comp_batch_id, comp_metadata)
                            rep_results["avg_fit_score"] = completeness_results.get("avg_fit_score", 0.0)
                            rep_results["avg_fit_score_by_cluster_id"] = completeness_results.get("avg_fit_score_by_cluster_id", {})
                            rep_results["completeness_responses"] = completeness_results.get("responses", [])
                            # Normalize completeness score to 0-1 scale for compatibility with final score calculation
                            rep_results["avg_confidence"] = completeness_results.get("avg_fit_score", 0.0) / 10.0
                        else:
                            print_and_flush(f"Completeness batch {comp_batch_id} not completed (status: {status})")
                            continue
                elif 'avg_confidence' not in rep_results and not args.no_completeness:
                    print_and_flush(f"WARNING: Completeness results not found for {method} {cluster_size} rep {rep_idx} and --no-completeness is set.")

                # Process semantic orthogonality batch
                if not args.no_sem_orth:
                    if "semantic_orthogonality" in rep_data and rep_data["semantic_orthogonality"]["batch_id"]:
                        sem_batch_id = rep_data["semantic_orthogonality"]["batch_id"]
                        sem_metadata = rep_data["semantic_orthogonality"]["metadata"]
                        
                        if sem_batch_id:
                            status = check_batch_status(sem_batch_id)
                            if status in ["completed", "expired", "cancelled"]:
                                semantic_orthogonality_result = process_semantic_orthogonality_batch(sem_batch_id, sem_metadata)
                                if semantic_orthogonality_result:
                                    rep_results["semantic_orthogonality_matrix"] = semantic_orthogonality_result.get("semantic_orthogonality_matrix", np.array([]).tolist())
                                    rep_results["semantic_orthogonality_explanations"] = semantic_orthogonality_result.get("semantic_orthogonality_explanations", {})
                                    rep_results["semantic_orthogonality_score"] = semantic_orthogonality_result.get("semantic_orthogonality_score", 0.0)
                                    rep_results["semantic_orthogonality_threshold"] = semantic_orthogonality_result.get("semantic_orthogonality_threshold", 0.0)
                            else:
                                print_and_flush(f"Semantic orthogonality batch {sem_batch_id} not completed (status: {status})")
                elif 'semantic_orthogonality_score' not in rep_results and not args.no_sem_orth:
                    print_and_flush(f"WARNING: Semantic orthogonality results not found for {method} {cluster_size} rep {rep_idx} and --no-sem-orth is set.")

                # Compute centroid orthogonality
                if not args.no_orth:
                    orthogonality = compute_centroid_orthogonality(cluster_centers)
                    rep_results["orthogonality"] = orthogonality
                elif 'orthogonality' not in rep_results:
                    print_and_flush(f"WARNING: Orthogonality results not found for {method} {cluster_size} rep {rep_idx} and --no-orth is set.")
                
                # Add categories (repetition-specific)
                rep_results["categories"] = categories
                
                # Calculate final score
                final_score_components = []
                if "avg_f1" in rep_results:
                    final_score_components.append(rep_results["avg_f1"])
                else:
                    print_and_flush(f"WARNING WHEN COMPUTING FINAL SCORE: avg_f1 not found for {method} {cluster_size} rep {rep_idx} and --no-accuracy is set.")
                if "avg_confidence" in rep_results:
                    final_score_components.append(rep_results["avg_confidence"])
                else:
                    print_and_flush(f"WARNING WHEN COMPUTING FINAL SCORE: avg_confidence not found for {method} {cluster_size} rep {rep_idx} and --no-completeness is set.")
                if "semantic_orthogonality_score" in rep_results:
                    final_score_components.append(rep_results["semantic_orthogonality_score"])
                else:
                    print_and_flush(f"WARNING WHEN COMPUTING FINAL SCORE: semantic_orthogonality_score not found for {method} {cluster_size} rep {rep_idx} and --no-sem-orth is set.")

                if final_score_components:
                    final_score = sum(final_score_components) / len(final_score_components)
                else:
                    final_score = 0.0

                rep_results["final_score"] = final_score
                
                all_results.append(rep_results)
                
                print_and_flush(f"  Processed repetition {rep_idx + 1}: final_score={final_score:.4f}")
            
            if not all_results:
                print_and_flush(f"No valid results for {method} with {n_clusters} clusters")
                continue
            
            # Calculate average final score and statistics
            avg_final_score = np.mean([result['final_score'] for result in all_results])
            
            # Compute statistics across all repetitions
            statistics = {}
            metrics_to_stat = [
                'avg_accuracy', 'avg_f1', 'avg_precision', 'avg_recall', 'orthogonality',
                'semantic_orthogonality_score', 'avg_confidence', 'final_score'
            ]

            for metric in metrics_to_stat:
                values = [res[metric] for res in all_results if metric in res]
                if not values:
                    continue

                mean = np.mean(values)
                
                n = len(values)
                if n > 1:
                    from scipy import stats
                    std_dev = np.std(values, ddof=1)  # Sample standard deviation
                    sem = stats.sem(values)
                    # 95% confidence interval using the t-distribution
                    conf = 0.95
                    t_score = stats.t.ppf((1+conf)/2, df=n - 1)
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
            
            eval_results_by_cluster_size[cluster_size] = {
                "all_results": all_results,
                "avg_final_score": avg_final_score,
                "statistics": statistics
            }
            
            print_and_flush(f"Completed {method} with {n_clusters} clusters: avg_score={avg_final_score:.4f}")
        
        # Save results using the existing function
        if eval_results_by_cluster_size:
            save_clustering_results(args.model, args.layer, method, eval_results_by_cluster_size)
            print_and_flush(f"Saved results for {method}")
        else:
            print_and_flush(f"No results to save for {method}")
    
    print_and_flush("All evaluation batch results processed successfully!")


def evaluate_clustering_direct():
    """Evaluate clustering directly without using batch API."""
    print_and_flush("=== EVALUATING CLUSTERING DIRECTLY ===")
    
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
    for method in clustering_methods:
        print_and_flush(f"\n=== Processing {method.upper()} ===")
        
        # Load existing results to get cluster sizes
        results_json_path = f'results/vars/{method}_results_{model_id}_layer{args.layer}.json'
        if not os.path.exists(results_json_path):
            print_and_flush(f"No existing results found for {method} at {results_json_path}. Skipping.")
            continue
            
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
        
        eval_results_by_cluster_size = {}
        
        # Process each cluster size
        for cluster_size in clusters:
            n_clusters = int(cluster_size)
            print_and_flush(f"Evaluating {method} with {n_clusters} clusters...")
            
            try:
                # Load the trained clustering model
                clustering_data = load_trained_clustering_data(model_id, args.layer, n_clusters, method)
                cluster_centers = clustering_data['cluster_centers']
                
                # Predict cluster labels for current activations
                if method == 'sae_topk':
                    cluster_labels = predict_clusters(all_activations, clustering_data, model_id, args.layer, n_clusters)
                else:
                    cluster_labels = predict_clusters(all_activations, clustering_data)
                
                # Get all categories from existing results (one set per repetition)
                cluster_results = existing_results["results_by_cluster_size"].get(cluster_size, {})
                all_categories = []
                if "all_results" in cluster_results and len(cluster_results["all_results"]) > 0:
                    for repetition_data in cluster_results["all_results"]:
                        if "categories" in repetition_data:
                            all_categories.append(repetition_data["categories"])
                
                if not all_categories:
                    print_and_flush(f"No categories found for {method} with {n_clusters} clusters. Skipping evaluation.")
                    continue
                
                # Ensure we have enough category sets for the number of repetitions
                if len(all_categories) < args.repetitions:
                    raise ValueError(f"Only {len(all_categories)} category sets found, but {args.repetitions} repetitions requested. Using available categories.")
                
                # Evaluate clustering using the direct method from clustering.py
                evaluation_results = evaluate_clustering_scoring_metrics(
                    all_texts, 
                    cluster_labels, 
                    n_clusters, 
                    all_activations, 
                    cluster_centers, 
                    args.model, 
                    args.n_autograder_examples, 
                    args.description_examples,
                    all_categories,
                    repetitions=args.repetitions,
                    model_id=model_id,
                    layer=args.layer,
                    clustering_data=clustering_data,
                    no_accuracy=args.no_accuracy,
                    no_completeness=args.no_completeness,
                    no_sem_orth=args.no_sem_orth,
                    no_orth=args.no_orth,
                    existing_results=existing_results,
                    target_cluster_percentage=args.accuracy_target_cluster_percentage
                )
                
                eval_results_by_cluster_size[cluster_size] = evaluation_results
                
                print_and_flush(f"Completed evaluation for {method} with {n_clusters} clusters: avg_score={evaluation_results['avg_final_score']:.4f}")
                
            except Exception as e:
                print_and_flush(f"Error processing {method} with {n_clusters} clusters: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save results using the existing function
        if eval_results_by_cluster_size:
            save_clustering_results(args.model, args.layer, method, eval_results_by_cluster_size)
            print_and_flush(f"Saved results for {method}")
        else:
            print_and_flush(f"No results to save for {method}")
    
    print_and_flush("All direct evaluations completed successfully!")


def print_evaluation_summary(results, method):
    """
    Print a summary of evaluation results.
    
    Parameters:
    -----------
    results : dict
        Evaluation results
    method : str
        Clustering method name
    """
    
    print_and_flush("\n" + "="*50)
    print_and_flush(f"{method.upper()} EVALUATION SUMMARY")
    print_and_flush("="*50)
    print_and_flush(f"Model: {model_id.upper()}, Layer: {args.layer}")
    
    # Extract data from new format
    best_cluster = results['best_cluster']
    results_by_cluster_size = results['results_by_cluster_size']
    
    print_and_flush(f"Optimal clusters: {best_cluster['size']}")
    print_and_flush(f"Optimal final score: {best_cluster['avg_final_score']:.4f}")
    if 'avg_accuracy' in best_cluster:
        print_and_flush(f"Optimal accuracy: {best_cluster['avg_accuracy']:.4f}")
    if 'avg_precision' in best_cluster:
        print_and_flush(f"Optimal precision: {best_cluster['avg_precision']:.4f}")
    if 'avg_recall' in best_cluster:
        print_and_flush(f"Optimal recall: {best_cluster['avg_recall']:.4f}")
    if 'avg_f1' in best_cluster:
        print_and_flush(f"Optimal F1: {best_cluster['avg_f1']:.4f}")
    if 'avg_completeness' in best_cluster:
        print_and_flush(f"Optimal completeness: {best_cluster['avg_completeness']:.4f}")
    if 'orthogonality' in best_cluster:
        print_and_flush(f"Optimal orthogonality: {best_cluster['orthogonality']:.4f}")
    if 'semantic_orthogonality' in best_cluster:
        print_and_flush(f"Optimal semantic orthogonality: {best_cluster['semantic_orthogonality']:.4f}")
    
    print_and_flush("\nMetrics for all cluster sizes:")
    
    headers = ["Clusters", "Final"]
    if not args.no_accuracy:
        headers.extend(["Accuracy", "Precision", "Recall", "F1"])
    if not args.no_completeness:
        headers.append("Conf")
    if not args.no_orth:
        headers.append("Orthog")
    if not args.no_sem_orth:
        headers.append("SemOrth")
        
    header_format = f"{'':<2}{'{:<8}' * len(headers)}"
    print_and_flush(header_format.format(*headers))
    print_and_flush("  " + " ".join(["-"*8] * len(headers)))
    
    # Extract metrics for all cluster sizes from new format
    clusters = sorted([int(k) for k in results_by_cluster_size.keys()])
    optimal_n_clusters = int(best_cluster['size'])
    
    for n_clusters in clusters:
        cluster_results = results_by_cluster_size[str(n_clusters)]
        avg_final_score = cluster_results['avg_final_score']
        
        # Get average metrics across all repetitions
        all_repetitions = cluster_results['all_results']
        
        row_data = [n_clusters, f"{avg_final_score:.4f}"]
        
        if not args.no_accuracy:
            avg_accuracy = np.mean([rep['avg_accuracy'] for rep in all_repetitions if 'avg_accuracy' in rep])
            avg_precision = np.mean([rep['avg_precision'] for rep in all_repetitions if 'avg_precision' in rep])
            avg_recall = np.mean([rep['avg_recall'] for rep in all_repetitions if 'avg_recall' in rep])
            avg_f1 = np.mean([rep['avg_f1'] for rep in all_repetitions if 'avg_f1' in rep])
            row_data.extend([f"{avg_accuracy:.4f}", f"{avg_precision:.4f}", f"{avg_recall:.4f}", f"{avg_f1:.4f}"])

        if not args.no_completeness:
            avg_confidence = np.mean([rep.get('avg_confidence', 0) for rep in all_repetitions])
            row_data.append(f"{avg_confidence:.4f}")

        if not args.no_orth:
            avg_orthogonality = np.mean([rep['orthogonality'] for rep in all_repetitions if 'orthogonality' in rep])
            row_data.append(f"{avg_orthogonality:.4f}")

        if not args.no_sem_orth:
            avg_semantic_orthogonality = np.mean([rep['semantic_orthogonality_score'] for rep in all_repetitions if 'semantic_orthogonality_score' in rep])
            row_data.append(f"{avg_semantic_orthogonality:.4f}")

        prefix = "* " if n_clusters == optimal_n_clusters else "  "
        
        row_format = f"{prefix}{'{:<8}' * len(row_data)}"
        print_and_flush(row_format.format(*row_data))

def main():
    if args.command == "submit":
        submit_evaluation_batches()
    elif args.command == "process":
        process_evaluation_batches()
    elif args.command == "direct":
        evaluate_clustering_direct()


if __name__ == "__main__":
    main() 