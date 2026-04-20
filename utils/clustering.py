"""
Clustering utilities for neural activation analysis.
This module contains common functions for clustering analysis, model saving/loading,
and evaluation metrics.
"""

import os
import json
import re
import numpy as np
import pickle
import time
import random
from tqdm import tqdm
from utils.utils import print_and_flush, chat_batch, convert_numpy_types, NumpyEncoder
from utils.autograder_prompts import (
    build_cluster_description_prompt,
    build_accuracy_autograder_prompt,
    build_semantic_orthogonality_prompt,
    build_completeness_autograder_prompt,
    format_sentences_text_simple
)
from utils.responses import extract_thinking_process
from scipy import stats




def run_chat_batch_with_event_loop_handling(batch_prompts, model, json_mode=False):
    """
    Helper function to run chat_batch with proper async event loop handling.
    Handles both Jupyter (with running event loop) and regular script environments.
    
    Parameters:
    -----------
    batch_prompts : list
        List of prompts to process in batch
    model : str
        Model to use for the chat batch
        
    Returns:
    --------
    list
        List of responses from the chat batch
    """
    import asyncio
    import concurrent.futures
    import threading
    
    # Handle both Jupyter (with running event loop) and regular scripts
    try:
        # Check if there's already a running event loop
        loop = asyncio.get_running_loop()
        # If we get here, we're in an environment with a running loop (like Jupyter)
        # We need to create a task and run it in a separate thread
        
        def run_in_thread():
            return asyncio.run(chat_batch(batch_prompts, model=model, json_mode=json_mode))
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            responses = future.result()
    except RuntimeError:
        # No running event loop, safe to use asyncio.run()
        responses = asyncio.run(chat_batch(batch_prompts, model=model, json_mode=json_mode))
    
    return responses


def parse_json_response(response, expected_field=None):
    """
    Parse JSON response from chat model with robust error handling.
    
    Parameters:
    -----------
    response : str
        Raw response from chat model
    expected_field : str, optional
        Expected top-level field name (e.g., 'categorizations', 'classifications')
        If provided, will attempt field-specific regex extraction
        
    Returns:
    --------
    dict
        Parsed JSON object
        
    Raises:
    -------
    ValueError
        If response is empty
    json.JSONDecodeError
        If JSON cannot be parsed after all cleaning attempts
    """
    import re
    import json
    
    if not response:
        raise ValueError("Empty response from model")
    
    # Try to extract JSON from markdown code blocks first
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try field-specific regex patterns
        if expected_field == 'categorizations':
            json_match = re.search(r'{\s*"categorizations":\s*\[[\s\S]*?\]\s*}', response)
        elif expected_field == 'classifications':
            json_match = re.search(r'{\s*"classifications":\s*\[[\s\S]*?\]\s*}', response)
        else:
            # Generic JSON object pattern
            json_match = re.search(r'{\s*"[^"]*":\s*[\s\S]*?\}', response)
        
        if json_match:
            json_str = json_match.group(0)
        else:
            # If all else fails, try the entire response
            json_str = response
    
    # Robust JSON sanitization
    # First, normalize line endings and handle smart quotes
    json_str = json_str.replace('\r\n', '\n').replace('\r', '\n')
    json_str = json_str.replace('"', '"').replace('"', '"')
    json_str = json_str.replace(''', "'").replace(''', "'")
    
    # Try to parse first, if it fails, apply more aggressive cleaning
    try:
        result = json.loads(json_str)
        return result
    except json.JSONDecodeError as e:
        print(f"Initial JSON parse failed: {e}")
        print("Applying more aggressive JSON cleaning...")
        
        # Fix common issues that break JSON parsing
        # Escape unescaped backslashes (but preserve valid escape sequences)
        json_str = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_str)
        
        # Handle potential issues with ellipses and other characters
        json_str = json_str.replace('…', '...')
        
        # Try parsing again
        try:
            result = json.loads(json_str)
            return result
        except json.JSONDecodeError as e2:
            print(f"Second JSON parse also failed: {e2}")
            print("Trying to fix unescaped quotes in string values...")
            
            # Fix unescaped quotes within string values
            json_str = fix_unescaped_quotes_in_json(json_str)
            
            try:
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError as e3:
                # Last resort: try to fix the JSON by finding the problematic location
                print(f"Third JSON parse also failed: {e3}")
                # Save the problematic JSON for debugging
                print(f"Problematic JSON excerpt around error: {json_str[max(0, e3.pos-100):e3.pos+100]}")
                raise e3


def fix_unescaped_quotes_in_json(json_str):
    """
    Fix unescaped quotes within JSON string values using a simple state machine.
    
    Parameters:
    -----------
    json_str : str
        JSON string that may contain unescaped quotes
        
    Returns:
    --------
    str
        JSON string with quotes properly escaped
    """
    result = []
    i = 0
    in_string = False
    
    while i < len(json_str):
        char = json_str[i]
        
        if char == '"' and (i == 0 or json_str[i-1] != '\\'):
            # Found an unescaped quote
            if not in_string:
                # Starting a string
                in_string = True
                result.append(char)
            else:
                # Check if this ends the string by looking at what comes next
                # Skip whitespace to see what's after this quote
                j = i + 1
                while j < len(json_str) and json_str[j] in ' \t\n\r':
                    j += 1
                
                if j < len(json_str) and json_str[j] in ',}]':
                    # This quote ends the string
                    in_string = False
                    result.append(char)
                else:
                    # This quote is inside the string and should be escaped
                    result.append('\\"')
            i += 1
        elif char == '\\' and i + 1 < len(json_str) and json_str[i + 1] == '"':
            # This is already an escaped quote, keep it as is
            result.append(char)
            result.append(json_str[i + 1])
            i += 2
        else:
            result.append(char)
            i += 1
    
    return ''.join(result)


def load_trained_clustering_data(model_id, layer, n_clusters, method):
    """
    Load trained clustering data for a specific model, layer, and clustering method.
    
    Parameters:
    -----------
    model_id : str
        Model identifier
    layer : int
        Layer number
    n_clusters : int
        Number of clusters
    method : str
        Clustering method name
        
    Returns:
    --------
    dict
        Dictionary containing the clustering data
    """
    if method == 'sae_topk':
        # SAE uses a different path and torch.load
        sae_path = f'results/vars/saes/sae_{model_id}_layer{layer}_clusters{n_clusters}.pt'
        
        if not os.path.exists(sae_path):
            raise FileNotFoundError(f"SAE clustering model not found at {sae_path}")
        
        # Load the SAE model
        import torch
        sae_data = torch.load(sae_path, map_location='cpu', weights_only=False)
        
        # Return the data with method information
        sae_data['method'] = method
        return sae_data
    
    else:
        # All other methods use pickle files
        clustering_path = f'results/vars/{method}/{model_id}_layer{layer}_clusters{n_clusters}.pkl'
        
        if not os.path.exists(clustering_path):
            raise FileNotFoundError(f"Clustering model not found at {clustering_path}")
        
        # Load the clustering model
        with open(clustering_path, 'rb') as f:
            clustering_data = pickle.load(f)
        
        # Add method information
        clustering_data['method'] = method
        return clustering_data


def predict_clusters(activations, clustering_data, model_id=None, layer=None, n_clusters=None):
    """
    Predict cluster labels for new activations using loaded clustering data.
    
    Parameters:
    -----------
    activations : numpy.ndarray
        Normalized activation vectors to predict clusters for
    clustering_data : dict
        Dictionary containing the clustering data loaded from file
    model_id : str, optional
        Model identifier (required for SAE direct loading mode)
    layer : int, optional
        Layer number (required for SAE direct loading mode)
    n_clusters : int, optional
        Number of clusters (required for SAE direct loading mode)
        
    Returns:
    --------
    numpy.ndarray
        cluster_labels array of cluster assignments
    """
    method = clustering_data['method']
    
    if method == 'agglomerative':
        # For agglomerative clustering, assign to nearest cluster center using cosine similarity
        cluster_centers = clustering_data['cluster_centers']
        
        # Normalize both activations and cluster centers
        activations_norm = activations / np.linalg.norm(activations, axis=1, keepdims=True)
        centers_norm = cluster_centers / np.linalg.norm(cluster_centers, axis=1, keepdims=True)
        
        # Compute cosine similarity and assign to nearest center
        similarities = np.dot(activations_norm, centers_norm.T)
        cluster_labels = np.argmax(similarities, axis=1)
        
        return cluster_labels
    
    elif method == 'spherical_kmeans':
        # Use KMeans predict method on normalized data
        model = clustering_data['model']
        
        # Normalize activations for spherical kmeans
        activations_norm = activations / np.linalg.norm(activations, axis=1, keepdims=True)
        
        # Predict using the KMeans model
        cluster_labels = model.predict(activations_norm)
        
        return cluster_labels
    
    elif method == 'gmm':
        # Use GMM predict method
        model = clustering_data['model']
        
        # Predict using the GMM model
        cluster_labels = model.predict(activations)
        
        return cluster_labels
    
    elif method == 'pca_kmeans':
        # Apply PCA transform first, then use KMeans predict
        pca = clustering_data['pca']
        kmeans = clustering_data['kmeans']
        
        # Apply PCA transformation
        reduced_data = pca.transform(activations)
        
        # Predict using KMeans
        cluster_labels = kmeans.predict(reduced_data)
        
        return cluster_labels
    
    elif method == 'pca_gmm':
        # Apply PCA transform first, then use GMM predict
        pca = clustering_data['pca']
        gmm = clustering_data['gmm']
        
        # Apply PCA transformation
        reduced_data = pca.transform(activations)
        
        # Predict using GMM
        cluster_labels = gmm.predict(reduced_data)
        
        return cluster_labels
    
    elif method == 'pca_agglomerative':
        # Apply PCA transform first, then assign to nearest cluster center
        pca = clustering_data['pca']
        cluster_centers = clustering_data['cluster_centers']
        
        # Apply PCA transformation
        reduced_data = pca.transform(activations)
        
        # Normalize reduced data for cosine similarity
        reduced_norm = reduced_data / np.linalg.norm(reduced_data, axis=1, keepdims=True)
        
        # Get reduced cluster centers by applying PCA to original centers
        reduced_centers = pca.transform(cluster_centers)
        centers_norm = reduced_centers / np.linalg.norm(reduced_centers, axis=1, keepdims=True)
        
        # Compute cosine similarity and assign to nearest center
        similarities = np.dot(reduced_norm, centers_norm.T)
        cluster_labels = np.argmax(similarities, axis=1)
        
        return cluster_labels
    
    elif method == 'sae_topk':
        # Use SAE encoder - requires model_id, layer, and n_clusters for direct SAE loading
        if model_id is None or layer is None or n_clusters is None:
            raise ValueError("model_id, layer, and n_clusters are required for SAE method predictions")
        
        from utils.sae import load_sae
        import torch
        
        # Load the SAE
        sae, _ = load_sae(model_id, layer, n_clusters)
        
        # Convert activations to torch tensor and move to appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sae = sae.to(device)
        activations_tensor = torch.from_numpy(activations).float().to(device)
        
        # Use SAE encoder to get latent activations
        with torch.no_grad():
            encoded_activations = sae.encoder(activations_tensor - sae.b_dec)
            # Get the cluster assignment as the argmax of encoder activations
            cluster_labels = encoded_activations.argmax(dim=1).cpu().numpy()
        
        return cluster_labels
    
    else:
        raise ValueError(f"Unknown clustering method: {method}")


def get_latent_descriptions(model_id, layer, n_clusters, clustering_method='sae_topk', sorted=False):
    """Get titles and descriptions for cluster latents from the new results format"""
    # Use the new file naming convention
    model_short_name = model_id.split("/")[-1].lower()
    results_path = f'../train-saes/results/vars/{clustering_method}_results_{model_short_name}_layer{layer}.json'
    
    if not os.path.exists(results_path):
        print(f"Warning: Results file not found at {results_path}")
        return {}
    
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Extract category descriptions from the first repetition for the specified n_clusters
        if str(n_clusters) in results.get('results_by_cluster_size', {}):
            cluster_results = results['results_by_cluster_size'][str(n_clusters)]
            if 'all_results' in cluster_results and len(cluster_results['all_results']) > 0:
                first_repetition = cluster_results['all_results'][0]
                if 'categories' in first_repetition:
                    if sorted:
                        categories = [{'key': f"idx{cluster_id}", 'title': title, 'description': description} for cluster_id, title, description in first_repetition['categories']]
                        categories.sort(key=lambda x: int(x['key'][3:]))
                        return {pos: item for pos, item in enumerate(categories)}
                    else:
                        categories = {}
                        for cluster_id, title, description in first_repetition['categories']:
                            categories[int(cluster_id)] = {'key': f"idx{cluster_id}", 'title': title, 'description': description}
                        return categories
    except Exception as e:
        print(f"Error loading cluster descriptions: {e}")
    
    return {}

def generate_cluster_descriptions(model_name, cluster_examples_list, evaluator_model, n_trace_examples=3, n_categories_examples=5):
    """
    Generate descriptions for multiple clusters in batch.
    
    Args:
        cluster_examples_list (list): List of tuples (cluster_idx, examples) for each cluster
        model (str): Model to use for generating descriptions
        n_trace_examples (int): Number of full reasoning trace examples to include in prompts
        model_name (str): Name of the model whose responses should be loaded for trace examples
        
    Returns:
        list: List of tuples (cluster_idx, title, description) for each cluster
    """
    # Prepare trace examples if requested (shared across all clusters)
    trace_examples_text = ""
    if n_trace_examples > 0 and model_name is not None:
        try:
            # Get model identifier for file naming
            model_id = model_name.split('/')[-1].lower()
            responses_json_path = f"../generate-responses/results/vars/responses_{model_id}.json"
            
            # Load responses
            with open(responses_json_path, 'r') as f:
                responses_data = json.load(f)
            
            # Select random examples
            trace_samples = random.sample(responses_data, min(n_trace_examples, len(responses_data)))
            
            # Extract thinking processes
            trace_examples = []
            for sample in trace_samples:
                thinking_process = extract_thinking_process(sample["full_response"])
                if thinking_process:
                    trace_examples.append(thinking_process)
            
            if trace_examples:
                trace_examples_text = "Here are some full reasoning traces to help understand the context:\n'''\n"
                for i, trace in enumerate(trace_examples):
                    trace_examples_text += f"TRACE {i+1}:\n{trace}\n\n"
                trace_examples_text += "'''"
        except Exception as e:
            print(f"Error loading trace examples: {e}")
    
    # Create prompts for all clusters
    batch_prompts = []
    cluster_indices = []
    
    for cluster_idx, examples in cluster_examples_list:
        # Create a prompt for this cluster using the centralized prompt builder
        prompt = build_cluster_description_prompt(
            examples, 
            trace_examples_text, 
            n_categories_examples
        )
        
        batch_prompts.append(prompt)
        cluster_indices.append(cluster_idx)
    
    # Process all prompts in batch
    print(f"Processing {len(batch_prompts)} cluster description prompts in batch...")
    responses = run_chat_batch_with_event_loop_handling(batch_prompts, evaluator_model)
    
    # Parse responses to extract titles and descriptions
    results = []
    for i, response in enumerate(responses):
        cluster_idx = cluster_indices[i]
        
        # Parse the response to extract title and description
        title = "Unnamed Cluster"
        description = "No description available"
        
        title_match = re.search(r"Title:\s*(.*?)(?:\n|$)", response)
        if title_match:
            title = title_match.group(1).strip()
            
        desc_match = re.search(r"Description:\s*(.*?)(?:\n|$)", response)
        if desc_match:
            description = desc_match.group(1).strip()
        
        results.append((str(cluster_idx), title, description))
    
    return results

def simplify_category_name(category_name):
    """
    Simplify a category name by extracting just the number if it matches 'Category N'
    or return the original name if not.
    """
    import re
    match = re.match(r'Category\s+(\d+)', category_name)
    if match:
        return match.group(1)
    return category_name



def accuracy_autograder(sentences, categories, ground_truth_labels, model, n_autograder_examples, max_sentences_per_prompt=50, target_cluster_percentage=0.2):
    """
    Binary autograder that evaluates each cluster independently against examples from outside the cluster.
    
    Args:
        sentences (list): List of all sentences to potentially sample from
        categories (list): List of tuples where each tuple is (cluster_id, title, description)
        ground_truth_labels (list): List of cluster IDs (as strings) for each sentence in sentences
        model (str): Model to use for the autograding
        n_autograder_examples (int): Total number of examples to use for testing each cluster
        max_sentences_per_prompt (int): Maximum number of sentences to process per prompt
        target_cluster_percentage (float): Percentage of examples to take from target cluster (default: 0.2)
    
    Returns:
        dict: Metrics including precision, recall, accuracy and F1 score for each category
    """
    results = {}
    
    # Get a mapping from sentence index to cluster ID for easy lookup
    sentence_to_cluster = {i: label for i, label in enumerate(ground_truth_labels)}
    
    # Collect all prompts and metadata for batch processing
    batch_prompts = []
    batch_metadata = []
    
    # For each category, prepare data and prompts
    for cluster_id, title, description in categories:
        cluster_id_str = str(cluster_id)
        
        # Find all examples in this cluster and not in this cluster
        in_cluster_indices = [i for i, label in enumerate(ground_truth_labels) if label == cluster_id_str]
        out_cluster_indices = [i for i, label in enumerate(ground_truth_labels) if label != cluster_id_str]
        
        # Calculate number of examples from target cluster and outside clusters based on percentage
        from_cluster_count = max(1, int(n_autograder_examples * target_cluster_percentage))
        from_outside_count = n_autograder_examples - from_cluster_count
        
        # Ensure we don't exceed available examples
        from_cluster_count = min(len(in_cluster_indices), from_cluster_count)
        from_outside_count = min(len(out_cluster_indices), from_outside_count)
        
        # Sample examples
        in_cluster_sample = random.sample(in_cluster_indices, from_cluster_count)
        out_cluster_sample = random.sample(out_cluster_indices, from_outside_count)
        
        # Combine the samples and remember the ground truth
        test_indices = in_cluster_sample + out_cluster_sample
        test_sentences = [sentences[i] for i in test_indices]
        test_ground_truth = ["Yes" if i in in_cluster_sample else "No" for i in test_indices]
        
        # Shuffle to avoid position bias, ensuring all lists are shuffled in the same order
        combined = list(zip(test_indices, test_sentences, test_ground_truth))
        random.shuffle(combined)
        test_indices, test_sentences, test_ground_truth = zip(*combined) if combined else ([], [], [])

        # Convert tuples back to lists for slicing
        test_indices = list(test_indices)
        test_sentences = list(test_sentences)
        test_ground_truth = list(test_ground_truth)
        
        # Split sentences into chunks if necessary
        total_test_sentences = len(test_sentences)
        sentence_chunks = []
        ground_truth_chunks = []
        indices_chunks = []

        if total_test_sentences <= max_sentences_per_prompt:
            # Process all sentences in a single prompt if the total is within the limit
            if total_test_sentences > 0:
                sentence_chunks.append(test_sentences)
                ground_truth_chunks.append(test_ground_truth)
                indices_chunks.append(test_indices)
        else:
            # Split into multiple chunks if the total exceeds the limit
            for i in range(0, total_test_sentences, max_sentences_per_prompt):
                end_idx = min(i + max_sentences_per_prompt, total_test_sentences)
                sentence_chunks.append(test_sentences[i:end_idx])
                ground_truth_chunks.append(test_ground_truth[i:end_idx])
                indices_chunks.append(test_indices[i:end_idx])
        
        # Create prompts for all chunks of this cluster
        for chunk_idx, (sentence_chunk, ground_truth_chunk, indices_chunk) in enumerate(zip(sentence_chunks, ground_truth_chunks, indices_chunks)):
            # Format the sentences into a numbered list (starting from 0 for each chunk)
            sentences_text = format_sentences_text_simple(sentence_chunk)

            # Create a prompt for binary classification using the centralized prompt builder
            prompt = build_accuracy_autograder_prompt(title, description, sentences_text)
            
            # Add prompt and metadata to batch
            batch_prompts.append(prompt)
            batch_metadata.append({
                "cluster_id": cluster_id,
                "cluster_id_str": cluster_id_str,
                "title": title,
                "description": description,
                "chunk_idx": chunk_idx,
                "test_sentences": sentence_chunk,
                "test_ground_truth": ground_truth_chunk,
                "test_indices": indices_chunk  # Store original indices for reference
            })
    
    # Process all prompts in batch
    print(f"Processing {len(batch_prompts)} prompts in batch for evaluating accuracy...")
    responses = run_chat_batch_with_event_loop_handling(batch_prompts, model, json_mode=True)
    
    # Group responses by cluster_id for aggregation
    cluster_responses = {}
    parsing_errors = []
    
    # Process each response from the batch
    for i, response in enumerate(responses):
        metadata = batch_metadata[i]
        cluster_id_str = metadata["cluster_id_str"]
        
        if cluster_id_str not in cluster_responses:
            cluster_responses[cluster_id_str] = {
                "metadata": metadata,  # Store metadata from first chunk (contains cluster info)
                "all_classifications": []
            }
        
        try:
            result = parse_json_response(response, expected_field='classifications')
            
            # Add sentence text and ground truth to each classification
            for item in result["classifications"]:
                sentence_idx = item["sentence_id"]
                if sentence_idx < len(metadata["test_sentences"]):
                    item["sentence_text"] = metadata["test_sentences"][sentence_idx]
                    item["ground_truth"] = metadata["test_ground_truth"][sentence_idx]
                else:
                    print(f"Warning: sentence_id {sentence_idx} out of range for chunk with {len(metadata['test_sentences'])} sentences")
            
            # Add the classifications from this chunk to the cluster's overall list
            cluster_responses[cluster_id_str]["all_classifications"].extend(result["classifications"])
            
        except Exception as e:
            print(f"Error parsing response for cluster {metadata['cluster_id']} chunk {metadata['chunk_idx']}: {e}")
            print(f"Raw response: {response}")
            parsing_errors.append({
                "cluster_id": metadata["cluster_id"],
                "chunk_idx": metadata["chunk_idx"],
                "error": str(e),
                "raw_response": response
            })
    
    # Process aggregated results for each cluster
    for cluster_id_str, cluster_data in cluster_responses.items():
        metadata = cluster_data["metadata"]
        cluster_id = metadata["cluster_id"]
        all_classifications = cluster_data["all_classifications"]
        
        if not all_classifications:
            print(f"No valid classifications for cluster {cluster_id}")
            results[cluster_id_str] = {
                "error": "No valid classifications",
                "parsing_errors": [e for e in parsing_errors if e["cluster_id"] == cluster_id]
            }
            continue
        
        # Compute metrics for this cluster
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        predictions = []
        enhanced_classifications = []
        
        for item in all_classifications:
            if "belongs_to_category" not in item or "ground_truth" not in item:
                print(f"Warning: Missing required fields in classification item: {item}")
                continue
                
            belongs = item["belongs_to_category"]
            predictions.append(belongs)
            
            true_label = item["ground_truth"]
            
            if belongs == "Yes" and true_label == "Yes":
                true_positives += 1
            elif belongs == "Yes" and true_label == "No":
                false_positives += 1
            elif belongs == "No" and true_label == "Yes":
                false_negatives += 1
            elif belongs == "No" and true_label == "No":
                true_negatives += 1
            
            enhanced_classifications.append(item)
        
        # Calculate metrics
        total_predictions = len(enhanced_classifications)
        accuracy = (true_positives + true_negatives) / total_predictions if total_predictions > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results[cluster_id_str] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
        }
    
    # Calculate overall averages across all clusters
    if results:
        valid_results = {k: v for k, v in results.items() if "error" not in v}
        if valid_results:
            avg_accuracy = sum(r["accuracy"] for r in valid_results.values()) / len(valid_results)
            avg_precision = sum(r["precision"] for r in valid_results.values()) / len(valid_results)
            avg_recall = sum(r["recall"] for r in valid_results.values()) / len(valid_results)
            avg_f1 = sum(r["f1"] for r in valid_results.values()) / len(valid_results)
            
            results["avg"] = {
                "accuracy": avg_accuracy,
                "precision": avg_precision,
                "recall": avg_recall,
                "f1": avg_f1
            }
    
    # Add parsing error summary if any occurred
    if parsing_errors:
        results["parsing_errors"] = parsing_errors
    
    return results


def completeness_autograder(sentences, cluster_labels, categories, model, max_sentences_per_prompt=50):
    """
    Autograder that evaluates completeness by measuring how well sentences fit their assigned clusters.
    
    Args:
        sentences (list): List of sentences to evaluate
        cluster_labels (list): List of cluster IDs (as strings) for each sentence
        categories (list): List of tuples where each tuple is (cluster_id, title, description)
        model (str): Model to use for evaluation
        max_sentences_per_prompt (int): Maximum number of sentences to process per prompt
    
    Returns:
        dict: Statistics about completeness scores including detailed analysis
    """
    # Create a mapping from cluster ID to category info
    cluster_id_to_category = {str(cluster_id): (title, description) for cluster_id, title, description in categories}
    
    # Filter out sentences that don't have valid cluster assignments
    valid_evaluations = []
    for i, (sentence, cluster_id) in enumerate(zip(sentences, cluster_labels)):
        if str(cluster_id) in cluster_id_to_category:
            title, description = cluster_id_to_category[str(cluster_id)]
            valid_evaluations.append((i, sentence, str(cluster_id), title, description))
    
    if not valid_evaluations:
        return {
            "error": "No valid sentence-cluster assignments found",
            "total_sentences": len(sentences),
            "evaluated_sentences": 0,
            "avg_fit_score": 0.0,
            "responses": []
        }
    
    # Split evaluations into chunks if necessary
    if len(valid_evaluations) <= max_sentences_per_prompt:
        # Process all evaluations in a single prompt per sentence
        evaluation_chunks = [[eval_item] for eval_item in valid_evaluations]
    else:
        # Split into multiple chunks - but still one sentence per prompt for now
        # This can be optimized later to batch multiple sentences per prompt if needed
        evaluation_chunks = [[eval_item] for eval_item in valid_evaluations]
    
    # Create prompts for all evaluations
    batch_prompts = []
    evaluation_metadata = []
    
    for chunk_idx, chunk in enumerate(evaluation_chunks):
        for eval_idx, (sentence_idx, sentence, cluster_id, title, description) in enumerate(chunk):
            prompt = build_completeness_autograder_prompt(sentence, title, description)
            batch_prompts.append(prompt)
            evaluation_metadata.append({
                "sentence_idx": sentence_idx,
                "sentence": sentence,
                "cluster_id": cluster_id,
                "title": title,
                "description": description
            })
    
    # Process all prompts in batch
    print(f"Processing {len(batch_prompts)} completeness evaluation prompts in batch...")
    responses = run_chat_batch_with_event_loop_handling(batch_prompts, model, json_mode=True)
    
    # Process responses
    all_responses = []
    parsing_errors = []
    total_fit_score = 0.0
    
    for i, response in enumerate(responses):
        metadata = evaluation_metadata[i]
        
        try:
            result = parse_json_response(response)
            
            completeness_score = result.get('completeness_score', 0)
            explanation = result.get('explanation', '')
            
            # Validate completeness score is in range
            if not isinstance(completeness_score, (int, float)) or completeness_score < 0 or completeness_score > 10:
                print(f"Warning: Invalid completeness score {completeness_score}, clamping to valid range")
                completeness_score = max(0, min(10, int(completeness_score)))
            
            evaluation_item = {
                "sentence_idx": metadata["sentence_idx"],
                "sentence": metadata["sentence"],
                "cluster_id": metadata["cluster_id"],
                "title": metadata["title"],
                "description": metadata["description"],
                "completeness_score": completeness_score,
                "explanation": explanation
            }
            
            all_responses.append(evaluation_item)
            total_fit_score += completeness_score
            
        except Exception as e:
            print(f"Error parsing completeness evaluation response {i}: {e}")
            print(f"Raw response: {response}")
            parsing_errors.append({
                "sentence_idx": metadata["sentence_idx"],
                "error": str(e),
                "raw_response": response
            })
    
    # Calculate statistics
    num_evaluated = len(all_responses)
    avg_fit_score = total_fit_score / num_evaluated if num_evaluated > 0 else 0.0
    
    # Calculate average fit score by cluster
    avg_fit_score_by_cluster_id = {}
    cluster_scores = {}
    for response in all_responses:
        cluster_id = response["cluster_id"]
        if cluster_id not in cluster_scores:
            cluster_scores[cluster_id] = []
        cluster_scores[cluster_id].append(response["completeness_score"])
    
    for cluster_id, scores in cluster_scores.items():
        avg_fit_score_by_cluster_id[cluster_id] = sum(scores) / len(scores) if scores else 0.0
    
    result_dict = {
        "total_sentences": len(sentences),
        "evaluated_sentences": num_evaluated,
        "avg_fit_score": avg_fit_score,
        "avg_fit_score_by_cluster_id": avg_fit_score_by_cluster_id,
        "responses": all_responses,
        "parsing_errors": parsing_errors
    }
    
    return result_dict


def compute_centroid_orthogonality(cluster_centers):
    """
    Compute the orthogonality of cluster centroids using 1 - cosine similarity.
    Uses pairwise_distances from sklearn to explicitly compute all pairwise similarities.
    
    Parameters:
    -----------
    cluster_centers : numpy.ndarray
        Cluster center vectors
        
    Returns:
    --------
    float
        Average orthogonality (1 - cosine similarity) between centroids
    """
    start_time = time.time()
    # First compute cosine similarity (not distance)
    norm_cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1, keepdims=True)
    # Use dot product for cosine similarity
    cosine_sim = np.dot(norm_cluster_centers, norm_cluster_centers.T)
    # Take absolute value to treat opposite directions as similar
    abs_cosine_sim = np.abs(cosine_sim)
    # Calculate orthogonality as 1 - absolute similarity
    orthogonality = 1 - abs_cosine_sim
    
    # Get the indices of the upper triangular part (excluding diagonal)
    # This ensures we only count each pair once and exclude self-similarities
    indices = np.triu_indices(orthogonality.shape[0], k=1)
    
    # Extract the upper triangular values
    upper_tri_values = orthogonality[indices]
    
    # Calculate average orthogonality
    avg_orthogonality = np.mean(upper_tri_values) if len(upper_tri_values) > 0 else 0.0
    
    print_and_flush(f"Computed centroid orthogonality in {time.time() - start_time} seconds")
    return avg_orthogonality


def compute_semantic_orthogonality(categories, model="gpt-4.1-mini", orthogonality_threshold=0.5):
    """
    Compute the semantic orthogonality of categories using LLM-based similarity evaluation.
    
    Parameters:
    -----------
    categories : list
        List of tuples (cluster_id, title, description) for each category
    model : str
        Model to use for semantic similarity evaluation
    orthogonality_threshold : float
        Threshold for counting orthogonal pairs (default: 0.5)
        
    Returns:
    --------
    dict
        Dictionary containing:
        - similarity_matrix: Full similarity matrix
        - orthogonality_matrix: Full orthogonality matrix
        - explanations: Dictionary mapping (i, j) pairs to explanation strings
        - orthogonality_score: Fraction of pairs in upper triangle with orthogonality below threshold
        - orthogonality_threshold: The threshold used for computing orthogonality_score
    """
    start_time = time.time()
    n_categories = len(categories)
    
    if n_categories <= 1:
        return {
            "orthogonality_matrix": np.array([[0.0]]) if n_categories == 1 else np.array([]),
            "explanations": {},
            "orthogonality_score": 0.0,
            "orthogonality_threshold": orthogonality_threshold
        }
    
    # Initialize similarity matrix and explanations dictionary
    similarity_matrix = np.zeros((n_categories, n_categories))
    explanations = {}
    
    # Fill diagonal with 1.0 (perfect self-similarity)
    np.fill_diagonal(similarity_matrix, 1.0)
    
    # Prepare batch prompts for all pairs in lower triangle
    batch_prompts = []
    batch_pairs = []
    
    for i in range(n_categories):
        for j in range(i + 1, n_categories):  # Only lower triangle (i < j)
            category1 = categories[i]
            category2 = categories[j]
            
            cluster_id1, title1, description1 = category1
            cluster_id2, title2, description2 = category2
            
            prompt = build_semantic_orthogonality_prompt(title1, description1, title2, description2)
            
            batch_prompts.append(prompt)
            batch_pairs.append((i, j))
    
    # Process all prompts in batch
    print(f"Processing {len(batch_prompts)} semantic similarity prompts in batch...")
    responses = run_chat_batch_with_event_loop_handling(batch_prompts, model, json_mode=True)
    
    # Parse responses and fill similarity matrix
    for idx, response in enumerate(responses):
        i, j = batch_pairs[idx]
        
        try:
            result = parse_json_response(response)
            similarity_score = result.get('similarity_score', 0)
            explanation = result.get('explanation', '')
            
            # Validate score is in range
            if not isinstance(similarity_score, (int, float)) or similarity_score < 0 or similarity_score > 10:
                print(f"Warning: Invalid similarity score {similarity_score}, clamping to valid range")
                similarity_score = max(0, min(10, int(similarity_score)))
            
            # Normalize to 0-1
            normalized_score = similarity_score / 10.0
            
            # Fill both positions in the matrix (symmetric)
            similarity_matrix[i, j] = normalized_score
            similarity_matrix[j, i] = normalized_score
            
            # Store explanations for both pairs (symmetric)
            explanations[f"{i},{j}"] = explanation
            explanations[f"{j},{i}"] = explanation
            
        except Exception as e:
            print(f"Error parsing semantic similarity response for pair ({i}, {j}): {e}")
            print(f"Raw response: {response}")
            # Default to 0 similarity on error
            similarity_matrix[i, j] = 0.0
            similarity_matrix[j, i] = 0.0
            # Default explanation on error
            explanations[f"{i},{j}"] = "Error parsing response"
            explanations[f"{j},{i}"] = "Error parsing response"
    
    # Calculate orthogonality as 1 - similarity
    orthogonality_matrix = 1 - similarity_matrix
    
    # Get the indices of the upper triangular part (excluding diagonal)
    indices = np.triu_indices(orthogonality_matrix.shape[0], k=1)
    
    # Extract the upper triangular values
    upper_tri_values = orthogonality_matrix[indices]
    
    # Calculate orthogonality score (fraction of pairs above threshold)
    orthogonality_score = np.sum(upper_tri_values > orthogonality_threshold) / len(upper_tri_values) if len(upper_tri_values) > 0 else 0
    
    print_and_flush(f"Computed semantic orthogonality in {time.time() - start_time} seconds")
    
    return {
        "semantic_orthogonality_matrix": orthogonality_matrix,
        "semantic_orthogonality_explanations": explanations,
        "semantic_orthogonality_score": orthogonality_score,
        "semantic_orthogonality_threshold": orthogonality_threshold
    }


def generate_representative_examples(cluster_centers, texts, cluster_labels, example_activations, clustering_data=None, model_id=None, layer=None, n_clusters=None):
    """
    Generate representative examples for each cluster.

    For most clustering methods, examples are ranked by Euclidean distance to the
    corresponding cluster centroid.  However, for the SAE-based clustering
    (method == "sae_topk") the decoder weights used as centroids are often not
    good proxies for membership strength.  In that special case we instead
    load the SAE directly and use its encoder to compute activations for every 
    example and rank sentences by the magnitude of the activation for the latent 
    that produced the cluster label.

    Parameters:
    -----------
    cluster_centers : numpy.ndarray
        Cluster centres (unused for SAE ranking).
    texts : list
        List of sentences / reasoning steps.
    cluster_labels : numpy.ndarray
        Cluster labels for each text (length == len(texts)).
    example_activations : numpy.ndarray
        Normalised activation vectors (same order as texts).
    clustering_data : dict, optional
        Full clustering artefact returned by `load_trained_clustering_data`.
        If provided and `clustering_data["method"] == "sae_topk"`, the SAE
        will be loaded directly; otherwise the original centroid-distance logic is applied.
    model_id : str, optional
        Model identifier (required for SAE mode).
    layer : int, optional
        Layer number (required for SAE mode).
    n_clusters : int, optional
        Number of clusters (required for SAE mode).

    Returns:
    --------
    dict
        Mapping `cluster_idx -> list[str]` of sentences ordered from most to
        least representative.
    """
    start_time = time.time()
    representative_examples = {}

    sae_mode = clustering_data is not None and clustering_data.get('method') == 'sae_topk'

    if sae_mode:
        # --- Load SAE directly for more accurate ranking ---
        if model_id is None or layer is None or n_clusters is None:
            raise ValueError("model_id, layer, and n_clusters are required for SAE mode")
        
        # Import load_sae from utils.sae
        from utils.sae import load_sae
        import torch
        
        # Load the SAE
        sae, _ = load_sae(model_id, layer, n_clusters)
        
        # Convert activations to torch tensor and move to appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sae = sae.to(device)
        activations_tensor = torch.from_numpy(example_activations).float().to(device)
        
        # Use SAE encoder to get latent activations
        with torch.no_grad():
            encoded_activations = sae.encoder(activations_tensor - sae.b_dec).detach().cpu().numpy()

    # Iterate over clusters
    skipped_empty_clusters = 0
    for cluster_idx in tqdm(range(len(cluster_centers)), desc="Generating representative examples"):
        # Indices belonging to this cluster
        cluster_indices = np.where(cluster_labels == cluster_idx)[0]

        # Handle empty clusters gracefully
        if len(cluster_indices) == 0:
            representative_examples[cluster_idx] = []
            print_and_flush(f"WARNING:Skipping empty cluster {cluster_idx} in generate_representative_examples")
            skipped_empty_clusters += 1
            continue

        cluster_texts = [texts[i] for i in cluster_indices]

        if sae_mode:
            # Rank by descending encoder activation for this latent
            cluster_scores = encoded_activations[cluster_indices, cluster_idx]
            sorted_local_idx = np.argsort(-cluster_scores)  # negative for descending
        else:
            # Original behaviour: rank by distance to centroid
            cluster_vectors = np.stack([example_activations[i] for i in cluster_indices])
            centroid = cluster_centers[cluster_idx]
            distances = np.linalg.norm(cluster_vectors - centroid, axis=1)
            sorted_local_idx = np.argsort(distances)  # ascending distance

        sorted_examples = [cluster_texts[i] for i in sorted_local_idx]
        representative_examples[cluster_idx] = sorted_examples

    print_and_flush(f"Generated representative examples in {time.time() - start_time} seconds")

    print_and_flush(f"Skipped {skipped_empty_clusters} empty clusters in generate_representative_examples")
    # Assert we haven't skipped more than 10% of clusters
    if skipped_empty_clusters > len(cluster_centers) * 0.25:
        raise ValueError(f"Skipped more than 25% of clusters in generate_representative_examples: {skipped_empty_clusters} / {len(cluster_centers)}")

    return representative_examples


def generate_category_descriptions(cluster_centers, model_name, evaluator_model, n_description_examples, representative_examples, n_trace_examples=3, n_categories_examples=5):
    """
    Generate descriptions for each cluster based on most representative sentences.
    Uses half top examples and half random examples from the cluster.
    
    Parameters:
    -----------
    cluster_centers : numpy.ndarray
        Cluster centers
    texts : list
        List of texts
    cluster_labels : numpy.ndarray
        Cluster labels for each text
    example_activations : numpy.ndarray
        Normalized activation vectors
    model_name : str
        Name of the model to use for generating descriptions
    n_description_examples : int
        Number of examples to use for generating descriptions
        
    Returns:
    --------
    list
        List of tuples (cluster_id, category_title, category_description)
    """
    start_time = time.time()    
    # Prepare batch data for all non-empty clusters
    cluster_examples_list = []
    for cluster_idx in range(len(cluster_centers)):
        # Skip empty clusters
        if len(representative_examples[cluster_idx]) == 0:
            print_and_flush(f"WARNING:Skipping empty cluster {cluster_idx} in generate_category_descriptions")
            continue
        
        # Sample examples from across the entire cluster for better diversity
        cluster_examples = representative_examples[cluster_idx]
        total_examples = len(cluster_examples)
        
        if total_examples <= n_description_examples:
            # If we have fewer examples than requested, use all of them
            examples = cluster_examples
        else:
            # Divide examples into 10 deciles and sample from each
            examples = []
            examples_per_decile = n_description_examples // 10
            remainder = n_description_examples % 10
            
            for decile in range(10):
                # Calculate start and end indices for this decile
                start_idx = (decile * total_examples) // 10
                end_idx = ((decile + 1) * total_examples) // 10
                
                # Get examples from this decile
                decile_examples = cluster_examples[start_idx:end_idx]
                
                # Determine how many to sample from this decile
                num_to_sample = examples_per_decile
                if decile < remainder:  # Distribute remainder across first few deciles
                    num_to_sample += 1
                    
                # Take the first num_to_sample examples from this decile
                # (they are already sorted by distance to centroid)
                examples.extend(decile_examples[:num_to_sample])
        
        # Shuffle examples to ensure diversity
        random.shuffle(examples)

        cluster_examples_list.append((cluster_idx, examples))
    
    # Generate descriptions in batch
    categories = generate_cluster_descriptions(
        model_name,
        cluster_examples_list, 
        evaluator_model,
        n_trace_examples=n_trace_examples,
        n_categories_examples=n_categories_examples
    )
    
    print_and_flush(f"Generated category descriptions in {time.time() - start_time} seconds")
    return categories


def evaluate_clustering_accuracy(texts, cluster_labels, categories, model, n_autograder_examples, target_cluster_percentage=0.2):
    """
    Evaluate clustering using the binary accuracy autograder.
    Tests each cluster independently against examples from other clusters.
    
    Parameters:
    -----------
    texts : list
        List of texts
    cluster_labels : numpy.ndarray
        Cluster labels for each text
    categories : list
        List of tuples (cluster_id, title, description)
    n_autograder_examples : int
        Number of examples from each cluster to use for autograding
        
    Returns:
    --------
    dict
        Autograder results including precision, recall, accuracy and F1 for each cluster
    """
    start_time = time.time()
    # Convert cluster_labels to list of strings for compatibility
    str_cluster_labels = [str(label) for label in cluster_labels]
    
    # Run binary autograder
    for _ in range(3):
        try:
            results = accuracy_autograder(texts, categories, str_cluster_labels, model, n_autograder_examples, target_cluster_percentage=target_cluster_percentage)
            break
        except Exception as e:
            print_and_flush(f"Error running accuracy autograder: {e}")
            time.sleep(5)
    
    print_and_flush(results["avg"])
    print_and_flush(f"Evaluated clustering accuracy in {time.time() - start_time} seconds")
    return results

def evaluate_clustering_completeness(texts, categories, model, n_test_examples, cluster_labels=None):
    """
    Evaluate clustering completeness by measuring how well sentences fit their assigned clusters.
    
    Parameters:
    -----------
    texts : list
        List of texts
    categories : list
        List of tuples (cluster_id, title, description)
    model : str
        Model to use for evaluation
    n_test_examples : int
        Number of examples to use for testing completeness
    cluster_labels : list, optional
        Cluster labels for each text (required for completeness evaluation)
        
    Returns:
    --------
    dict
        Autograder results with detailed analysis
    """
    start_time = time.time()
    
    if cluster_labels is None:
        print_and_flush("Error: cluster_labels are required for completeness evaluation")
        return {
            "error": "cluster_labels are required for completeness evaluation",
            "total_sentences": len(texts),
            "evaluated_sentences": 0,
            "avg_fit_score": 0.0
        }
    
    # Sample n_test_examples randomly from all texts
    if len(texts) > n_test_examples:
        # Get random indices for sampling
        sample_indices = random.sample(range(len(texts)), n_test_examples)
        test_texts = [texts[i] for i in sample_indices]
        test_labels = [cluster_labels[i] for i in sample_indices]
    else:
        test_texts = texts
        test_labels = cluster_labels
    
    # Run completeness autograder on the sampled texts
    for _ in range(3):
        try:
            results = completeness_autograder(test_texts, test_labels, categories, model)
            break
        except Exception as e:
            print_and_flush(f"Error running completeness autograder: {e}")
            time.sleep(5)
    
    print_and_flush(f"Evaluated clustering completeness in {time.time() - start_time} seconds")
    return results


def evaluate_clustering_scoring_metrics(
    texts, cluster_labels, n_clusters, example_activations, cluster_centers, 
    model_name, n_autograder_examples, n_description_examples, existing_categories, 
    repetitions=3, model_id=None, layer=None, clustering_data=None,
    no_accuracy=False, no_completeness=False, no_sem_orth=False, no_orth=False, existing_results=None,
    target_cluster_percentage=0.2
):
    """
    Evaluate clustering using both accuracy and optionally completeness autograders.
    
    Parameters:
    -----------
    texts : list
        List of texts
    cluster_labels : numpy.ndarray
        Cluster labels for each text
    n_clusters : int
        Number of clusters
    example_activations : numpy.ndarray
        Normalized activation vectors
    cluster_centers : numpy.ndarray
        Cluster centers
    model_name : str
        Name of the model to use for generating descriptions
    n_autograder_examples : int
        Number of examples from each cluster to use for autograding
    n_description_examples : int
        Number of examples to use for generating descriptions
    existing_categories : list
        List of existing categories for each repetition
    repetitions : int, default 5
        Number of repetitions to run
    model_id : str, optional
        Model identifier (required for SAE mode)
    layer : int, optional
        Layer number (required for SAE mode)
    clustering_data : dict, optional
        Clustering data (used to determine if SAE mode is needed)
    no_accuracy : bool, optional
        If True, skip accuracy evaluation and use existing results.
    no_completeness : bool, optional
        If True, skip completeness evaluation and use existing results.
    no_sem_orth : bool, optional
        If True, skip semantic orthogonality evaluation and use existing results.
    no_orth : bool, optional
        If True, skip centroid orthogonality evaluation and use existing results.
    existing_results : dict, optional
        Previously computed results to use when skipping evaluations.
        
    Returns:
    --------
    dict
        Combined evaluation results
    """
    start_time = time.time()
    print_and_flush(f" Starting clustering scoring metrics evaluation")

    # Generate representative examples
    representative_examples = generate_representative_examples(
        cluster_centers, texts, cluster_labels, example_activations, 
        clustering_data=clustering_data, model_id=model_id, layer=layer, n_clusters=n_clusters
    )
    
    all_results = []
    for i in range(repetitions):
        print_and_flush(f" ## Running repetition {i+1} of {repetitions}")
        
        # Use existing categories for this repetition
        categories = existing_categories[i]

        # Get existing repetition results if available
        existing_rep_result = {}
        if existing_results and 'results_by_cluster_size' in existing_results and \
           str(n_clusters) in existing_results['results_by_cluster_size'] and \
           'all_results' in existing_results['results_by_cluster_size'][str(n_clusters)] and \
           i < len(existing_results['results_by_cluster_size'][str(n_clusters)]['all_results']):
            existing_rep_result = existing_results['results_by_cluster_size'][str(n_clusters)]['all_results'][i]
        
        rep_results = existing_rep_result.copy()
        rep_results["categories"] = categories

        for cluster_id, title, description in categories:
            print_and_flush(f"Cluster {cluster_id}: {title}")
            print_and_flush(f"\tDescription: {description}")
        print_and_flush(f"")

        # Run binary accuracy autograder (evaluates each cluster independently)
        if not no_accuracy:
            accuracy_results = evaluate_clustering_accuracy(
                texts, cluster_labels, categories, "gpt-4.1-mini", n_autograder_examples, target_cluster_percentage=target_cluster_percentage
            )
            rep_results["avg_accuracy"] = accuracy_results["avg"]["accuracy"]
            rep_results["avg_f1"] = accuracy_results["avg"]["f1"]
            rep_results["avg_precision"] = accuracy_results["avg"]["precision"]
            rep_results["avg_recall"] = accuracy_results["avg"]["recall"]
            rep_results["accuracy_results_by_cluster"] = {k: v for k, v in accuracy_results.items() if k != "avg"}
        else:
            if 'avg_accuracy' not in rep_results:
                print_and_flush(f"WARNING: Accuracy results not found for cluster size {n_clusters} rep {i} and --no-accuracy is set.")
        print_and_flush(f" -> Average F1: {rep_results.get('avg_f1', 0.0):.4f}")
        
        # Compute centroid orthogonality
        if not no_orth:
            orthogonality = compute_centroid_orthogonality(cluster_centers)
            rep_results["orthogonality"] = orthogonality
        else:
            if 'orthogonality' not in rep_results:
                print_and_flush(f"WARNING: Orthogonality results not found for cluster size {n_clusters} rep {i} and --no-orth is set.")
        print_and_flush(f" -> Centroid orthogonality: {rep_results.get('orthogonality', 0.0):.4f}")
        
        # Compute semantic orthogonality
        if not no_sem_orth:
            semantic_orthogonality_results = compute_semantic_orthogonality(categories, "gpt-4.1-mini", 0.5)
            rep_results["semantic_orthogonality_matrix"] = semantic_orthogonality_results["semantic_orthogonality_matrix"]
            rep_results["semantic_orthogonality_explanations"] = semantic_orthogonality_results["semantic_orthogonality_explanations"]
            rep_results["semantic_orthogonality_score"] = semantic_orthogonality_results["semantic_orthogonality_score"]
            rep_results["semantic_orthogonality_threshold"] = semantic_orthogonality_results["semantic_orthogonality_threshold"]
        else:
            if 'semantic_orthogonality_score' not in rep_results:
                print_and_flush(f"WARNING: Semantic orthogonality results not found for cluster size {n_clusters} rep {i} and --no-sem-orth is set.")
        print_and_flush(f" -> Semantic orthogonality score: {rep_results.get('semantic_orthogonality_score', 0.0):.4f}")
        
        # Run completeness autograder
        if not no_completeness:
            str_cluster_labels = [str(label) for label in cluster_labels]
            completeness_results = evaluate_clustering_completeness(texts, categories, "gpt-4.1-mini", 200, str_cluster_labels)
            rep_results["avg_fit_score"] = completeness_results["avg_fit_score"]
            rep_results["avg_fit_score_by_cluster_id"] = completeness_results["avg_fit_score_by_cluster_id"]
            rep_results["completeness_responses"] = completeness_results["responses"]
            # Normalize completeness score to 0-1 scale for compatibility with final score calculation
            rep_results["avg_confidence"] = completeness_results["avg_fit_score"] / 10.0
        else:
            if 'avg_confidence' not in rep_results:
                print_and_flush(f"WARNING: Completeness results not found for cluster size {n_clusters} rep {i} and --no-completeness is set.")
        print_and_flush(f" -> Completeness score: {rep_results.get('avg_fit_score', 0.0):.2f}/10 (normalized: {rep_results.get('avg_confidence', 0.0):.4f})")

        # Calculate final score
        final_score_components = []
        if "avg_f1" in rep_results:
            final_score_components.append(rep_results["avg_f1"])
        else:
            print_and_flush(f"WARNING WHEN COMPUTING FINAL SCORE: avg_f1 not found for cluster size {n_clusters} rep {i} and --no-accuracy is set.")
        if "avg_confidence" in rep_results:
            final_score_components.append(rep_results["avg_confidence"])
        else:
            print_and_flush(f"WARNING WHEN COMPUTING FINAL SCORE: avg_confidence not found for cluster size {n_clusters} rep {i} and --no-completeness is set.")
        if "semantic_orthogonality_score" in rep_results:
            final_score_components.append(rep_results["semantic_orthogonality_score"])
        else:
            print_and_flush(f"WARNING WHEN COMPUTING FINAL SCORE: semantic_orthogonality_score not found for cluster size {n_clusters} rep {i} and --no-sem-orth is set.")

        if final_score_components:
            final_score = sum(final_score_components) / len(final_score_components)
        else:
            final_score = 0.0
            
        rep_results["final_score"] = final_score
        print_and_flush(f" -> Final score: {rep_results['final_score']:.4f}")

        # Create detailed results by cluster
        detailed_results = {}
        accuracy_results = rep_results.get("accuracy_results_by_cluster", {})
        for cluster_id, title, description in categories:
            cluster_id_str = str(cluster_id)
            cluster_metrics = accuracy_results.get(cluster_id_str, {})
            cluster_idx = int(cluster_id)
            cluster_examples = representative_examples[cluster_idx]
            
            detailed_results[cluster_id_str] = {
                'title': title,
                'description': description,
                'size': len(np.where(cluster_labels == cluster_idx)[0]),
                'precision': cluster_metrics.get('precision', 0),
                'recall': cluster_metrics.get('recall', 0),
                'accuracy': cluster_metrics.get('accuracy', 0),
                'f1': cluster_metrics.get('f1', 0),
                'examples': cluster_examples[:15]  # Top 15 examples
            }
            detailed_results[cluster_id]["completeness_score"] = rep_results.get("avg_fit_score_by_cluster_id", {}).get(str(cluster_id), "N/A")
        
        # Add detailed results for this repetition
        rep_results["detailed_results"] = detailed_results
        all_results.append(rep_results)
        print_and_flush("-"*100)

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
            std_dev = np.std(values, ddof=1)  # Sample standard deviation
            sem = stats.sem(values)
            # 95% confidence interval using the t-distribution, which is more accurate for small sample sizes.
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

    print_and_flush(f"Finished clustering scoring metrics evaluation in {time.time() - start_time} seconds")
    
    return {
        "all_results": all_results,
        "avg_final_score": avg_final_score,
        "statistics": statistics
    }

def save_clustering_results(model_id, layer, clustering_method, eval_results_by_cluster_size):
    """
    Save clustering results to a JSON file.

    Parameters:
    -----------
    model_id : str
        Model ID
    layer : int
        Layer
    clustering_method : str
        Clustering method
    eval_results_by_cluster_size : dict
        Evaluation results by cluster size

    Returns:
    --------
    dict
        Clustering results for the given model, layer, and clustering method
    """
    # Define results path and load existing data
    model_short_name = model_id.split("/")[-1].lower()
    results_json_path = (
        f"../train-saes/results/vars/{clustering_method}_results_{model_short_name}_layer{layer}.json"
    )
    results_data = {}
    if os.path.exists(results_json_path):
        try:
            with open(results_json_path, "r") as f:
                results_data = json.load(f)
            print_and_flush(f"Loaded existing results from {results_json_path}")
        except json.JSONDecodeError:
            print_and_flush(
                f"Warning: Could not decode JSON from {results_json_path}. Starting fresh."
            )

    results_data.update({
        "clustering_method": clustering_method,
        "model_id": model_id,
        "layer": layer,
    })

    if "results_by_cluster_size" not in results_data:
        results_data["results_by_cluster_size"] = {}

    for n_clusters, eval_results in eval_results_by_cluster_size.items():
        results_data["results_by_cluster_size"][str(n_clusters)] = eval_results

    # Find the best cluster size
    best_avg_final_score = 0
    best_cluster_size = None
    best_cluster_eval_results = None
    for n_clusters, eval_results in results_data["results_by_cluster_size"].items():
        if eval_results["avg_final_score"] > best_avg_final_score:
            best_avg_final_score = eval_results["avg_final_score"]
            best_cluster_size = n_clusters
            best_cluster_eval_results = eval_results

    assert best_cluster_eval_results is not None, "No best cluster eval results found"

    if "best_cluster" not in results_data:
        results_data["best_cluster"] = {}

    # Save metrics for best cluster size
    results_data["best_cluster"].update({
        "size": best_cluster_size,
        "avg_final_score": best_avg_final_score,
        "avg_accuracy": np.mean([repetition_result.get("avg_accuracy", 0) for repetition_result in best_cluster_eval_results["all_results"]]),
        "avg_precision": np.mean([repetition_result.get("avg_precision", 0) for repetition_result in best_cluster_eval_results["all_results"]]),
        "avg_recall": np.mean([repetition_result.get("avg_recall", 0) for repetition_result in best_cluster_eval_results["all_results"]]),
        "avg_f1": np.mean([repetition_result.get("avg_f1", 0) for repetition_result in best_cluster_eval_results["all_results"]]),
        "avg_completeness": np.mean([repetition_result.get("avg_confidence", 0) for repetition_result in best_cluster_eval_results["all_results"]]),
        "orthogonality": np.mean([repetition_result.get("orthogonality", 0) for repetition_result in best_cluster_eval_results["all_results"]]),
        "semantic_orthogonality": np.mean([repetition_result.get("semantic_orthogonality_score", 0) for repetition_result in best_cluster_eval_results["all_results"]]),
        "statistics": best_cluster_eval_results["statistics"]
    })

    # Convert any numpy types to Python native types for JSON serialization
    results_data = convert_numpy_types(results_data)

    # Save results to JSON
    with open(results_json_path, "w") as f:
        json.dump(results_data, f, indent=2, cls=NumpyEncoder)
    print_and_flush(f"Saved {clustering_method} results to {results_json_path}")

    return results_data


def convert_numpy_types(obj):
    """
    Convert numpy types to Python native types for JSON serialization.
    
    Parameters:
    -----------
    obj : any
        Object to convert
        
    Returns:
    --------
    any
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


# Dictionary mapping clustering method names to their implementations
CLUSTERING_METHODS = {
    'agglomerative': 'clustering_agglomerative',
    'pca_agglomerative': 'clustering_pca_agglomerative',
    'gmm': 'clustering_gmm',
    'pca_gmm': 'clustering_pca_gmm',
    'spherical_kmeans': 'clustering_spherical_kmeans',
    'pca_kmeans': 'clustering_pca_kmeans',
    'sae_topk': 'clustering_sae_topk'
}

# Set of supported clustering methods
SUPPORTED_CLUSTERING_METHODS = {
    'agglomerative',
    'pca_agglomerative',
    'gmm',
    'pca_gmm',
    'spherical_kmeans',
    'pca_kmeans',
    'sae_topk'
} 