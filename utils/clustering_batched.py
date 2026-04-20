"""
Batch API versions of clustering evaluation methods using OpenAI's batch API.
This module contains methods for generating cluster descriptions and evaluating
clustering accuracy, completeness, and semantic orthogonality using batched requests.
"""

import os
import json
import re
import numpy as np
import time
import random
import tempfile
from openai import OpenAI
from utils.utils import print_and_flush
from utils.clustering import (
    parse_json_response,
)
from utils.autograder_prompts import (
    build_cluster_description_prompt,
    build_accuracy_autograder_prompt,
    build_semantic_orthogonality_prompt,
    build_completeness_autograder_prompt,
    format_sentences_text_simple
)
from utils.responses import extract_thinking_process


def submit_openai_batch(prompts_with_ids, batch_description="Clustering evaluation batch", model="gpt-4.1", temperature=1e-19, max_tokens=28000, json_mode=False):
    """
    Submit a batch of prompts to OpenAI's batch API.
    
    Args:
        prompts_with_ids (dict): Dictionary mapping custom_id to prompt text
        batch_description (str): Description for the batch
        model (str): OpenAI model to use
        temperature (float): Temperature parameter
        max_tokens (int): Maximum tokens per response
        
    Returns:
        str: Batch ID for tracking the submitted batch
    """
    client = OpenAI()
    
    # Create batch requests
    batch_requests = []
    for custom_id, prompt in prompts_with_ids.items():
        request_body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if model.startswith("o3") or model.startswith("o4"):
            request_body["max_completion_tokens"] = max_tokens
            request_body["temperature"] = 1.0
        else:
            request_body["max_tokens"] = max_tokens
            request_body["temperature"] = temperature
        if json_mode:
            request_body["response_format"] = {"type": "json_object"}

        batch_requests.append({
            "custom_id": custom_id,
            "method": "POST", 
            "url": "/v1/chat/completions",
            "body": request_body
        })
    
    # Create temporary JSONL file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for request in batch_requests:
            json.dump(request, f)
            f.write('\n')
        batch_input_path = f.name
    
    try:
        # Upload batch input file
        with open(batch_input_path, 'rb') as f:
            batch_input_file = client.files.create(file=f, purpose="batch")
        
        # Create the batch
        batch = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions", 
            completion_window="24h",
            metadata={"description": batch_description}
        )
        
        print(f"Submitted batch {batch.id} with {len(batch_requests)} requests")
        return batch.id
        
    finally:
        # Clean up temporary file
        os.unlink(batch_input_path)


def process_batch_results(batch_id):
    """
    Process results from a completed OpenAI batch.
    
    Args:
        batch_id (str): The batch ID to process
        
    Returns:
        dict: Dictionary mapping custom_id to response content
    """
    client = OpenAI()
    
    # Get batch status
    batch = client.batches.retrieve(batch_id)
    if batch.status not in ["completed", "expired", "cancelled"]:
        raise ValueError(f"Batch not completed. Current status: {batch.status}")
    
    results = {}

    # Download and process results if an output file exists
    if batch.output_file_id:
        file_response = client.files.content(batch.output_file_id)
        
        # Process each line of the output file
        for line in file_response.text.splitlines():
            result = json.loads(line)
            custom_id = result.get("custom_id")
            
            if custom_id:
                # Check for errors in individual requests
                if result.get("error"):
                    print(f"Error in request {custom_id}: {result['error']}")
                    results[custom_id] = None
                    continue
                    
                # Extract response content
                response = result.get("response", {}).get("body", {})
                choices = response.get("choices", [])
                if choices and choices[0].get("message", {}).get("content"):
                    content = choices[0]["message"]["content"]
                    results[custom_id] = content
                else:
                    print(f"Invalid content in request {custom_id}")
                    results[custom_id] = None
    else:
        print(f"Warning: Batch {batch_id} completed with no output file.")

    # Check for and print errors from the error file, if it exists
    if batch.error_file_id:
        print(f"Batch {batch_id} has an error file. Content:")
        try:
            error_response = client.files.content(batch.error_file_id)
            for line in error_response.text.splitlines():
                try:
                    error = json.loads(line)
                    print(f"Batch error details: {error}")
                except json.JSONDecodeError:
                    print(f"Could not parse error line: {line}")
        except Exception as e:
            print(f"Could not retrieve or process error file for batch {batch_id}: {e}")

    if not results and not batch.error_file_id:
        print(f"Processed batch {batch_id}, which had no successful results and no error file.")
    elif not results and batch.error_file_id:
        print(f"Processed batch {batch_id}, which had no successful results. See error file details above.")
    else:
        print(f"Processed batch {batch_id} with {len(results)} results")

    return results


def check_batch_status(batch_id):
    """
    Check the status of an OpenAI batch.
    
    Args:
        batch_id (str): The batch ID to check
        
    Returns:
        str: The current status of the batch
    """
    client = OpenAI()
    batch = client.batches.retrieve(batch_id)
    return batch.status


def generate_cluster_descriptions_batch(model_name, cluster_examples_list, n_trace_examples=3, n_categories_examples=5, model="o4-mini"):
    """
    Generate descriptions for multiple clusters using batch API.
    
    Args:
        model_name (str): Name of the model whose responses should be loaded for trace examples
        cluster_examples_list (list): List of tuples (cluster_idx, examples) for each cluster
        n_trace_examples (int): Number of full reasoning trace examples to include in prompts
        n_categories_examples (int): Number of category examples to include
        model (str): Model to use for generating descriptions
        
    Returns:
        tuple: (batch_id, cluster_indices) - batch_id for tracking, cluster_indices for processing results
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
    prompts_with_ids = {}
    cluster_indices = []
    
    for cluster_idx, examples in cluster_examples_list:
        # Create a prompt for this cluster using the centralized prompt builder
        prompt = build_cluster_description_prompt(
            examples, 
            trace_examples_text, 
            n_categories_examples
        )
        
        custom_id = f"cluster_description_{cluster_idx}"
        prompts_with_ids[custom_id] = prompt
        cluster_indices.append(cluster_idx)
    
    # Submit batch
    print(f"Submitting batch for {len(prompts_with_ids)} cluster descriptions...")
    batch_id = submit_openai_batch(
        prompts_with_ids, 
        f"Cluster descriptions for {model_name}",
        model=model
    )
    
    return batch_id, cluster_indices


def process_cluster_descriptions_batch(batch_id, cluster_indices):
    """
    Process results from cluster descriptions batch.
    
    Args:
        batch_id (str): The batch ID to process
        cluster_indices (list): List of cluster indices in the same order as submitted
        
    Returns:
        list: List of tuples (cluster_id, title, description) for each cluster
    """
    # Process batch results
    results = process_batch_results(batch_id)
    
    # Parse responses to extract titles and descriptions
    categories = []
    for i, cluster_idx in enumerate(cluster_indices):
        custom_id = f"cluster_description_{cluster_idx}"
        response = results.get(custom_id)
        
        if response:
            # Parse the response to extract title and description
            title = "Unnamed Cluster"
            description = "No description available"
            
            title_match = re.search(r"Title:\s*(.*?)(?:\n|$)", response)
            if title_match:
                title = title_match.group(1).strip()
                
            desc_match = re.search(r"Description:\s*(.*?)(?:\n|$)", response)
            if desc_match:
                description = desc_match.group(1).strip()
            
            categories.append((str(cluster_idx), title, description))
        else:
            print(f"No response received for cluster {cluster_idx}")
            categories.append((str(cluster_idx), "Unnamed Cluster", "No description available"))
    
    return categories


def accuracy_autograder_batch(sentences, categories, ground_truth_labels, n_autograder_examples, max_sentences_per_prompt=50, model="gpt-4.1-mini", target_cluster_percentage=0.2):
    """
    Binary autograder that evaluates each cluster independently against examples from outside the cluster using batch API.
    
    Args:
        sentences (list): List of all sentences to potentially sample from
        categories (list): List of tuples where each tuple is (cluster_id, title, description)
        ground_truth_labels (list): List of cluster IDs (as strings) for each sentence in sentences
        n_autograder_examples (int): Total number of examples to use for testing each cluster
        max_sentences_per_prompt (int): Maximum number of sentences to process per prompt
        model (str): Model to use for the autograding
        target_cluster_percentage (float): Percentage of examples to take from target cluster (default: 0.2)
    
    Returns:
        tuple: (batch_id, metadata) for processing results later
    """
    # Collect all prompts and metadata for batch processing
    prompts_with_ids = {}
    metadata = {
        "categories": categories,
        "batch_metadata": []
    }
    
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
            custom_id = f"accuracy_{cluster_id}_{chunk_idx}"
            prompts_with_ids[custom_id] = prompt
            metadata["batch_metadata"].append({
                "custom_id": custom_id,
                "cluster_id": cluster_id,
                "cluster_id_str": cluster_id_str,
                "title": title,
                "description": description,
                "chunk_idx": chunk_idx,
                "test_sentences": sentence_chunk,
                "test_ground_truth": ground_truth_chunk,
                "test_indices": indices_chunk  # Store original indices for reference
            })
    
    # Submit batch
    print(f"Submitting batch for accuracy evaluation with {len(prompts_with_ids)} prompts...")
    batch_id = submit_openai_batch(
        prompts_with_ids, 
        "Accuracy evaluation",
        model=model,
        max_tokens=4000,
        json_mode=True
    )
    
    return batch_id, metadata


def process_accuracy_batch(batch_id, metadata):
    """
    Process results from accuracy evaluation batch.
    
    Args:
        batch_id (str): The batch ID to process
        metadata (dict): Metadata from the batch submission
        
    Returns:
        dict: Metrics including precision, recall, accuracy and F1 score for each category
    """
    # Process batch results
    results = process_batch_results(batch_id)
    
    # Extract metadata
    categories = metadata["categories"]
    batch_metadata = metadata["batch_metadata"]
    
    # Group responses by cluster_id for aggregation
    cluster_responses = {}
    parsing_errors = []
    
    # Process each response from the batch
    for item in batch_metadata:
        custom_id = item["custom_id"]
        cluster_id_str = item["cluster_id_str"]
        
        if cluster_id_str not in cluster_responses:
            cluster_responses[cluster_id_str] = {
                "metadata": item,  # Store metadata from first chunk (contains cluster info)
                "all_classifications": []
            }
        
        response = results.get(custom_id)
        if response:
            try:
                result = parse_json_response(response, expected_field='classifications')
                
                # Add sentence text and ground truth to each classification
                for classification in result["classifications"]:
                    sentence_idx = classification["sentence_id"]
                    if sentence_idx < len(item["test_sentences"]):
                        classification["sentence_text"] = item["test_sentences"][sentence_idx]
                        classification["ground_truth"] = item["test_ground_truth"][sentence_idx]
                    else:
                        print(f"Warning: sentence_id {sentence_idx} out of range for chunk with {len(item['test_sentences'])} sentences")
                
                # Add the classifications from this chunk to the cluster's overall list
                cluster_responses[cluster_id_str]["all_classifications"].extend(result["classifications"])
                
            except Exception as e:
                print(f"Error parsing response for cluster {item['cluster_id']} chunk {item['chunk_idx']}: {e}")
                print(f"Raw response: {response}")
                parsing_errors.append({
                    "cluster_id": item["cluster_id"],
                    "chunk_idx": item["chunk_idx"],
                    "error": str(e),
                    "raw_response": response
                })
        else:
            print(f"No response received for {custom_id}")
            parsing_errors.append({
                "cluster_id": item["cluster_id"],
                "chunk_idx": item["chunk_idx"],
                "error": "No response received",
                "raw_response": None
            })
    
    # Process aggregated results for each cluster
    final_results = {}
    for cluster_id_str, cluster_data in cluster_responses.items():
        cluster_metadata = cluster_data["metadata"]
        cluster_id = cluster_metadata["cluster_id"]
        all_classifications = cluster_data["all_classifications"]
        
        if not all_classifications:
            print(f"No valid classifications for cluster {cluster_id}")
            final_results[cluster_id_str] = {
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
        
        final_results[cluster_id_str] = {
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
    if final_results:
        valid_results = {k: v for k, v in final_results.items() if "error" not in v}
        if valid_results:
            avg_accuracy = sum(r["accuracy"] for r in valid_results.values()) / len(valid_results)
            avg_precision = sum(r["precision"] for r in valid_results.values()) / len(valid_results)
            avg_recall = sum(r["recall"] for r in valid_results.values()) / len(valid_results)
            avg_f1 = sum(r["f1"] for r in valid_results.values()) / len(valid_results)
            
            final_results["avg"] = {
                "accuracy": avg_accuracy,
                "precision": avg_precision,
                "recall": avg_recall,
                "f1": avg_f1
            }
    
    # Add parsing error summary if any occurred
    if parsing_errors:
        final_results["parsing_errors"] = parsing_errors
    
    return final_results


def completeness_autograder_batch(sentences, cluster_labels, categories, model="gpt-4.1-mini"):
    """
    Autograder that evaluates completeness by measuring how well sentences fit their assigned clusters using batch API.
    
    Args:
        sentences (list): List of sentences to evaluate
        cluster_labels (list): List of cluster IDs (as strings) for each sentence
        categories (list): List of tuples where each tuple is (cluster_id, title, description)
        model (str): Model to use for evaluation
    
    Returns:
        tuple: (batch_id, metadata) for processing results later
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
        return None, {
            "error": "No valid sentence-cluster assignments found",
            "total_sentences": len(sentences),
            "evaluated_sentences": 0,
            "valid_evaluations": []
        }
    
    # Create prompts for all evaluations
    prompts_with_ids = {}
    metadata = {
        "valid_evaluations": valid_evaluations,
        "cluster_id_to_category": cluster_id_to_category,
        "total_sentences": len(sentences)
    }
    
    for eval_idx, (sentence_idx, sentence, cluster_id, title, description) in enumerate(valid_evaluations):
        prompt = build_completeness_autograder_prompt(sentence, title, description)
        custom_id = f"completeness_eval_{eval_idx}"
        prompts_with_ids[custom_id] = prompt
    
    # Submit batch
    print(f"Submitting batch for completeness evaluation with {len(prompts_with_ids)} evaluations...")
    batch_id = submit_openai_batch(
        prompts_with_ids, 
        "Completeness evaluation",
        model=model,
        max_tokens=1000,
        json_mode=True
    )
    
    return batch_id, metadata


def process_completeness_batch(batch_id, metadata):
    """
    Process results from completeness evaluation batch.
    
    Args:
        batch_id (str): The batch ID to process
        metadata (dict): Metadata from the batch submission
        
    Returns:
        dict: Statistics about completeness scores including detailed analysis
    """
    # Handle error case
    if batch_id is None:
        return metadata
    
    # Process batch results
    results = process_batch_results(batch_id)
    
    # Extract metadata
    valid_evaluations = metadata["valid_evaluations"]
    cluster_id_to_category = metadata["cluster_id_to_category"]
    total_sentences = metadata["total_sentences"]
    
    # Process responses
    all_responses = []
    parsing_errors = []
    total_fit_score = 0.0
    
    for eval_idx, (sentence_idx, sentence, cluster_id, title, description) in enumerate(valid_evaluations):
        custom_id = f"completeness_eval_{eval_idx}"
        response = results.get(custom_id)
        
        if response:
            try:
                result = parse_json_response(response)
                
                completeness_score = result.get('completeness_score', 0)
                explanation = result.get('explanation', '')
                
                # Validate completeness score is in range
                if not isinstance(completeness_score, (int, float)) or completeness_score < 0 or completeness_score > 10:
                    print(f"Warning: Invalid completeness score {completeness_score}, clamping to valid range")
                    completeness_score = max(0, min(10, int(completeness_score)))
                
                evaluation_item = {
                    "sentence_idx": sentence_idx,
                    "sentence": sentence,
                    "cluster_id": cluster_id,
                    "title": title,
                    "description": description,
                    "completeness_score": completeness_score,
                    "explanation": explanation
                }
                
                all_responses.append(evaluation_item)
                total_fit_score += completeness_score
                
            except Exception as e:
                print(f"Error parsing completeness evaluation response {eval_idx}: {e}")
                print(f"Raw response: {response}")
                parsing_errors.append({
                    "sentence_idx": sentence_idx,
                    "error": str(e),
                    "raw_response": response
                })
        else:
            print(f"No response received for completeness evaluation {eval_idx}")
            parsing_errors.append({
                "sentence_idx": sentence_idx,
                "error": "No response received",
                "raw_response": None
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
        "total_sentences": total_sentences,
        "evaluated_sentences": num_evaluated,
        "avg_fit_score": avg_fit_score,
        "avg_fit_score_by_cluster_id": avg_fit_score_by_cluster_id,
        "responses": all_responses,
        "parsing_errors": parsing_errors
    }
    
    return result_dict


def compute_semantic_orthogonality_batch(categories, orthogonality_threshold=0.5, model="gpt-4.1-mini"):
    """
    Compute the semantic orthogonality of categories using LLM-based similarity evaluation with batch API.
    
    Parameters:
    -----------
    categories : list
        List of tuples (cluster_id, title, description) for each category
    orthogonality_threshold : float
        Threshold for counting orthogonal pairs (default: 0.5)
    model : str
        Model to use for semantic similarity evaluation
        
    Returns:
    --------
    tuple: (batch_id, metadata) for processing results later
    """
    start_time = time.time()
    n_categories = len(categories)
    
    if n_categories <= 1:
        return None, {
            "orthogonality_matrix": np.array([[0.0]]) if n_categories == 1 else np.array([]),
            "explanations": {},
            "orthogonality_score": 0.0,
            "orthogonality_threshold": orthogonality_threshold,
            "categories": categories
        }
    
    # Prepare batch prompts for all pairs in upper triangle
    prompts_with_ids = {}
    batch_pairs = []
    
    for i in range(n_categories):
        for j in range(i + 1, n_categories):  # Only upper triangle (i < j)
            category1 = categories[i]
            category2 = categories[j]
            
            cluster_id1, title1, description1 = category1
            cluster_id2, title2, description2 = category2
            
            prompt = build_semantic_orthogonality_prompt(title1, description1, title2, description2)
            
            custom_id = f"similarity_{i}_{j}"
            prompts_with_ids[custom_id] = prompt
            batch_pairs.append((i, j))
    
    metadata = {
        "categories": categories,
        "batch_pairs": batch_pairs,
        "orthogonality_threshold": orthogonality_threshold,
        "n_categories": n_categories
    }
    
    # Submit batch
    print(f"Submitting batch for semantic similarity evaluation with {len(prompts_with_ids)} pairs...")
    batch_id = submit_openai_batch(
        prompts_with_ids, 
        "Semantic similarity evaluation",
        model=model,
        max_tokens=1000,
        json_mode=True
    )
    
    print_and_flush(f"Submitted semantic orthogonality batch in {time.time() - start_time} seconds")
    
    return batch_id, metadata


def process_semantic_orthogonality_batch(batch_id, metadata):
    """
    Process results from semantic orthogonality evaluation batch.
    
    Args:
        batch_id (str): The batch ID to process
        metadata (dict): Metadata from the batch submission
        
    Returns:
        dict: Dictionary containing orthogonality matrix and metrics
    """
    start_time = time.time()
    
    # Handle single category case
    if batch_id is None:
        return metadata
    
    # Process batch results
    results = process_batch_results(batch_id)
    
    # Extract metadata
    categories = metadata["categories"]
    batch_pairs = metadata["batch_pairs"]
    orthogonality_threshold = metadata["orthogonality_threshold"]
    n_categories = metadata["n_categories"]
    
    # Initialize similarity matrix and explanations dictionary
    similarity_matrix = np.zeros((n_categories, n_categories))
    explanations = {}
    
    # Fill diagonal with 1.0 (perfect self-similarity)
    np.fill_diagonal(similarity_matrix, 1.0)
    
    # Parse responses and fill similarity matrix
    for idx, (i, j) in enumerate(batch_pairs):
        custom_id = f"similarity_{i}_{j}"
        response = results.get(custom_id)
        
        if response:
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
        else:
            print(f"No response received for similarity pair ({i}, {j})")
            # Default to 0 similarity on no response
            similarity_matrix[i, j] = 0.0
            similarity_matrix[j, i] = 0.0
            # Default explanation on no response
            explanations[f"{i},{j}"] = "No response received"
            explanations[f"{j},{i}"] = "No response received"
    
    # Calculate orthogonality as 1 - similarity
    orthogonality_matrix = 1 - similarity_matrix
    
    # Get the indices of the upper triangular part (excluding diagonal)
    indices = np.triu_indices(orthogonality_matrix.shape[0], k=1)
    
    # Extract the upper triangular values
    upper_tri_values = orthogonality_matrix[indices]
    
    # Calculate orthogonality score (fraction of pairs above threshold)
    orthogonality_score = np.sum(upper_tri_values > orthogonality_threshold) / len(upper_tri_values) if len(upper_tri_values) > 0 else 0
    
    print_and_flush(f"Processed semantic orthogonality batch in {time.time() - start_time} seconds")
    
    return {
        "semantic_orthogonality_matrix": orthogonality_matrix,
        "semantic_orthogonality_explanations": explanations,
        "semantic_orthogonality_score": orthogonality_score,
        "semantic_orthogonality_threshold": orthogonality_threshold
    }