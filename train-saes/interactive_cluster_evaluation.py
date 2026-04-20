# %%
import random
import numpy as np
import torch
import gc
import matplotlib.pyplot as plt

# Import utility functions
from utils.clustering import (
    SUPPORTED_CLUSTERING_METHODS,
    load_trained_clustering_data, predict_clusters,
    generate_category_descriptions, evaluate_clustering_accuracy, compute_centroid_orthogonality,
    generate_representative_examples, evaluate_clustering_completeness, compute_semantic_orthogonality
)
from utils import utils

# Note: If you get asyncio errors in Jupyter, install nest_asyncio:
# !pip install nest_asyncio

print("Available clustering methods:", list(SUPPORTED_CLUSTERING_METHODS))

# %%
# Configuration - modify these parameters as needed
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
LAYER = 6
N_EXAMPLES = 100000
N_CLUSTERS = 30 # 29
CLUSTERING_METHOD = "sae_topk"  # Choose from SUPPORTED_CLUSTERING_METHODS
LOAD_IN_8BIT = False

# Get model identifier for file naming
model_id = MODEL_NAME.split('/')[-1].lower()

print(f"Configuration:")
print(f"Model: {MODEL_NAME}")
print(f"Model ID: {model_id}")
print(f"Layer: {LAYER}")
print(f"Number of examples: {N_EXAMPLES}")
print(f"Number of clusters: {N_CLUSTERS}")
print(f"Clustering method: {CLUSTERING_METHOD}")

# %%

# Clustering evaluation config
REPETITIONS = 3

MODEL_NAME_FOR_CATEGORY_DESCRIPTIONS = "o4-mini"
N_DESCRIPTION_EXAMPLES = 200

MODEL_NAME_FOR_COMPLETENESS_EVALUATION = "gpt-4.1-mini"
N_COMPLETENESS_EXAMPLES = 500

MODEL_NAME_FOR_ACCURACY_EVALUATION = "gpt-4.1-mini"
N_ACCURACY_EXAMPLES = 100

MODEL_NAME_FOR_SEMANTIC_ORTHOGONALITY = "gpt-4.1-mini"

# %%
# Load model and process activations
print("Loading model and processing activations...")
model, tokenizer = utils.load_model(
    model_name=MODEL_NAME,
    load_in_8bit=LOAD_IN_8BIT
)

print(f"Model loaded successfully. Model type: {type(model)}")

# %%
# Process saved responses to get activations
print("Processing saved responses...")
all_activations, all_texts = utils.process_saved_responses(
    MODEL_NAME, 
    N_EXAMPLES,
    model,
    tokenizer,
    LAYER
)

print(f"Processed {len(all_activations)} activations")
print(f"Activation shape: {all_activations[0].shape}")
print(f"Number of texts: {len(all_texts)}")

# Clean up GPU memory
del model, tokenizer
torch.cuda.empty_cache()
gc.collect()

# %%

# Show examples of random texts
sample_size = 500
sample = random.sample(all_texts, sample_size)
for i in range(sample_size):
    print(sample[i])
    print("-"*100)

# %%
print(f"Final activation matrix shape: {all_activations.shape}")

# %%
# Load trained clustering data
print(f"Loading trained clustering data for {CLUSTERING_METHOD}...")
try:
    clustering_data = load_trained_clustering_data(model_id, LAYER, N_CLUSTERS, CLUSTERING_METHOD)
    print("Clustering data loaded successfully!")
    print(f"Available keys in clustering data: {list(clustering_data.keys())}")
    
    if 'cluster_centers' in clustering_data:
        cluster_centers = clustering_data['cluster_centers']
        print(f"Cluster centers shape: {cluster_centers.shape}")
    else:
        print("No cluster_centers found in clustering data")
        
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Make sure you have trained clustering models for this configuration first.")

# %%
# Predict cluster labels for current activations
print("Predicting cluster labels...")
cluster_labels = predict_clusters(all_activations, clustering_data)

print(f"Cluster labels shape: {cluster_labels.shape}")
print(f"Unique cluster labels: {np.unique(cluster_labels)}")

# Count examples per cluster
unique_labels, counts = np.unique(cluster_labels, return_counts=True)
cluster_counts = dict(zip(unique_labels, counts))
print(f"Cluster distribution: {cluster_counts}")

# %%
# Visualize cluster distribution
plt.figure(figsize=(10, 6))
plt.bar(list(cluster_counts.keys()), list(cluster_counts.values()))
plt.title(f'Cluster Distribution - {CLUSTERING_METHOD} (Layer {LAYER})')
plt.xlabel('Cluster ID')
plt.ylabel('Number of Examples')
plt.xticks(list(cluster_counts.keys()))
for i, count in cluster_counts.items():
    plt.text(i, count + 1, str(count), ha='center')
plt.show()

# %%
# Compute centroid orthogonality
print("Computing centroid orthogonality...")
avg_orthogonality = compute_centroid_orthogonality(cluster_centers)
print(f"Average centroid orthogonality: {avg_orthogonality:.4f}")

# Plot orthogonality between each cluster center
norm_cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1, keepdims=True)
cosine_sim = np.dot(norm_cluster_centers, norm_cluster_centers.T)
abs_cosine_sim = np.abs(cosine_sim)
orthogonality = 1 - abs_cosine_sim

plt.figure(figsize=(10, 8))
im = plt.imshow(orthogonality, cmap='viridis', interpolation='nearest')
plt.colorbar(im, label='Orthogonality (1 - similarity)')
plt.title(f'Orthogonality Matrix Between Cluster Centers - {CLUSTERING_METHOD} (Layer {LAYER})')
plt.xlabel('Cluster ID')
plt.ylabel('Cluster ID')
plt.xticks(range(len(cluster_centers)))
plt.yticks(range(len(cluster_centers)))

# Add text annotations for values
for i in range(len(cluster_centers)):
    for j in range(len(cluster_centers)):
        plt.text(j, i, f'{orthogonality[i, j]:.2f}', 
                ha='center', va='center', color='white' if orthogonality[i, j] < 0.5 else 'black')

plt.tight_layout()
plt.show()

# %%
# Generate representative examples for each cluster
print("Generating representative examples...")
representative_examples = generate_representative_examples(
    cluster_centers, all_texts, cluster_labels, all_activations, 
    clustering_data=clustering_data, model_id=model_id, layer=LAYER, n_clusters=N_CLUSTERS
)

print("Representative examples by cluster:")
n_top_examples_per_cluster = 25
n_bottom_examples_per_cluster = 0
for cluster_idx in sorted(representative_examples.keys()):
    examples = representative_examples[cluster_idx]
    n_examples = len(examples)
    print(f"\nCluster {cluster_idx} ({n_examples} examples):")
    # Print top N
    for i, example in enumerate(examples[:n_top_examples_per_cluster]):
        print(f"  {i+1}. {example}")
    # Print "... and X more" if there are more than top+bottom
    n_more = n_examples - (n_top_examples_per_cluster + n_bottom_examples_per_cluster)
    if n_more > 0:
        print(f"  ... and {n_more} more")
    if n_bottom_examples_per_cluster > 0:
        # Print bottom N
        for i, example in enumerate(examples[-n_bottom_examples_per_cluster:]):
            print(f"  {n_examples - n_bottom_examples_per_cluster + i + 1}. {example}")

# %%
# Generate category descriptions
print("Generating category descriptions...")
categories = generate_category_descriptions(
    cluster_centers, 
    MODEL_NAME, 
    MODEL_NAME_FOR_CATEGORY_DESCRIPTIONS, 
    N_DESCRIPTION_EXAMPLES, 
    representative_examples,
    n_trace_examples=0,
    n_categories_examples=5
)

print("Generated category descriptions:")
for cluster_id, title, description in categories:
    print(f"\nCluster {cluster_id}:")
    print(f"  Title: {title}")
    print(f"  Description: {description}")

title_by_cluster = {cluster_id: title for cluster_id, title, description in categories}

#%%
print("Computing semantic orthogonality...")
semantic_orthogonality_results = compute_semantic_orthogonality(categories, MODEL_NAME_FOR_SEMANTIC_ORTHOGONALITY)
print(f"Semantic orthogonality: {semantic_orthogonality_results['semantic_orthogonality_score']}")

for i in range(len(cluster_centers)):
    for j in range(i+1, len(cluster_centers)):
        print(f"Cluster {i} and Cluster {j} orthogonality: {semantic_orthogonality_results['semantic_orthogonality_matrix'][i, j]} -> {semantic_orthogonality_results['semantic_orthogonality_explanations'][f'{i},{j}']}")

# %%

# Show the cluster titles and descriptions of pairs of clusters with semantic orthogonality 0
print(f"Pairs of clusters with semantic orthogonality below threshold: {semantic_orthogonality_results['semantic_orthogonality_threshold']}")
for i in range(len(cluster_centers)):
    for j in range(i+1, len(cluster_centers)):
        if semantic_orthogonality_results['semantic_orthogonality_matrix'][i, j] < semantic_orthogonality_results['semantic_orthogonality_threshold']:
            print(f"Cluster {i} and Cluster {j} semantic orthogonality: {semantic_orthogonality_results['semantic_orthogonality_matrix'][i, j]}")
            print(f"- Explanation: {semantic_orthogonality_results['semantic_orthogonality_explanations'][f'{i},{j}']}")
            print(f"- Cluster {i}: {title_by_cluster[str(i)]}")
            print(f"\tDescription: {categories[i][2]}")
            print(f"- Cluster {j}: {title_by_cluster[str(j)]}")
            print(f"\tDescription: {categories[j][2]}")
            print("-"*100)

# %%

# Plot semantic orthogonality matrix
plt.figure(figsize=(12, 10))
im = plt.imshow(semantic_orthogonality_results['semantic_orthogonality_matrix'], cmap='viridis', interpolation='nearest')
plt.colorbar(im, label='Orthogonality (1 - |cosine similarity|)')
plt.title(f'Semantic Orthogonality Matrix - {CLUSTERING_METHOD} (Layer {LAYER})')
plt.xlabel('Cluster ID')
plt.ylabel('Cluster ID')
plt.xticks(range(len(cluster_centers)))
plt.yticks(range(len(cluster_centers)))

# Add text annotations for values
for i in range(len(cluster_centers)):
    for j in range(len(cluster_centers)):
        plt.text(j, i, f'{semantic_orthogonality_results["semantic_orthogonality_matrix"][i, j]:.2f}', 
                ha='center', va='center', color='white' if semantic_orthogonality_results["semantic_orthogonality_matrix"][i, j] < 0.5 else 'black', fontsize=8)

plt.tight_layout()
plt.show()

# %%
print("Running completeness evaluation...")
completeness_results = evaluate_clustering_completeness(
    all_texts,
    categories,
    MODEL_NAME_FOR_COMPLETENESS_EVALUATION,
    N_COMPLETENESS_EXAMPLES,
    [str(label) for label in cluster_labels]
)
print(f"Completeness results: {completeness_results}")
print(f"Total sentences evaluated: {completeness_results['total_sentences']}")
print(f"Assigned sentences: {completeness_results['assigned']} ({completeness_results['assigned_fraction']:.2f})")
print(f"Not assigned sentences: {completeness_results['not_assigned']} ({completeness_results['not_assigned_fraction']:.2f})")
print(f"Average confidence: {completeness_results.get('avg_confidence', 0.0):.2f}")
print(f"Category average confidences: {completeness_results.get('category_avg_confidences', {})}")

# %%
# Display detailed completeness analysis
print("\n" + "="*80)
print("DETAILED COMPLETENESS ANALYSIS")
print("="*80)
print("Shows detailed breakdown of how sentences were assigned during completeness evaluation")

if "detailed_analysis" in completeness_results:
    detailed_analysis = completeness_results["detailed_analysis"]
    completeness_metrics = completeness_results.get("completeness_metrics", {})
    
    print(f"\n{'='*60}")
    print(f"COMPLETENESS METRICS SUMMARY")
    print(f"{'='*60}")
    print(f"Total sentences evaluated: {completeness_results['total_sentences']}")
    print(f"Assignment accuracy: {completeness_metrics.get('assignment_accuracy', 0):.4f}")
    print(f"Assignment recall: {completeness_metrics.get('assignment_recall', 0):.4f}")
    print(f"Assignment precision: {completeness_metrics.get('assignment_precision', 0):.4f}")
    print(f"Average confidence for correct assignments: {completeness_metrics.get('avg_correct_confidence', 0):.4f}")
    print(f"Average confidence for incorrect assignments: {completeness_metrics.get('avg_incorrect_confidence', 0):.4f}")
    
    print(f"\nCorrect assignments: {completeness_metrics.get('correct_assignments', 0)}")
    print(f"Incorrect assignments: {completeness_metrics.get('incorrect_assignments', 0)}")
    print(f"Missed assignments: {completeness_metrics.get('missed_assignments', 0)}")
    print(f"Spurious assignments: {completeness_metrics.get('spurious_assignments', 0)}")
    print(f"Correctly unassigned: {completeness_metrics.get('unassigned_correctly', 0)}")
    
    # Show examples of each type
    print(f"\n🟢 CORRECT ASSIGNMENTS ({len(detailed_analysis['correct_assignments'])}): Assigned to correct category")
    enumeration_length = 100
    for i, item in enumerate(detailed_analysis['correct_assignments'][:enumeration_length]):
        ground_truth_category_id = item['ground_truth_category']
        assigned_category_id = item['assigned_category'].split(" ")[1]
        confidence = item.get('confidence', 0)
        print(f"  {i+1}. Ground truth: Category {item['ground_truth_category']} ({title_by_cluster[str(ground_truth_category_id)]}) | Assigned: {item['assigned_category']} ({title_by_cluster[str(assigned_category_id)]}) | Confidence: {confidence}")
        print(f"     Explanation: {item['explanation']}")
        print(f"     Text: {item['sentence_text']}")
    if len(detailed_analysis['correct_assignments']) > enumeration_length:
        print(f"     ... and {len(detailed_analysis['correct_assignments']) - enumeration_length} more")
    
    print(f"\n🔴 MISSED ASSIGNMENTS ({len(detailed_analysis['missed_assignments'])}): Should have been assigned but weren't")
    for i, item in enumerate(detailed_analysis['missed_assignments'][:enumeration_length]):
        ground_truth_category_id = item['ground_truth_category']
        confidence = item.get('confidence', 0)
        print(f"  {i+1}. Ground truth: Category {item['ground_truth_category']} ({title_by_cluster[str(ground_truth_category_id)]}) | Assigned: {item['assigned_category']} | Confidence: {confidence}")
        print(f"     Explanation: {item['explanation']}")
        print(f"     Text: {item['sentence_text']}")
    if len(detailed_analysis['missed_assignments']) > enumeration_length:
        print(f"     ... and {len(detailed_analysis['missed_assignments']) - enumeration_length} more")
    
    print(f"\n🟡 INCORRECT ASSIGNMENTS ({len(detailed_analysis['incorrect_assignments'])}): Assigned to wrong category")
    for i, item in enumerate(detailed_analysis['incorrect_assignments'][:enumeration_length]):
        ground_truth_category_id = item['ground_truth_category']
        assigned_category_id = item['assigned_category'].split(" ")[1]
        confidence = item.get('confidence', 0)
        print(f"  {i+1}. Ground truth: Category {item['ground_truth_category']} ({title_by_cluster[str(ground_truth_category_id)]}) | Assigned: {item['assigned_category']} ({title_by_cluster[str(assigned_category_id)]}) | Confidence: {confidence}")
        print(f"     Explanation: {item['explanation']}")
        print(f"     Text: {item['sentence_text']}")
    if len(detailed_analysis['incorrect_assignments']) > enumeration_length:
        print(f"     ... and {len(detailed_analysis['incorrect_assignments']) - enumeration_length} more")

# %%
# Run accuracy evaluation (binary autograder)
print("Running accuracy evaluation...")
accuracy_results = evaluate_clustering_accuracy(
    all_texts, cluster_labels, categories, MODEL_NAME_FOR_ACCURACY_EVALUATION, N_ACCURACY_EXAMPLES
)

print("\nPer-cluster results:")
for cluster_id, title, description in categories:
    if cluster_id in accuracy_results:
        results = accuracy_results[cluster_id]
        print(f"  Cluster {cluster_id} ({title}):")
        print(f"    Accuracy: {results.get('accuracy', 0):.4f}")
        print(f"    Precision: {results.get('precision', 0):.4f}")
        print(f"    Recall: {results.get('recall', 0):.4f}")
        print(f"    F1: {results.get('f1', 0):.4f}")

print("Accuracy evaluation results:")
if "avg" in accuracy_results:
    avg_results = accuracy_results["avg"]
    print(f"  Average accuracy: {avg_results['accuracy']:.4f}")
    print(f"  Average precision: {avg_results['precision']:.4f}")
    print(f"  Average recall: {avg_results['recall']:.4f}")
    print(f"  Average F1: {avg_results['f1']:.4f}")

# %%
# Display detailed classification results for each cluster
print("\n" + "="*80)
print("DETAILED CLASSIFICATION ANALYSIS")
print("="*80)
print("Shows which sentences were correctly/incorrectly classified during accuracy evaluation")

for cluster_id, title, description in categories:
    cluster_id_str = str(cluster_id)
    if cluster_id_str not in accuracy_results or "classifications" not in accuracy_results[cluster_id_str]:
        continue
        
    cluster_results = accuracy_results[cluster_id_str]
    classifications = cluster_results["classifications"]
    
    print(f"\n{'='*60}")
    print(f"CLUSTER {cluster_id}: {title}")
    print(f"{'='*60}")
    print(f"Description: {description}")
    print(f"Metrics: TP={cluster_results.get('true_positives', 0)}, "
          f"FP={cluster_results.get('false_positives', 0)}, "
          f"TN={cluster_results.get('true_negatives', 0)}, "
          f"FN={cluster_results.get('false_negatives', 0)}")
    
    # Group classifications by type
    true_positives = []   # Correctly identified as belonging to cluster
    false_negatives = []  # Incorrectly identified as NOT belonging to cluster  
    true_negatives = []   # Correctly identified as NOT belonging to cluster
    false_positives = []  # Incorrectly identified as belonging to cluster
    
    # Use ground truth information stored in enhanced classifications
    for item in classifications:
        sentence_idx = item["sentence_id"]
        prediction = item["belongs_to_category"]  # "Yes" or "No"
        ground_truth = item.get("ground_truth", "Unknown")  # "Yes" or "No"
        explanation = item.get("explanation", "")
        sentence_text = item.get("sentence_text", f"Sentence {sentence_idx}")
        
        if ground_truth == "Yes":
            # Should belong to cluster
            if prediction == "Yes":
                true_positives.append((sentence_text, explanation))
            else:
                false_negatives.append((sentence_text, explanation))
        else:
            # Should NOT belong to cluster
            if prediction == "No":
                true_negatives.append((sentence_text, explanation))
            else:
                false_positives.append((sentence_text, explanation))
    
    # Display results
    print(f"\n🟢 TRUE POSITIVES ({len(true_positives)}): Correctly identified as belonging to cluster")
    enumeration_length = 5
    for i, (sentence_text, explanation) in enumerate(true_positives[:enumeration_length]):
        print(f"  {i+1}. {explanation}")
        print(f"     Text: {sentence_text}")
    if len(true_positives) > enumeration_length:
        print(f"     ... and {len(true_positives) - enumeration_length} more")
    
    print(f"\n🔴 FALSE NEGATIVES ({len(false_negatives)}): Incorrectly identified as NOT belonging to cluster")
    for i, (sentence_text, explanation) in enumerate(false_negatives[:enumeration_length]):
        print(f"  {i+1}. {explanation}")
        print(f"     Text: {sentence_text}")
    if len(false_negatives) > enumeration_length:
        print(f"     ... and {len(false_negatives) - enumeration_length} more")
    
    print(f"\n🟢 TRUE NEGATIVES ({len(true_negatives)}): Correctly identified as NOT belonging to cluster")
    for i, (sentence_text, explanation) in enumerate(true_negatives[:enumeration_length]):  # Show first 3
        print(f"  {i+1}. {explanation}")
        print(f"     Text: {sentence_text}")
    if len(true_negatives) > enumeration_length:
        print(f"     ... and {len(true_negatives) - enumeration_length} more")
    
    print(f"\n🔴 FALSE POSITIVES ({len(false_positives)}): Incorrectly identified as belonging to cluster")
    for i, (sentence_text, explanation) in enumerate(false_positives[:enumeration_length]):  # Show first 3
        print(f"  {i+1}. {explanation}")
        print(f"     Text: {sentence_text}")
    if len(false_positives) > enumeration_length:
        print(f"     ... and {len(false_positives) - enumeration_length} more")
# %%
