# %%
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.utils import load_steering_vectors
import argparse

def compute_cosine_similarities(vectors_dict):
    """Compute pairwise cosine similarities between vectors."""
    categories = list(vectors_dict.keys())
    n = len(categories)
    similarities = torch.zeros((n, n))
    
    for i, cat1 in enumerate(categories):
        for j, cat2 in enumerate(categories):
            vec1 = vectors_dict[cat1].flatten()
            vec2 = vectors_dict[cat2].flatten()
            similarity = torch.nn.functional.cosine_similarity(vec1, vec2, dim=0)
            similarities[i, j] = similarity
            
    return similarities, categories

def plot_similarity_heatmap(similarities, categories, save_path=None):
    """Create and optionally save a heatmap of vector similarities."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        similarities,
        xticklabels=categories,
        yticklabels=categories,
        annot=True,  # Show numerical values
        fmt='.2f',   # Format to 2 decimal places
        cmap='RdBu_r',  # Red-Blue diverging colormap
        vmin=-1,
        vmax=1,
        center=0
    )
    plt.title('Cosine Similarity Between Steering Vectors')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")

    plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze similarities between steering vectors")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B",
                        help="Model name to analyze vectors for")
    parser.add_argument("--save_path", type=str, default="results/figures/vector_similarities.pdf",
                        help="Path to save the heatmap (optional)")
    args, _ = parser.parse_known_args()
    
    # Load vectors
    print("Loading steering vectors...")
    vectors = load_steering_vectors(verbose=True)
    
    if not vectors:
        print("No vectors found!")
        return
        
    print(f"\nFound {len(vectors)} vectors: {', '.join(vectors.keys())}")
    
    # Compute similarities
    print("\nComputing cosine similarities...")
    similarities, categories = compute_cosine_similarities(vectors)
    
    # Plot heatmap
    print("\nPlotting heatmap...")
    plot_similarity_heatmap(similarities, categories, args.save_path)
    
    # Print some statistics
    print("\nSummary statistics:")
    similarities_np = similarities.numpy()
    np.fill_diagonal(similarities_np, np.nan)  # Exclude diagonal for stats
    print(f"Mean similarity (excluding diagonal): {np.nanmean(similarities_np):.3f}")
    print(f"Min similarity: {np.nanmin(similarities_np):.3f}")
    print(f"Max similarity: {np.nanmax(similarities_np):.3f}")
    
    # Find most similar and dissimilar pairs
    n = len(categories)
    most_similar = float('-inf')
    most_dissimilar = float('inf')
    most_similar_pair = None
    most_dissimilar_pair = None
    
    for i in range(n):
        for j in range(i + 1, n):  # Only look at upper triangle
            sim = similarities_np[i, j]
            if sim > most_similar:
                most_similar = sim
                most_similar_pair = (categories[i], categories[j])
            if sim < most_dissimilar:
                most_dissimilar = sim
                most_dissimilar_pair = (categories[i], categories[j])
    
    print(f"\nMost similar pair ({most_similar:.3f}): {most_similar_pair[0]} & {most_similar_pair[1]}")
    print(f"Most dissimilar pair ({most_dissimilar:.3f}): {most_dissimilar_pair[0]} & {most_dissimilar_pair[1]}")

if __name__ == "__main__":
    main() 
# %%
