# %%
import torch
import matplotlib.pyplot as plt
import argparse

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B")
parser.add_argument("--layer", type=int, default=19)
args, _ = parser.parse_known_args()

model_name = args.model
feature_vectors = torch.load(f"results/vars/mean_vectors_{model_name.split('/')[-1].lower()}_fs3.pt")

def plot_cosine_similarity_heatmap(feature_vectors, model_id):
    labels = list(feature_vectors.keys())
    n_labels = len(labels)
    
    # Create similarity matrix
    similarity_matrix = torch.zeros((n_labels, n_labels))

    for i, label_1 in enumerate(labels):
        for j, label_2 in enumerate(labels):
            layer_idx = args.layer
            similarity_matrix[i, j] = torch.cosine_similarity(
                feature_vectors[label_1][layer_idx], 
                feature_vectors[label_2][layer_idx], 
                dim=-1
            )
                
    # Create heatmap
    plt.figure(figsize=(10, 8))
    im = plt.imshow(similarity_matrix, cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar(label='Cosine Similarity')
    
    # Add labels and text annotations
    plt.xticks(range(n_labels), labels, rotation=45, ha='right')
    plt.yticks(range(n_labels), labels)
    
    # Add text annotations
    for i in range(n_labels):
        for j in range(n_labels):
            value = similarity_matrix[i, j].item()
            # Choose text color based on similarity value
            text_color = 'white' if value > 0.8 else 'black'
            plt.text(j, i, f'{value:.2f}', 
                    ha='center', va='center',
                    color=text_color,
                    fontsize=8)
    
    plt.title('Cosine Similarity Between Feature Vectors')
    plt.tight_layout()
    plt.savefig(f'results/figures/cosine_similarity_heatmap_{model_id}.png', dpi=300)
    plt.show()


# Plot the aggregated heatmap
plot_cosine_similarity_heatmap(feature_vectors, model_id=model_name.split('/')[-1].lower())

# %%
# produce a line plot of the average cosine sim of the overall vector to each other vector for each layer
overall_cosine_sim_values = []
for layer in range(feature_vectors["overall"].shape[0]):
    layer_cosine_sim_values = []
    for other_vector in feature_vectors.keys():
        if other_vector == "overall":
            continue
        layer_cosine_sim_values.append(torch.cosine_similarity(feature_vectors["overall"][layer], feature_vectors[other_vector][layer], dim=-1).mean().item())
    overall_cosine_sim_values.append(sum(layer_cosine_sim_values) / len(layer_cosine_sim_values))

# %%
plt.plot(overall_cosine_sim_values)
plt.show()
