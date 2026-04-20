# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import json
from sklearn.decomposition import PCA
import seaborn as sns
import html
import matplotlib.colors as mcolors
import colorsys
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
from utils.clustering import get_latent_descriptions
from utils.utils import load_model
from utils.sae import load_sae

# %% Parse arguments
parser = argparse.ArgumentParser(description="SAE Visualization and Analysis")
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help="Model to analyze")
parser.add_argument("--layer", type=int, default=6, help="Layer to analyze")
parser.add_argument("--n_clusters", type=int, default=19, help="Number of clusters in the SAE")
parser.add_argument("--topk", type=int, default=1, help="Number of top activations to keep in SAE")
args, _ = parser.parse_known_args()

# %% Function to load SAE from saved file
def create_combined_visualization(sae, save_path=None):
    """Create a combined visualization with cosine similarity and PCA side by side"""
    # Calculate rows needed for the legend based on number of latents
    decoder_weights = sae.W_dec.data.cpu().numpy()
    n_latents = decoder_weights.shape[0]
    
    # Create a figure with visualizations only
    plt.figure(figsize=(20, 10))
    
    # Set up the main grid for the two visualizations
    gs = GridSpec(1, 2)
    
    # 1. Cosine similarity heatmap (left)
    ax1 = plt.subplot(gs[0, 0])
    
    # Generate distinct colors
    hex_colors = generate_distinct_colors(n_latents)
    
    # Create cluster info dictionary
    cluster_info = get_latent_descriptions(model_id, args.layer, args.n_clusters)
    
    # Normalize decoder weights for cosine similarity
    normalized_weights = decoder_weights / np.linalg.norm(decoder_weights, axis=1, keepdims=True)
    
    # Compute cosine similarity matrix
    cosine_sim = np.dot(normalized_weights, normalized_weights.T)
    
    # Create heatmap using seaborn with RdBu colormap (red to blue)
    ax1 = sns.heatmap(cosine_sim, cmap="RdBu", vmin=-1, vmax=1, center=0, 
                   xticklabels=range(cosine_sim.shape[0]), 
                   yticklabels=range(cosine_sim.shape[0]), 
                   square=True, ax=ax1)
    
    # Color the tick labels to match the latent colors
    for i, tick in enumerate(ax1.get_xticklabels()):
        tick.set_color(hex_colors[i])
        tick.set_fontweight('bold')
    for i, tick in enumerate(ax1.get_yticklabels()):
        tick.set_color(hex_colors[i])
        tick.set_fontweight('bold')
    
    ax1.set_title(f"Cosine Similarity Between SAE Decoder Latents\n(n_clusters={decoder_weights.shape[0]})")
    ax1.set_xlabel("Latent Feature")
    ax1.set_ylabel("Latent Feature")
    
    # 2. PCA visualization (right)
    ax2 = plt.subplot(gs[0, 1])
    
    # Apply PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    latent_pca = pca.fit_transform(decoder_weights)
    
    # Plot points with corresponding colors
    for i, (x, y) in enumerate(latent_pca):
        ax2.scatter(x, y, s=100, color=hex_colors[i], alpha=0.8)
        # Draw lines from origin to each point
        ax2.plot([0, x], [0, y], color=hex_colors[i], alpha=0.4, linewidth=1)
        ax2.text(x, y, str(i), fontsize=10, ha='center', va='center', 
                color='white', fontweight='bold',
                bbox=dict(facecolor=hex_colors[i], alpha=0.8, edgecolor='none', pad=2))
    
    explained_var = pca.explained_variance_ratio_
    ax2.set_title(f"PCA Visualization of SAE Decoder Latents\nExplained variance: {sum(explained_var):.2f}")
    ax2.set_xlabel(f"PC1 ({explained_var[0]:.2f})")
    ax2.set_ylabel(f"PC2 ({explained_var[1]:.2f})")
    ax2.grid(alpha=0.3)
    
    # Add origin point
    ax2.scatter([0], [0], color='red', s=100, marker='x')
    
    # Equal aspect ratio
    ax2.axis('equal')
    
    plt.tight_layout()
    
    # Create a separate figure for the legend with a tabular design
    legend_cols = 3  # Three columns for the legend
    legend_rows = (n_latents + legend_cols - 1) // legend_cols
    
    # Calculate a proportional legend figure size
    legend_fig = plt.figure(figsize=(18, 2 + legend_rows * 0.3))
    
    # Create a table for the legend
    cell_text = []
    cell_colors = []
    
    for row in range(legend_rows):
        text_row = []
        color_row = []
        for col in range(legend_cols):
            idx = row * legend_cols + col
            if idx < n_latents:
                latent_title = cluster_info.get(idx, {}).get('title', f'Latent {idx}')
                text_row.append(f'{idx}: {latent_title}')
                # Create gradient cell with white text background for readability
                color_row.append([hex_colors[idx], 'white'])
            else:
                text_row.append('')
                color_row.append(['white', 'white'])
        cell_text.append(text_row)
        cell_colors.append(color_row)
    
    # Create table with colored cells
    ax_table = legend_fig.add_subplot(111)
    ax_table.axis('off')
    
    # Function to create colored cell with number
    def color_cell_with_number(cell_colors, cell_text):
        cells = []
        for row_colors, row_texts in zip(cell_colors, cell_text):
            row_cells = []
            for color_pair, text in zip(row_colors, row_texts):
                if not text:  # Empty cell
                    row_cells.append('')
                    continue
                    
                idx_text = text.split(':', 1)[0]
                desc_text = text.split(':', 1)[1] if ':' in text else ''
                
                # Create formatted cell with colored box and text
                cell = f'<td style="border: 1px solid #ddd; padding: 6px;">'
                cell += f'<div style="display: flex; align-items: center;">'
                cell += f'<div style="background-color: {color_pair[0]}; width: 15px; height: 15px; margin-right: 6px; border: 1px solid black;"></div>'
                cell += f'<span style="font-weight: bold; margin-right: 4px;">{idx_text}:</span>'
                cell += f'<span>{desc_text}</span>'
                cell += f'</div></td>'
                
                row_cells.append(cell)
            cells.append(f'<tr>{"".join(row_cells)}</tr>')
        
        return f'<table style="border-collapse: collapse; width: 100%;">{"".join(cells)}</table>'
    
    # Create HTML-style table for the legend
    legend_html = color_cell_with_number(cell_colors, cell_text)
    
    # Use matplotlib's ability to render HTML
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    ax_table.annotate(
        "",
        xy=(0.5, 0.5),
        xycoords="axes fraction",
        bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.9),
        annotation_clip=False
    )
    
    # Use a table instead of HTML for better rendering
    the_table = ax_table.table(
        cellText=cell_text,
        cellLoc='left',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Style the table
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(9)
    the_table.scale(1, 1.5)
    
    # Color the cells in the first part of each cell
    for (i, j), cell in the_table.get_celld().items():
        if i < len(cell_text) and j < len(cell_text[0]):
            idx_text = cell_text[i][j].split(':', 1)[0] if cell_text[i][j] else ''
            if idx_text:
                idx = int(idx_text)
                # Add colored rectangle to cell
                cell.get_text().set_text(cell_text[i][j])
                cell.set_facecolor('#f8f8f8')
                
                # Add colored box before text
                fig_box = legend_fig.add_axes([cell.get_x()+0.005, cell.get_y()+0.01, 0.01, 0.01])
                fig_box.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=hex_colors[idx]))
                fig_box.axis('off')
    
    plt.suptitle("Latent Features Legend", fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Save figures if path is provided
    if save_path:
        # Main visualization
        main_path = save_path
        plt.figure(1)
        plt.savefig(main_path, bbox_inches='tight', dpi=300)
        
        # Legend as a separate file
        legend_path = save_path.replace('.pdf', '_legend.pdf')
        plt.figure(2)
        plt.savefig(legend_path, bbox_inches='tight', dpi=300)
        
        print(f"Saved visualization to {main_path}")
        print(f"Saved legend to {legend_path}")
    
    # Show the figures
    plt.show()
    
    return plt.figure(1)

def create_compact_legend_figure(hex_colors, cluster_info, save_path=None):
    """Create a compact tabular legend as a separate figure"""
    n_latents = len(hex_colors)
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, n_latents * 0.25 + 0.5))
    ax.axis('off')
    
    # Define the number of columns
    n_cols = 3
    n_rows = (n_latents + n_cols - 1) // n_cols
    
    # Set up a grid layout
    table_data = []
    for row in range(n_rows):
        table_row = []
        for col in range(n_cols):
            idx = row * n_cols + col
            if idx < n_latents:
                title = cluster_info.get(idx, {}).get('title', f'Latent {idx}')
                table_row.append(f"{idx}: {title}")
            else:
                table_row.append("")
        table_data.append(table_row)
    
    # Create the table
    table = ax.table(
        cellText=table_data,
        cellLoc='left',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Style the table cells
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    
    # Scale the table to fit the figure
    table.scale(1, 1.2)
    
    # Modify table cells to include colored boxes
    for (i, j), cell in table.get_celld().items():
        idx = i * n_cols + j
        if idx < n_latents:
            # Set cell face color to light gray
            cell.set_facecolor('#f5f5f5')
            
            # Get cell dimensions
            x, y = cell.get_xy()
            width, height = cell.get_width(), cell.get_height()
            
            # Add colored rectangle at the beginning of cell
            color_rect = plt.Rectangle(
                (x + 0.05, y + 0.25 * height),
                0.05, 0.5 * height,
                facecolor=hex_colors[idx],
                edgecolor='black',
                linewidth=0.5,
                transform=ax.transData
            )
            ax.add_patch(color_rect)
            
            # Adjust text position to make room for colored box
            cell.set_x(x + 0.15)
    
    # Add figure title
    plt.suptitle("Latent Features Legend", fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved compact legend to {save_path}")
    
    return fig

# Generate a diverse and visually appealing color palette
def generate_distinct_colors(n):
    """Generate n visually distinct colors that are aesthetically pleasing"""
    if n <= 10:
        # For small number of colors, use a handpicked palette
        base_colors = [
            "#1f77b4",  # blue
            "#ff7f0e",  # orange
            "#2ca02c",  # green
            "#d62728",  # red
            "#9467bd",  # purple
            "#8c564b",  # brown
            "#e377c2",  # pink
            "#7f7f7f",  # gray
            "#bcbd22",  # olive
            "#17becf",  # teal
        ]
        # Return as many colors as needed from the palette, cycling if necessary
        return base_colors[:n] if n <= len(base_colors) else [base_colors[i % len(base_colors)] for i in range(n)]
    
    # For larger sets, use HSV color space for better separation
    colors = []
    
    # Use the golden ratio to ensure good separation in hue
    golden_ratio_conjugate = 0.618033988749895
    h = 0.1  # Start with a nice blue-ish hue
    
    for i in range(n):
        # Calculate hue using golden angle
        h = (h + golden_ratio_conjugate) % 1.0
        
        # Use high saturation and varied brightness for better distinction
        s = 0.7 + (i % 3) * 0.1  # Vary saturation slightly (0.7-0.9)
        v = 0.85 - (i % 4) * 0.05  # Vary brightness slightly (0.7-0.85)
        
        # Convert HSV to RGB
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        
        # Convert to hex
        hex_color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
        colors.append(hex_color)
    
    return colors

# Helper function to determine if a color is dark
def is_dark(hex_color):
    # Convert hex to RGB
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    # Calculate perceived brightness using common formula
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    return brightness < 0.5

def visualize_token_activations(model, tokenizer, sae, text, layer, save_path=None):
    """Visualize SAE activations for each token in a text"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cluster_info = get_latent_descriptions(model_id, args.layer, args.n_clusters)
    
    # Tokenize the input
    tokens = tokenizer.encode(text, return_tensors="pt").to(device)
    
    # Get token strings for better visualization
    token_strings = [tokenizer.decode([token_id]) for token_id in tokens[0]]
    
    # Find indices of <think> and </think> tags
    think_start_idx = None
    think_end_idx = None
    
    for i, token in enumerate(token_strings):
        # Look for <think> tag (might be tokenized differently)
        if "<think>" in token:
            think_start_idx = i
        # Look for </think> tag (might be tokenized differently)
        if "</think>" in token:
            think_end_idx = i
    
    # Get activations from the model
    with torch.no_grad():
        with model.trace(
            {
                "input_ids": tokens,
                "attention_mask": (tokens != tokenizer.pad_token_id).long()
            }
        ) as tracer:
            activations = model.model.layers[layer].output[0].save()
        
        # Move activations to CPU for processing
        activations = activations.cpu()

        # Process each token position
        token_latents = []
        top_latent_indices = []
        top_activation_strengths = []
        
        for i in range(tokens.shape[1]):
            # Get activation for this token position
            token_activation = activations[0, i, :]
            token_activation = token_activation - sae.b_dec

            # Get the top activating latent features
            all_activations = sae.encoder(token_activation.unsqueeze(0))
            all_activations = all_activations.squeeze(0)
            
            # Get top-1 activating latent
            top_value, top_index = torch.topk(all_activations, k=1)
            top_latent_indices.append(top_index.item())
            top_activation_strengths.append(top_value.item())
            
            # Get all latent activations for this token
            token_latents.append(all_activations.cpu().numpy())
    
    # Determine which tokens are in the think range for normalization
    in_think_range = []
    for i in range(len(token_strings)):
        should_color = True
        if think_start_idx is not None:
            # If we have a <think> tag but we're before it, don't color
            if i <= think_start_idx:
                should_color = False
            # If we have both <think> and </think> tags and we're after </think>, don't color
            elif think_end_idx is not None and i >= think_end_idx:
                should_color = False
        in_think_range.append(should_color)
    
    # Normalize activation strengths to [0, 1], but only considering tokens in the think range
    think_range_activations = [strength for i, strength in enumerate(top_activation_strengths) if in_think_range[i]]
    if think_range_activations:
        max_activation = max(think_range_activations)
    else:
        max_activation = max(top_activation_strengths) if top_activation_strengths else 1.0
    
    # Normalize all activations by the max activation in the think range
    normalized_activations = [act / max_activation for act in top_activation_strengths]
    
    # Create a color map for the latents with better diversity and aesthetics
    n_latents = len(token_latents[0])
    
    # Generate distinct colors
    hex_colors = generate_distinct_colors(n_latents)
    
    # Generate HTML output
    html_output = "<html><head><style>"
    html_output += "body { font-family: Arial, sans-serif; line-height: 1.5; background-color: #f5f5f5; padding: 20px; }"
    html_output += ".token-container { padding: 20px; background-color: white; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }"
    html_output += ".token { padding: 3px 5px; margin: 2px; display: inline-block; border-radius: 3px; position: relative; }"
    html_output += ".token-info { font-size: 10px; color: #666; }"
    html_output += ".token:hover .tooltip { display: block; }"
    html_output += ".tooltip { display: none; position: absolute; bottom: 100%; left: 50%; transform: translateX(-50%); "
    html_output += "background-color: rgba(0,0,0,0.8); color: white; padding: 5px 10px; border-radius: 4px; "
    html_output += "width: 300px; z-index: 100; text-align: left; font-size: 12px; }"
    html_output += ".tooltip::after { content: ''; position: absolute; top: 100%; left: 50%; margin-left: -5px; "
    html_output += "border-width: 5px; border-style: solid; border-color: rgba(0,0,0,0.8) transparent transparent transparent; }"
    html_output += ".legend { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 10px; margin-top: 20px; }"
    html_output += ".legend-item { padding: 10px; border-radius: 4px; display: flex; align-items: center; background-color: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }"
    html_output += ".legend-color { width: 20px; height: 20px; margin-right: 10px; border-radius: 3px; }"
    html_output += ".legend-text { flex: 1; }"
    html_output += ".activation-bar { height: 4px; margin: 2px 0; background: linear-gradient(to right, var(--start-color), var(--end-color)); }"
    html_output += ".activation-info { display: flex; align-items: center; margin: 2px 0; }"
    html_output += ".activation-label { flex: 1; }"
    html_output += ".activation-value { margin-left: 10px; font-weight: bold; }"
    html_output += "</style></head><body>"
    
    html_output += f"<h2>SAE Activations Visualization</h2>"
    html_output += f"<p>Model: {model_id}, Layer: {args.layer}, Clusters: {args.n_clusters}</p>"
    
    # Add info about think tag coloring
    if think_start_idx is not None:
        html_output += f"<p><em>Note: Color intensity reflects relative activation strength within the think region. Only tokens between &lt;think&gt; and "
        if think_end_idx is not None:
            html_output += f"&lt;/think&gt; are colored.</em></p>"
        else:
            html_output += f"the end are colored.</em></p>"
    else:
        html_output += "<p><em>Note: Color intensity reflects relative activation strength.</em></p>"
        html_output += "<p><strong>Warning: No &lt;think&gt; tag found in the text. All tokens are colored.</strong></p>"
    
    html_output += "<div class='token-container'>"
    
    # Add each token with its color based on top-1 latent activation
    for i, (token_str, latent_idx, strength) in enumerate(zip(token_strings, top_latent_indices, normalized_activations)):
        # Determine if this token should be colored (only if between <think> and </think> tags if present)
        should_color = in_think_range[i]
        
        if should_color:
            # Get color for top latent
            color = hex_colors[latent_idx]
            # Adjust color opacity based on activation strength
            rgb_color = mcolors.hex2color(color)
            # Use the color at full strength but adjust opacity
            rgba_color = (*rgb_color, min(1.0, strength + 0.2))  # Add 0.2 to base opacity for readability
            blended_color = mcolors.to_hex(rgba_color)
            text_color = "#fff" if is_dark(color) else "#000"
        else:
            # Use a light gray for tokens outside the <think> region
            blended_color = "#f0f0f0"
            text_color = "#888"
        
        # Handle special characters for display
        display_token = html.escape(token_str).replace(" ", "&nbsp;").replace("\n", "\\n")
        if not display_token:
            display_token = "·"  # Placeholder for invisible tokens
        
        # Get cluster info for tooltip
        if should_color:
            tooltip_content = ""
            if latent_idx in cluster_info:
                title = cluster_info[latent_idx].get('title', '')
                percentage = strength * 100
                tooltip_content += f"<div class='activation-info'>"
                tooltip_content += f"<div class='activation-label'>L{latent_idx}: {title}</div>"
                tooltip_content += f"<div class='activation-value'>{percentage:.1f}%</div>"
                tooltip_content += "</div>"
        else:
            if "<think>" in token_str:
                tooltip_content = "Think tag - Coloring starts after this token"
            elif "</think>" in token_str:
                tooltip_content = "End think tag - Coloring stops at this token"
            else:
                tooltip_content = "Token outside of think region"
        
        html_output += f"<span class='token' style='background-color: {blended_color}; color: {text_color};'>"
        html_output += f"{display_token}<br/>"
        
        # Only show latent info for colored tokens
        if should_color:
            html_output += f"<span class='token-info' style='color: {text_color};'>L{latent_idx}</span>"
        else:
            html_output += f"<span class='token-info' style='color: {text_color};'>-</span>"
            
        html_output += f"<div class='tooltip'>{tooltip_content}</div></span>"
    
    html_output += "</div>"
    
    # Add a legend with better layout
    html_output += "<h3>Latent Features Legend</h3>"
    html_output += "<div class='legend'>"
    for i, color in enumerate(hex_colors):
        # Add cluster title to legend if available
        legend_text = f"Latent {i}"
        if i in cluster_info:
            legend_text = f"Latent {i}: {cluster_info[i]['title']}"
            
        html_output += f"<div class='legend-item'>"
        html_output += f"<div class='legend-color' style='background-color: {color};'></div>"
        html_output += f"<div class='legend-text'>{legend_text}</div>"
        html_output += "</div>"
    html_output += "</div>"
    
    html_output += "</body></html>"
    
    # Save HTML output if path is provided
    if save_path:
        with open(save_path, 'w') as f:
            f.write(html_output)
        print(f"Saved token activations visualization to {save_path}")
    
    return html_output

def create_paper_visualization(model, tokenizer, sae, text, layer, save_path=None, figsize=(14, 8), dpi=300):
    """
    Create a publication-quality visualization of SAE reasoning traces
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        sae: The sparse autoencoder
        text: The input text
        layer: The layer to analyze
        save_path: Path to save the visualization
        figsize: Figure size for the plot
        dpi: DPI for the output figure
    
    Returns:
        The figure object
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get cluster information
    cluster_info = get_latent_descriptions(model_id, args.layer, args.n_clusters)
    
    # Tokenize the input
    tokens = tokenizer.encode(text, return_tensors="pt").to(device)
    
    # Get token strings for visualization
    token_strings = [tokenizer.decode([token_id]) for token_id in tokens[0]]
    
    # Find indices of <think> and </think> tags
    think_start_idx = None
    think_end_idx = None
    
    for i, token in enumerate(token_strings):
        if "<think>" in token:
            think_start_idx = i
        if "</think>" in token:
            think_end_idx = i
    
    # Get activations from the model
    with torch.no_grad():
        with model.trace(
            {
                "input_ids": tokens,
                "attention_mask": (tokens != tokenizer.pad_token_id).long()
            }
        ) as tracer:
            activations = model.model.layers[layer].output[0].save()
        
        # Move activations to CPU for processing
        activations = activations.cpu()

        # Process each token position
        token_latents = []
        top_latent_indices = []
        top_activation_strengths = []
        
        for i in range(tokens.shape[1]):
            # Get activation for this token position
            token_activation = activations[0, i, :]
            token_activation = token_activation - sae.b_dec

            # Get the activations for latent features
            all_activations = sae.encoder(token_activation.unsqueeze(0))
            all_activations = all_activations.squeeze(0)
            
            # Get top activating latent
            top_value, top_index = torch.topk(all_activations, k=1)
            top_latent_indices.append(top_index.item())
            top_activation_strengths.append(top_value.item())
            
            # Store all latent activations for this token
            token_latents.append(all_activations.cpu().numpy())
    
    # Determine which tokens are in the think range
    in_think_range = []
    for i in range(len(token_strings)):
        should_color = True
        if think_start_idx is not None:
            if i <= think_start_idx:
                should_color = False
            elif think_end_idx is not None and i >= think_end_idx:
                should_color = False
        in_think_range.append(should_color)
    
    # Normalize activation strengths for visualization
    think_range_activations = [strength for i, strength in enumerate(top_activation_strengths) if in_think_range[i]]
    if think_range_activations:
        max_activation = max(think_range_activations)
    else:
        max_activation = max(top_activation_strengths) if top_activation_strengths else 1.0
    
    normalized_activations = [act / max_activation for act in top_activation_strengths]
    
    # Generate colors for latents
    n_latents = len(token_latents[0])
    hex_colors = generate_distinct_colors(n_latents)
    
    # Create figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # Define layout
    if think_start_idx is not None and think_end_idx is not None:
        gs = GridSpec(3, 1, height_ratios=[0.15, 0.7, 0.15])
    else:
        gs = GridSpec(2, 1, height_ratios=[0.2, 0.8])
    
    # Create main axes for the visualization
    if think_start_idx is not None and think_end_idx is not None:
        ax_intro = plt.subplot(gs[0])
        ax_main = plt.subplot(gs[1])
        ax_outro = plt.subplot(gs[2])
        axes = [ax_intro, ax_main, ax_outro]
    else:
        ax_main = plt.subplot(gs[0])
        ax_legend = plt.subplot(gs[1])
        axes = [ax_main, ax_legend]
    
    # Function to truncate long strings
    def truncate_str(s, max_len=16):
        if len(s) > max_len:
            return s[:max_len-3] + "..."
        return s
    
    # Prepare token sections
    if think_start_idx is not None and think_end_idx is not None:
        intro_tokens = token_strings[:think_start_idx+1]
        intro_latents = top_latent_indices[:think_start_idx+1]
        intro_activations = normalized_activations[:think_start_idx+1]
        
        main_tokens = token_strings[think_start_idx+1:think_end_idx]
        main_latents = top_latent_indices[think_start_idx+1:think_end_idx]
        main_activations = normalized_activations[think_start_idx+1:think_end_idx]
        
        outro_tokens = token_strings[think_end_idx:]
        outro_latents = top_latent_indices[think_end_idx:]
        outro_activations = normalized_activations[think_end_idx:]
        
        # Draw token sections
        token_sections = [
            (ax_intro, intro_tokens, intro_latents, intro_activations, False),
            (ax_main, main_tokens, main_latents, main_activations, True),
            (ax_outro, outro_tokens, outro_latents, outro_activations, False)
        ]
    else:
        # No think tags, just render everything in the main section
        token_sections = [(ax_main, token_strings, top_latent_indices, normalized_activations, True)]
    
    # Draw token visualizations for each section
    for ax, tokens, latents, activations, should_color_section in token_sections:
        ax.set_xlim(0, len(tokens))
        ax.set_ylim(0, 1)
        
        # Hide axes
        ax.axis('off')
        
        # Add tokens and color blocks
        for i, (token, latent_idx, strength) in enumerate(zip(tokens, latents, activations)):
            should_color = should_color_section
            
            # Format token for display
            display_token = token.replace('\n', '↵').replace(' ', '␣')
            if not display_token:
                display_token = "·"  # For invisible tokens
            
            # Draw token
            text = ax.text(i + 0.5, 0.5, display_token, 
                          ha='center', va='center', fontsize=10,
                          bbox=dict(boxstyle='round,pad=0.3', 
                                    facecolor='white', 
                                    alpha=1.0, 
                                    edgecolor='gray', 
                                    linewidth=0.5))
            
            # Add token index as superscript
            ax.text(i + 0.5, 0.85, f"{i}", fontsize=6, ha='center', va='center', alpha=0.7)
            
            # Add colored rectangle for latent feature if in the <think> section
            if should_color:
                color = hex_colors[latent_idx]
                # Adjust alpha based on activation strength
                rect = Rectangle((i, 0.0), 1.0, 0.3, 
                                linewidth=0, 
                                facecolor=color,
                                alpha=min(0.2 + strength * 0.8, 1.0))  # Minimum visibility
                ax.add_patch(rect)
                
                # Add latent index below
                latent_text = ax.text(i + 0.5, 0.15, f"L{latent_idx}", 
                                    fontsize=7, ha='center', va='center', 
                                    color=color if is_dark(color) else 'k',
                                    weight='bold')
                
                # Add white outline for better visibility
                latent_text.set_path_effects([
                    path_effects.withStroke(linewidth=2, foreground='white')
                ])
                
                # Add activation strength indicator
                bar_width = 0.8 * strength
                strength_bar = Rectangle((i + 0.5 - bar_width/2, 0.05), 
                                        bar_width, 0.03, 
                                        linewidth=0, 
                                        facecolor=color,
                                        alpha=0.9)
                ax.add_patch(strength_bar)
    
    # Add section labels if think tags are present
    if think_start_idx is not None and think_end_idx is not None:
        ax_intro.text(0.02, 0.98, "Input", fontsize=12, ha='left', va='top', 
                     transform=ax_intro.transAxes, fontweight='bold')
        ax_main.text(0.02, 0.98, "Chain of Thought", fontsize=12, ha='left', va='top', 
                    transform=ax_main.transAxes, fontweight='bold')
        ax_outro.text(0.02, 0.98, "Output", fontsize=12, ha='left', va='top', 
                     transform=ax_outro.transAxes, fontweight='bold')
    
    # Create a legend with the most active latent features
    if think_start_idx is not None and think_end_idx is not None:
        ax_legend = fig.add_axes([0.1, 0.02, 0.8, 0.07])
    
    # Create latent feature legend
    top_latents_dict = {}
    if think_start_idx is not None and think_end_idx is not None:
        # Count frequency of latent features in the thinking section
        for idx in main_latents:
            if idx not in top_latents_dict:
                top_latents_dict[idx] = 1
            else:
                top_latents_dict[idx] += 1
    else:
        # Use all tokens
        for idx in top_latent_indices:
            if idx not in top_latents_dict:
                top_latents_dict[idx] = 1
            else:
                top_latents_dict[idx] += 1
    
    # Get top N most frequent latents
    top_n = 7  # Adjust based on space
    top_latents = sorted(top_latents_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Create legend elements
    legend_elements = []
    for latent_idx, count in top_latents:
        color = hex_colors[latent_idx]
        label = f"L{latent_idx}: "
        if latent_idx in cluster_info:
            label += truncate_str(cluster_info[latent_idx].get('title', ''))
        else:
            label += f"Latent {latent_idx}"
        legend_elements.append(Line2D([0], [0], color=color, lw=4, label=label))
    
    # Add legend
    ax_legend.axis('off')
    if legend_elements:
        ax_legend.legend(handles=legend_elements, loc='center', ncol=min(3, len(legend_elements)), 
                      frameon=True, fontsize=9, title="Most Active Latent Features")
    
    # Add title with model and layer info
    plt.suptitle(f"SAE Reasoning Trace - {model_id.upper()}, Layer {layer}", fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05)
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        print(f"Saved paper visualization to {save_path}")
    
    return fig

# %%
# Get model ID from args.model
model_id = args.model.split('/')[-1].lower()

# Create directories for outputs
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/html', exist_ok=True)

# %% Load the model and tokenizer
print(f"Loading model {args.model}...")
model, tokenizer = load_model(
    model_name=args.model,
    load_in_8bit=False
)

# %% Load the SAE
sae, checkpoint = load_sae(model_id, args.layer, args.n_clusters)

# %% Create the combined visualization with cosine similarity matrix and PCA side by side
print("Creating combined visualization...")
combined_path = f'results/figures/sae_combined_viz_{model_id}_layer{args.layer}_clusters{args.n_clusters}.pdf'
create_combined_visualization(sae, save_path=combined_path)

# %% Load the evaluation responses
eval_example_idx = 5

eval_responses_path = f'../generate-responses/results/vars/eval_responses_{model_id}.json'
if not os.path.exists(eval_responses_path):
    print(f"Warning: Evaluation responses not found at {eval_responses_path}")
    eval_text = "This is a sample text for visualization. The SAE will try to represent each token using latent features."
else:
    with open(eval_responses_path, 'r') as f:
        eval_responses = json.load(f)
    
    # Get the example text
    example = eval_responses[eval_example_idx]
    eval_text = example.get('full_response', '')

# Visualize token activations
print("Creating token activation visualization...")
html_path = f'results/html/sae_token_viz_{model_id}_layer{args.layer}_clusters{args.n_clusters}_example{eval_example_idx}.html'
html_output = visualize_token_activations(model, tokenizer, sae, eval_text, args.layer, save_path=html_path)

print("\nVisualization complete!")
print(f"Combined visualization saved to: {combined_path}")
print(f"Token activation visualization saved to: {html_path}")

# display the html file
from IPython.display import HTML
HTML(html_path)

# %%
# Creating paper visualization
print("Creating paper visualization...")
paper_viz_path = f'results/figures/sae_paper_viz_{model_id}_layer{args.layer}_clusters{args.n_clusters}_example{eval_example_idx}.pdf'
paper_figure = create_paper_visualization(model, tokenizer, sae, eval_text, args.layer, save_path=paper_viz_path)

# %%
