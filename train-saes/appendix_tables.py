#!/usr/bin/env python3

import json

def load_json_file(filepath):
    """Load JSON file and return the data."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_examples(examples, max_examples=10):
    """Format examples for LaTeX table, limiting to max_examples."""
    formatted_examples = []
    for example in examples[:max_examples]:
        # Escape special LaTeX characters and wrap in quotes
        escaped = example.replace('\\', '\\textbackslash{}').replace('&', '\\&').replace('%', '\\%').replace('$', '\\$').replace('#', '\\#').replace('^', '\\textasciicircum{}').replace('_', '\\_').replace('{', '\\{').replace('}', '\\}').replace('~', '\\textasciitilde{}').replace('λ', '\\lambda')
        formatted_examples.append(f'``{escaped}\'\'')
    
    return ', '.join(formatted_examples)

def generate_latex_table(model_name, model_label, layer, cluster_size, json_filepath):
    """Generate LaTeX table for a specific model configuration."""
    data = load_json_file(json_filepath)
    
    # Check if this is the correct layer
    if data.get('layer') != layer:
        raise ValueError(f"Expected layer {layer}, but JSON contains layer {data.get('layer')}")
    
    cluster_data = data['results_by_cluster_size'].get(str(cluster_size))
    if not cluster_data:
        raise ValueError(f"Cluster size {cluster_size} not found for layer {layer} in {json_filepath}")
    
    # Get the first result (as requested)
    first_result = cluster_data['all_results'][0]
    
    # Get examples directly from the cluster data
    examples_dict = cluster_data.get('examples', {})
    categories = first_result.get('categories', [])
    
    # Generate LaTeX
    latex_lines = []
    
    latex_lines.append(f"\\subsection{{{model_name} (Layer {layer}, Dict Size {cluster_size})}}")
    latex_lines.append(f"\\label{{subsec:{model_label}_features}}")
    latex_lines.append("")
    latex_lines.append("\\small")
    latex_lines.append("\\begin{longtable}{p{0.3\\linewidth}|p{0.65\\linewidth}}")
    latex_lines.append(f"\\caption{{Categories and representative examples for {model_name} (Layer {layer}, Dict Size {cluster_size})}}")
    latex_lines.append(f"\\label{{tab:{model_label}}} \\\\")
    latex_lines.append("\\toprule")
    latex_lines.append("\\textbf{Category} & \\textbf{Representative Example} \\\\")
    latex_lines.append("\\midrule")
    latex_lines.append("\\endhead")
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\endfoot")
    
    # Process categories in order
    for i, category in enumerate(categories):
        category_id = category[0]
        title = category[1]
        
        # Get examples for this category - pick the longest one
        category_examples = examples_dict.get(category_id, [])
        if category_examples:
            # Find the longest example
            example = max(category_examples, key=len)
            # Escape special LaTeX characters and Unicode characters
            escaped = example.replace('\\', '\\textbackslash{}').replace('&', '\\&').replace('%', '\\%').replace('$', '\\$').replace('#', '\\#').replace('^', '\\textasciicircum{}').replace('_', '\\_').replace('{', '\\{').replace('}', '\\}').replace('~', '\\textasciitilde{}')
            # Handle common Unicode characters
            escaped = escaped.replace('Δ', '$\\Delta$').replace('π', '$\\pi$').replace('×', '$\\times$').replace('β', '$\\beta$').replace('≈', '$\\approx$')
            # Handle subscripts
            escaped = escaped.replace('₀', '$_0$').replace('₁', '$_1$').replace('₂', '$_2$').replace('₃', '$_3$').replace('₄', '$_4$').replace('₅', '$_5$').replace('₆', '$_6$').replace('₇', '$_7$').replace('₈', '$_8$').replace('₉', '$_9$')
            # Handle superscripts
            escaped = escaped.replace('²', '$^2$').replace('³', '$^3$')
            formatted_example = f'``{escaped}\'\''
        else:
            formatted_example = "``No examples available''"
        
        latex_lines.append(f"{title} & ")
        latex_lines.append(f"{formatted_example} \\\\")
        
        # Add midrule between rows (except for the last row)
        if i < len(categories) - 1:
            latex_lines.append("\\midrule")
    
    latex_lines.append("\\end{longtable}")
    latex_lines.append("")
    
    return '\n'.join(latex_lines)

def main():
    """Generate the complete LaTeX appendix."""
    # Configuration for each model
    configs = [
        {
            'model_name': 'DeepSeek-R1-Distill-Llama-8b',
            'model_label': 'llama_8b',
            'layer': 6,
            'cluster_size': 15,
            'json_file': 'results/vars/sae_topk_results_deepseek-r1-distill-llama-8b_layer6.json'
        },
        {
            'model_name': 'DeepSeek-R1-Distill-Qwen-1.5b',
            'model_label': 'qwen_1.5b',
            'layer': 4,
            'cluster_size': 25,
            'json_file': 'results/vars/sae_topk_results_deepseek-r1-distill-qwen-1.5b_layer4.json'
        },
        {
            'model_name': 'DeepSeek-R1-Distill-Qwen-14b',
            'model_label': 'qwen_14b',
            'layer': 38,
            'cluster_size': 5,
            'json_file': 'results/vars/sae_topk_results_deepseek-r1-distill-qwen-14b_layer38.json'
        },
        {
            'model_name': 'QwQ-32B',
            'model_label': 'qwq_32b',
            'layer': 27,
            'cluster_size': 10,
            'json_file': 'results/vars/sae_topk_results_qwq-32b_layer27.json'
        },
        {
            'model_name': 'DeepSeek-R1-Distill-Qwen-32b',
            'model_label': 'qwen_32b',
            'layer': 27,
            'cluster_size': 15,
            'json_file': 'results/vars/sae_topk_results_deepseek-r1-distill-qwen-32b_layer27.json'
        }
    ]
    
    # Print header
    print("\\section{Sparse Autoencoder Features}")
    print("\\label{app:sae_appendix}")
    print("")
    print("In this section, we provide detailed tables showing the complete reasoning taxonomies for our best-performing SAE configurations. For transparency and to demonstrate the full scope of our approach, we present all discovered categories rather than a curated subset. For each model, we list all category titles and representative examples for the sparse autoencoder features identified during our analysis.")
    print("")
    
    # Generate tables for each configuration
    for config in configs:
        try:
            latex_table = generate_latex_table(
                config['model_name'],
                config['model_label'],
                config['layer'], 
                config['cluster_size'], 
                config['json_file']
            )
            print(latex_table)
        except Exception as e:
            print(f"% Error generating table for {config['model_name']}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()