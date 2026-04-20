# parser.add_argument('--dataset', type=str, choices=['gsm8k', 'math500', "aime"], default='gsm8k',
#                     help='Dataset to evaluate on (gsm8k or math500)')
# parser.add_argument('--thinking_model', type=str, default='deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
#                     help='Model for thinking/perplexity')
# parser.add_argument('--base_model', type=str, default='meta-llama/Llama-3.1-8B',
#                     help='Model for base generation')
# parser.add_argument('--thinking_layer', type=int, default=6,
#                     help='Layer to extract from thinking model')
# parser.add_argument('--n_clusters', type=int, default=19,
#                     help='Number of clusters for SAE')
# parser.add_argument('--lookahead', action='store_true', default=False,
#                     help='Enable lookahead functionality in hybrid generation')
# parser.add_argument('--use_perplexity_selection', action='store_true', default=False,
#                     help='Use perplexity-based selection between steered and unsteered generation')
# parser.add_argument('--n_tasks', type=int, default=500,
#                     help='Number of tasks to evaluate')
# parser.add_argument('--max_new_tokens', type=int, default=1500,
#                     help='Maximum number of tokens to generate')
# parser.add_argument('--eval_start_idx', type=int, default=0,
#                     help='Starting index in the dataset')
# parser.add_argument('--cold_start_tokens', type=int, default=1,
#                     help='Number of initial tokens to use from thinking model')
# parser.add_argument('--temperature', type=float, default=0.3,
#                     help='Temperature for sampling')
# parser.add_argument('--repetition_penalty', type=float, default=1.0,
#                     help='Repetition penalty (1.0 means no penalty, >1.0 discourages repetition)')
# parser.add_argument('--repetition_window', type=int, default=0,
#                     help='Window size for repetition detection')
# parser.add_argument('--coefficient', type=float, default=1,
#                     help='Steering coefficient')
# parser.add_argument('--results_dir', type=str, default='results',
#                     help='Directory to save results')
# parser.add_argument('--example_idx', type=int, default=13,
#                     help='Index of example to run')

for dataset in gsm8k math500 aime24 aime25; do
    python interactive_hybrid_model.py --dataset $dataset --thinking_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --base_model meta-llama/Llama-3.1-8B --thinking_layer 6 --n_clusters 20
done