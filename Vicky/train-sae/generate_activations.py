
import argparse
import torch
import gc
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import utils

def main():
    """
    Main function to generate and cache model activations for specified layers.
    """
    parser = argparse.ArgumentParser(description="Generate and cache model activations.")
    parser.add_argument("--model", type=str, required=True,
                        help="Model to use for generating activations (e.g., 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B').")
    parser.add_argument("--layers", type=int, nargs='+', required=True,
                        help="A list of one or more layer numbers to process.")
    parser.add_argument("--n_examples", type=int, default=500,
                        help="Number of examples to use for generating activations.")
    parser.add_argument("--load_in_8bit", action="store_true", default=False,
                        help="Load the model in 8-bit mode to save memory.")

    args = parser.parse_args()

    print(f"Generating activations for model: {args.model}")
    print(f"Processing layers: {args.layers}")
    print(f"Number of examples: {args.n_examples}")

    # Load the model and tokenizer
    try:
        model, tokenizer = utils.load_model(
            model_name=args.model,
            load_in_8bit=args.load_in_8bit
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Process saved responses to generate and cache activations
    try:
        utils.process_saved_responses(
            model_name=args.model,
            n_examples=args.n_examples,
            model=model,
            tokenizer=tokenizer,
            layer_or_layers=args.layers
        )
        print("Successfully generated and cached activations.")
    except Exception as e:
        print(f"Error processing saved responses: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up resources
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        print("Cleaned up resources.")

if __name__ == "__main__":
    main() 