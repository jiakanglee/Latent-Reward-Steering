import torch
import random
import gc
from tqdm import tqdm
from utils.utils import get_random_distinct_colors

def custom_hybrid_generate(
        thinking_model, 
        base_model,
        base_tokenizer,
        input_ids, 
        max_new_tokens, 
        baseline_method="probe",  # Options: "probe", "random", "norm_diff", "kl_div"
        baseline_config=None,  # Configuration for the baseline method
        warmup=0,
        show_progress=True,
        color_output=False):
    """
    Unified hybrid generate function that supports different baseline methods.
    
    Args:
        thinking_model: The thinking model to use
        base_model: The base model to use
        base_tokenizer: The tokenizer
        input_ids: Input token ids (can be batched)
        max_new_tokens: Maximum number of tokens to generate
        baseline_method: The baseline method to use ("probe", "random", "norm_diff", "kl_div")
        baseline_config: Configuration for the baseline method
        warmup: Number of warmup tokens
        show_progress: Whether to show progress bar
        color_output: Whether to color the output
    """
    # Get the device of the thinking model
    device = thinking_model.device
    
    # Handle batched input
    batch_size = input_ids.shape[0]
    base_generated_ids = input_ids.clone().cpu()
    
    # Get random distinct colors for labels
    if baseline_method == "probe" and baseline_config is not None:
        all_labels = list(baseline_config.get("label_to_idx", {}).keys())
        all_labels.append("warmup")  # Add warmup to the list of labels
        label_colors = get_random_distinct_colors(all_labels)
    else:
        # For other methods, just create colors for the methods themselves
        all_labels = ["warmup", "random", "norm_diff", "kl_div"]
        label_colors = get_random_distinct_colors(all_labels)
    
    # Color for warmup
    if "warmup" not in label_colors:
        label_colors["warmup"] = "\033[90m"  # Default to gray if not in random assignment
    
    iterator = range(max_new_tokens)
    if show_progress:
        iterator = tqdm(iterator, desc="Generating response")
    
    # Track model usage and forced tokens for each sequence in batch
    base_model_tokens = [0] * batch_size
    thinking_model_tokens = [0] * batch_size
    forced_tokens = [{} for _ in range(batch_size)]  # List of dictionaries to track token frequencies
    
    # Three possible states: 0 = not forced, 1 = attempted force (models agreed), 2 = successful force (models disagreed)
    forced_states = [[] for _ in range(batch_size)]  
    forced_labels = [[] for _ in range(batch_size)]  # List of lists to track which label forced each token
    potential_forced_labels = [[] for _ in range(batch_size)]  # List of lists to track which label would have forced each token
    seen_end_think = [False] * batch_size  # Track if we've seen </think> token for each sequence

    for k in iterator:
        base_input_chunk = base_generated_ids.to(device)

        with torch.no_grad():
            with thinking_model.trace({
                        "input_ids": base_input_chunk, 
                        "attention_mask": (base_input_chunk != base_tokenizer.pad_token_id).long()
            }) as tracer:
                thinking_outputs = thinking_model.lm_head.output.save()
                
                # Get hidden states if using probe or norm_diff baseline
                if baseline_method == "probe":
                    probe_layer = baseline_config["probe_layer"]
                    hidden_states = thinking_model.model.layers[probe_layer].output[0][:, -1, :].save()
                elif baseline_method == "norm_diff":
                    target_layer = baseline_config["target_layer"]
                    thinking_hidden_states = thinking_model.model.layers[target_layer].output[0][:, -1, :].save()

        # Now run base model
        with torch.no_grad():
            with base_model.trace({
                        "input_ids": base_input_chunk, 
                        "attention_mask": (base_input_chunk != base_tokenizer.pad_token_id).long()
            }) as tracer:
                base_outputs = base_model.lm_head.output.save()
                
                # Get base model hidden states if using norm_diff
                if baseline_method == "norm_diff":
                    target_layer = baseline_config["target_layer"]
                    base_hidden_states = base_model.model.layers[target_layer].output[0][:, -1, :].save()

        # Get baseline predictions for each sequence in batch
        if baseline_method == "probe":
            # Get probe predictions
            hidden_states = hidden_states.to(torch.float32).to(device)
            probe = baseline_config["probe"].to(device)
            label_to_idx = baseline_config["label_to_idx"]
            force_categories = baseline_config.get("forcing", None)
            logits = probe(hidden_states)
            probs = torch.sigmoid(logits)
            max_probs, max_indices = torch.max(probs, dim=-1)
            max_labels = [label for idx in max_indices for label, label_idx in label_to_idx.items() if label_idx == idx]
            should_force = max_probs > baseline_config.get("threshold", 0.5)
            should_force = [0 if label not in force_categories else should_force[i] for i, label in enumerate(max_labels)]
            forced_labels_batch = [label if force else None for label, force in zip(max_labels, should_force)]
        elif baseline_method == "random":
            # Get random baseline predictions
            forced_token_rate = baseline_config.get("forced_token_rate", 0.5)
            should_force = [random.random() < forced_token_rate for _ in range(batch_size)]
            forced_labels_batch = ["random" if force else None for force in should_force]
        elif baseline_method == "norm_diff":
            # Calculate norm difference
            threshold = baseline_config.get("threshold", 0.1)
            
            # Calculate norms for both models' hidden states
            base_norms = torch.norm(base_hidden_states, dim=-1)
            thinking_norms = torch.norm(thinking_hidden_states, dim=-1)
            
            # Calculate relative difference (abs(base_norm - thinking_norm) / base_norm)
            norm_diff = torch.abs(base_norms - thinking_norms) / base_norms
            should_force = norm_diff > threshold
            forced_labels_batch = ["norm_diff" if force else None for force in should_force]
        elif baseline_method == "kl_div":
            # Calculate KL divergence
            threshold = baseline_config.get("threshold", 1.0)
            
            # Get softmax distributions
            base_probs = torch.nn.functional.softmax(base_outputs[:, -1, :], dim=-1)
            thinking_probs = torch.nn.functional.softmax(thinking_outputs[:, -1, :], dim=-1)
            
            # Calculate KL divergence: KL(thinking || base)
            kl_divs = torch.nn.functional.kl_div(
                thinking_probs.log(), 
                base_probs, 
                reduction='none'
            ).sum(dim=-1)
            
            should_force = kl_divs > threshold
            forced_labels_batch = ["kl_div" if force else None for force in should_force]
        
        # Get next tokens from both models for each sequence in batch
        base_next_tokens = base_outputs[:, -1, :].argmax(dim=-1)
        thinking_next_tokens = thinking_outputs[:, -1, :].argmax(dim=-1)

        # Process each sequence in the batch
        next_tokens = []
        for i in range(batch_size):
            # During warmup, use thinking model's predictions
            if k < warmup:
                next_token = thinking_next_tokens[i]
                thinking_model_tokens[i] += 1
                forced_states[i].append(2)  # Successfully forced (warmup)
                forced_labels[i].append("warmup")
                potential_forced_labels[i].append("warmup")
            else:
                # Check if we've seen </think> token
                current_text = base_tokenizer.decode(base_generated_ids[i], skip_special_tokens=True)
                if "</think>" in current_text:
                    seen_end_think[i] = True
                
                # Track potential forced label regardless of whether we force it
                if not seen_end_think[i] and forced_labels_batch[i] is not None:
                    potential_forced_labels[i].append(forced_labels_batch[i])
                else:
                    potential_forced_labels[i].append(None)
                
                # Determine forced state and next token
                if not seen_end_think[i] and forced_labels_batch[i] is not None:
                    if thinking_next_tokens[i] != base_next_tokens[i]:
                        # Criterion met and models disagree - successful force
                        next_token = thinking_next_tokens[i]
                        thinking_model_tokens[i] += 1
                        forced_states[i].append(2)  # 2 = successful force
                        forced_labels[i].append(forced_labels_batch[i])
                        # Track forced token
                        token_text = base_tokenizer.decode(next_token)
                        forced_tokens[i][token_text] = forced_tokens[i].get(token_text, 0) + 1
                    else:
                        # Criterion met but models agree - attempted force
                        next_token = base_next_tokens[i]
                        base_model_tokens[i] += 1
                        forced_states[i].append(1)  # 1 = attempted force
                        forced_labels[i].append(forced_labels_batch[i])
                else:
                    # No forcing criterion - use base model
                    next_token = base_next_tokens[i]
                    base_model_tokens[i] += 1
                    forced_states[i].append(0)  # 0 = not forced
                    forced_labels[i].append(None)
            
            next_tokens.append(next_token)

        # Stack next tokens and append to sequences
        next_tokens = torch.stack(next_tokens)
        base_generated_ids = torch.cat([base_generated_ids, next_tokens.unsqueeze(1).cpu()], dim=1)
        
        # Check for end of sequence for each sequence in batch
        if all(next_tokens == base_tokenizer.eos_token_id):
            break

        del trace, thinking_outputs, base_outputs, base_next_tokens, thinking_next_tokens, base_input_chunk
        if baseline_method == "norm_diff":
            del base_hidden_states, thinking_hidden_states
        elif baseline_method == "probe":
            del hidden_states
       
        torch.cuda.empty_cache()
        if k % 50 == 0:
            gc.collect()
    
    gc.collect()
    
    if color_output:
        # Print model usage statistics
        total_tokens = [base + think for base, think in zip(base_model_tokens, thinking_model_tokens)]
        print(f"\nModel Usage Statistics:")
        for i in range(batch_size):
            print(f"\nSequence {i+1}:")
            
            # Calculate forcing statistics as percentage of total tokens
            total_attempted = sum(1 for state in forced_states[i] if state == 1 or state == 2)
            successful_forced = sum(1 for state in forced_states[i] if state == 2)
            
            total_rate = total_attempted / total_tokens[i]
            successful_rate = successful_forced / total_tokens[i]
            
            print(f"Total attempted: {total_attempted} ({total_rate*100:.1f}%)")
            print(f"Successful forced: {successful_forced} ({successful_rate*100:.1f}%)")
        
        # Print top forced tokens for each sequence
        print("\nTop 10 Most Frequent Forced Tokens:")
        for i in range(batch_size):
            print(f"\nSequence {i+1}:")
            sorted_tokens = sorted(forced_tokens[i].items(), key=lambda x: x[1], reverse=True)
            for token, freq in sorted_tokens[:10]:
                print(f"'{token}': {freq} times")
        
        # Print colored output for each sequence
        print("\nColored Output (Colors indicate which label forced the token):")
        for i in range(batch_size):
            print(f"\nSequence {i+1}:")
            base_text = base_tokenizer.decode(base_generated_ids[i], skip_special_tokens=True)
            
            # Split into tokens and color them, skipping input tokens
            base_tokens = base_tokenizer.encode(base_text)
            input_length = len(base_tokenizer.encode(base_tokenizer.decode(input_ids[i], skip_special_tokens=True)))
            colored_base = []
            
            for j, token in enumerate(base_tokens):
                if j < input_length:
                    colored_base.append(base_tokenizer.decode(token))
                else:
                    token_idx = j - input_length
                    token_text = base_tokenizer.decode(token)
                    if token_idx < len(forced_states[i]):
                        state = forced_states[i][token_idx]
                        if state == 2:  # Successfully forced
                            # Token was actually forced - color and underline
                            label = forced_labels[i][token_idx]
                            if label == "warmup":
                                colored_base.append(f"\033[90m\033[4m{token_text}\033[0m")  # Gray for warmup with underline
                            else:
                                colored_base.append(f"{label_colors[label]}\033[4m{token_text}\033[0m")  # Colored with underline
                        elif state == 1:  # Attempted force
                            # Token wasn't forced but would have been - just color
                            label = forced_labels[i][token_idx]
                            colored_base.append(f"{label_colors[label]}{token_text}\033[0m")
                        else:  # Not forced
                            colored_base.append(token_text)
                    else:
                        colored_base.append(token_text)
            
            print("Base (with forced tokens colored and underlined by label):")
            print(base_tokenizer.convert_tokens_to_string(colored_base))
        
        # Print color legend
        print("\nColor Legend:")
        for label, color in label_colors.items():
            print(f"{color}{label}\033[0m")
        print("\nUnderlined tokens were successfully forced (models disagreed)")
        print("Colored but not underlined tokens were attempted forces (models agreed)")
    
    return base_generated_ids.cpu(), forced_states, forced_labels, forced_tokens
