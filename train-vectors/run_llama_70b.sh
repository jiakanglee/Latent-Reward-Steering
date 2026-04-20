python optimize_steering_vectors.py \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --max_iters 50 \
    --n_training_examples 2048 \
    --n_eval_examples 512 \
    --optim_minibatch_size 1 \
    --layer 30 \
    --steering_vector_idx -1 \
    --lr "1e-2"

for cluster in {0..9}; do  
    echo "Processing cluster: $cluster"
    python optimize_steering_vectors.py \
        --model meta-llama/Llama-3.3-70B-Instruct \
        --max_iters 50 \
        --n_training_examples 2048 \
        --n_eval_examples 512 \
        --optim_minibatch_size 1 \
        --layer 30 \
        --steering_vector_idx $cluster \
        --lr "1e-2" \
        --use_activation_perplexity_selection
done

python evaluate_steering_vectors.py --model meta-llama/Llama-3.3-70B-Instruct --steering_strategy linear

