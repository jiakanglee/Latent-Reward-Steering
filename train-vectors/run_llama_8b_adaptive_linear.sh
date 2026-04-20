for cluster in {0..14}; do  
    echo "Processing cluster: $cluster"
    python optimize_steering_vectors.py \
        --model meta-llama/Llama-3.1-8B \
        --max_iters 50 \
        --n_training_examples 2048 \
        --n_eval_examples 512 \
        --optim_minibatch_size 4 \
        --layer 12 \
        --steering_vector_idx $cluster \
        --lr "1e-2" \
        --use_activation_perplexity_selection \
        --steering_type adaptive_linear \
        --adaptive_hidden 128
done

python evaluate_steering_vectors.py --model meta-llama/Llama-3.1-8B --steering_strategy adaptive_linear

