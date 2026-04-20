# for LAYER in 4 8 12 16 20 24; do
#     for N_CLUSTERS in {4..20}; do
#         python visualize_sae.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --layer $LAYER --n_clusters $N_CLUSTERS
#     done
# done

for LAYER in 6 10 14 18 22 26; do
    for N_CLUSTERS in {4..20}; do
        python visualize_sae.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --layer $LAYER --n_clusters $N_CLUSTERS
    done
done

# for LAYER in 8 14 20 26 32 38; do
#     for N_CLUSTERS in {4..20}; do
#         python visualize_sae.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --layer $LAYER --n_clusters $N_CLUSTERS
#     done
# done