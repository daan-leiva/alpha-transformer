#!/bin/bash

# === Generate a timestamp for unique checkpoint directories ===
DATESTAMP=$(date +"%Y%m%d_%H%M%S")

# === Define model hyperparameters for each model size ===
# These are declared as associative arrays mapping size to specific configs
declare -A D_MODELS=( ["small"]=128 ["medium"]=256 ["large"]=512 )   # Embedding dimension
declare -A N_HEADS=(  ["small"]=4   ["medium"]=4   ["large"]=8   )   # Number of attention heads
declare -A N_LAYERS=( ["small"]=2   ["medium"]=4   ["large"]=6   )   # Number of encoder/decoder layers

# === Loop over each model size configuration ===
for size in small medium large
do
    d_model=${D_MODELS[$size]}      # Embedding size
    n_heads=${N_HEADS[$size]}       # Multi-head attention heads
    num_layers=${N_LAYERS[$size]}   # Transformer depth

    # === Loop over each SentencePiece vocabulary size ===
    for vocab_size in 8k 16k 32k
    do
        # Define where to save model artifacts and logs
        SAVE_PATH="/en_de_run1/en_de_${size}_d${d_model}_v${vocab_size}_${DATESTAMP}"
        echo "Running $size model with vocab $vocab_size..."

        # === Run training with defined hyperparameters ===
        python3 train.py \
            --d_model $d_model \
            --n_heads $n_heads \
            --num_layers $num_layers \
            --dropout_rate 0.1 \
            --batch_size 32 \
            --num_epochs 20 \
            --learning_rate 1e-4 \
            --max_len 100 \
            --src_lang en \
            --tgt_lang de \
            --sp_model_path data/spm_en_de_${vocab_size}.model \
            --save_path $SAVE_PATH \
            --label_smoothing
    done
done
