#!/bin/bash

DATESTAMP=$(date +"%Y%m%d_%H%M%S")

# Define configs for different model sizes
declare -A D_MODELS=( ["small"]=128 ["medium"]=256 ["large"]=512 )
declare -A N_HEADS=( ["small"]=4   ["medium"]=4   ["large"]=8 )
declare -A N_LAYERS=( ["small"]=2  ["medium"]=4   ["large"]=6 )

for size in small medium large
do
    d_model=${D_MODELS[$size]}
    n_heads=${N_HEADS[$size]}
    num_layers=${N_LAYERS[$size]}

    for vocab_size in 8k 16k 32k
    do
        SAVE_PATH="/en_fr_run1/en_fr_${size}_d${d_model}_v${vocab_size}_${DATESTAMP}"
        echo "Running $size model with vocab $vocab_size..."

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
            --tgt_lang fr \
            --sp_model_path data/spm_en_fr_${vocab_size}.model \
            --save_path $SAVE_PATH
    done
done
