#!/bin/bash

# Get current datetime in format YYYYMMDD_HHMMSS
timestamp=$(date +"%Y%m%d_%H%M%S")
save_path="debug_$timestamp"

# Run training with debug configuration
python3 train.py \
  --d_model 32 \
  --n_heads 2 \
  --num_layers 1 \
  --dropout_rate 0.1 \
  --batch_size 4 \
  --num_epochs 2   \
  --learning_rate 1e-3 \
  --max_len 20 \
  --src_lang en \
  --tgt_lang fr \
  --sp_model_path data/spm_en_fr_8k.model \
  --save_path "$save_path" \
  --small_subset
