#!/bin/bash

# ============================================
# Debug Training Script for Transformer Model
# ============================================
# This script runs a minimal training session to verify the pipeline.
# It uses very small model parameters and a limited dataset subset
# for fast iteration and debugging.

# Generate a timestamp for uniquely naming the save directory
timestamp=$(date +"%Y%m%d_%H%M%S")
save_path="debug_$timestamp"  # Directory to save outputs/checkpoints

# Execute training with a lightweight configuration
python3 train.py \
  --d_model 32 \                         # Low embedding size (small model)
  --n_heads 2 \                          # Fewer attention heads
  --num_layers 1 \                       # Minimal number of layers
  --dropout_rate 0.1 \                   # Dropout to help generalization
  --batch_size 4 \                       # Tiny batch size for fast iteration
  --num_epochs 2 \                       # Run only 2 epochs (quick debug)
  --learning_rate 1e-3 \                 # Standard small-model learning rate
  --max_len 20 \                         # Short sequence length
  --src_lang en \                        # Source language: English
  --tgt_lang fr \                        # Target language: French
  --sp_model_path data/spm_en_fr_8k.model \  # Path to SentencePiece tokenizer
  --save_path "$save_path" \            # Output folder for saving model
  --small_subset                        # Use a small dataset for debugging
