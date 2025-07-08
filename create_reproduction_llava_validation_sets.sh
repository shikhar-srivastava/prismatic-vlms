#!/usr/bin/env bash
# create_reproduction_llava_validation_sets.sh
#
# Utility script to create the validation sets used for the Vicuna LNS alignment + finetune experiments.
# This mirrors the command that was run interactively in the previous session.
#
# Validation set sizes:
#   - Align   : 100 samples  (≈13 batches @ 8)
#   - Finetune: 200 samples  (≈25 batches @ 8)
#
# NOTE: Run this ONCE before launching training so that the validation sets
#       are available at `validation_sets/reproduction-llava-v15+7b/`.

python scripts/create_validation_sets.py \
  --model reproduction-llava-v15+7b \
  --dataset llava-v15 \
  --align_val_size 100 \
  --finetune_val_size 200 \
  --output_dir validation_sets/reproduction-llava-v15+7b 