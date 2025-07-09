#!/usr/bin/env bash
# create_reproduction_llava_validation_sets.sh
#
# Utility script to create the validation sets used for the Vicuna LNS alignment + finetune experiments
# and new LLaMa 130m models with LNS and PRE norm support.
# This mirrors the command that was run interactively in the previous session.
#
# Validation set sizes:
#   - Align   : 200 samples  (≈13 batches @ 8)
#   - Finetune: 200 samples  (≈25 batches @ 8)
#
# NOTE: Run this ONCE before launching training so that the validation sets
#       are available at `validation_sets/MODEL_NAME/`.

# # Original LLaVa v1.5 7B reproduction
# echo "Creating validation sets for reproduction-llava-v15+7b..."
# python scripts/create_validation_sets.py \
#   --model reproduction-llava-v15+7b \
#   --dataset llava-v15 \
#   --align_val_size 200 \
#   --finetune_val_size 200 \
#   --output_dir validation_sets/reproduction-llava-v15+7b 

# LLaMa 130m with LNS norm
echo "Creating validation sets for stage0-llama+130m..."
python scripts/create_validation_sets.py \
  --model stage0-llama+130m \
  --dataset llava-v15 \
  --align_val_size 200 \
  --finetune_val_size 200 \
  --output_dir validation_sets/stage0-llama+130m

# LLaMa 130m with LNS norm
echo "Creating validation sets for stage0-llama+130m-lns..."
python scripts/create_validation_sets.py \
  --model stage0-llama+130m-lns \
  --dataset llava-v15 \
  --align_val_size 200 \
  --finetune_val_size 200 \
  --output_dir validation_sets/stage0-llama+130m-lns

# LLaMa 130m with PRE norm  
echo "Creating validation sets for stage0-llama+130m-pre..."
python scripts/create_validation_sets.py \
  --model stage0-llama+130m-pre \
  --dataset llava-v15 \
  --align_val_size 200 \
  --finetune_val_size 200 \
  --output_dir validation_sets/stage0-llama+130m-pre

echo "All validation sets created successfully!" 