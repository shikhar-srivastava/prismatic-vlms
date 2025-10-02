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
# echo "Creating validation sets for stage0-llama+130m..."
# python scripts/create_validation_sets.py \
#   --model stage0-llama+130m \
#   --dataset llava-v15 \
#   --align_val_size 200 \
#   --finetune_val_size 200 \
#   --output_dir validation_sets/stage0-llama+130m

# # LLaMa 130m with LNS norm
# echo "Creating validation sets for stage0-llama+130m-lns..."
# python scripts/create_validation_sets.py \
#   --model stage0-llama+130m-lns \
#   --dataset llava-v15 \
#   --align_val_size 200 \
#   --finetune_val_size 200 \
#   --output_dir validation_sets/stage0-llama+130m-lns

# # LLaMa 130m with PRE norm  
# echo "Creating validation sets for stage0-llama+130m-pre..."
# python scripts/create_validation_sets.py \
#   --model stage0-llama+130m-pre \
#   --dataset llava-v15 \
#   --align_val_size 200 \
#   --finetune_val_size 200 \
#   --output_dir validation_sets/stage0-llama+130m-pre


# LLaMa 60m with PRE norm (using _llama checkpoint)
echo "Creating validation sets for stage0-llama+60m-pre (using 60m_res_pre_lr1e-3_llama)..."
python scripts/create_validation_sets.py \
  --model stage0-llama+60m \
  --dataset llava-v15 \
  --align_val_size 200 \
  --finetune_val_size 200 \
  --llm_checkpoint_path "/scratch/ssrivas9/large-activations/60m_res_pre_lr1e-3_llama/model_20001" \
  --output_dir validation_sets/stage0-llama+60m-pre-llama

# LLaMa 60m with LNS norm (using _llama checkpoint)
echo "Creating validation sets for stage0-llama+60m-lns (using 60m_res_pre_lr1e-3_llama)..."
python scripts/create_validation_sets.py \
  --model stage0-llama+60m \
  --dataset llava-v15 \
  --align_val_size 200 \
  --finetune_val_size 200 \
  --llm_checkpoint_path "/scratch/ssrivas9/large-activations/60m_res_pre_lr1e-3_llama/model_20001" \
  --output_dir validation_sets/stage0-llama+60m-lns-llama

# LLaMa 250m with different checkpoint variants
echo "Creating validation sets for stage0-llama+250m (using 250m_res_lns_lr1e-3_llama)..."
python scripts/create_validation_sets.py \
  --model stage0-llama+250m \
  --dataset llava-v15 \
  --align_val_size 200 \
  --finetune_val_size 200 \
  --llm_checkpoint_path "/scratch/ssrivas9/large-activations/250m_res_lns_lr1e-3_llama/model_20001" \
  --output_dir validation_sets/stage0-llama+250m-llama

# LLaMa 350m with different checkpoint variants
echo "Creating validation sets for stage0-llama+350m (using 350m_res_lns_lr1e-3_llama)..."
python scripts/create_validation_sets.py \
  --model stage0-llama+350m \
  --dataset llava-v15 \
  --align_val_size 200 \
  --finetune_val_size 200 \
  --llm_checkpoint_path "/scratch/ssrivas9/large-activations/350m_res_lns_lr1e-3_llama/model_20001" \
  --output_dir validation_sets/stage0-llama+350m-llama


echo "All validation sets created successfully!" 