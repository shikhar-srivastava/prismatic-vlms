# REPRODUCTION LLAVA - ALIGN
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=12 torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py   --model.type "reproduction-llava-v15+7b" --stage "align"
# REPRODUCTION LLAVA - FINETUNE

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=12 torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py   --model.type "reproduction-llava-v15+7b" --stage "finetune"

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=12 torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py   --model.type "reproduction-llava-v15+7b"

# # Test ONE STAGE

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=12 torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py   --model.type "one-stage-align-only+7b"
