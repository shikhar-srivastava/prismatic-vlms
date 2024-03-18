PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=12 torchrun --standalone --nnodes 1 --nproc-per-node 4 scripts/pretrain.py   --model.type "reproduction-llava-v15+7b"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=12 torchrun --standalone --nnodes 1 --nproc-per-node 4 scripts/pretrain.py   --model.type "reproduction-llava-v15+7b"