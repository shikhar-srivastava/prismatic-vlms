# BASIC STAGE 0

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=12 torchrun --standalone --nnodes 1 --nproc-per-node 4 scripts/pretrain.py   --stage "finetune" --model.type "stage0-after-llava+7b"  --dataset.type "llava-v1" --pretrained_checkpoint "/localdisk/ssrivas9/prismatic-vlms/runs/reproduction-llava-v15+7b+stage-align+x7/checkpoints/latest-checkpoint.pt" --run_id "stage-0-after-llava-158k"
