# BASIC STAGE 0

#PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=12 torchrun --standalone --nnodes 1 --nproc-per-node 4 scripts/pretrain.py   --stage "finetune" --model.type "stage0-after-llava+7b"  --dataset.type "llava-v1" --pretrained_checkpoint "/localdisk/ssrivas9/prismatic-vlms/runs/reproduction-llava-v15+7b+stage-align+x7/checkpoints/latest-checkpoint.pt" --run_id "stage-0-after-llava-158k"

# LoRA

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=12 torchrun --standalone --nnodes 1 --nproc-per-node 4 scripts/pretrain.py   \
# --stage "finetune" --model.type "stage0-after-llava+7b"  --dataset.type "llava-v1" \
# --pretrained_checkpoint "/localdisk/ssrivas9/prismatic-vlms/runs/reproduction-llava-v15+7b+stage-align+x7/checkpoints/latest-checkpoint.pt" \
# --run_id "lora-stage-0-after-llava-158k" --mitigation "lora"

# PREFIX
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=12 torchrun --standalone --nnodes 1 --nproc-per-node 4 scripts/pretrain.py   \
# --stage "finetune" --model.type "stage0-after-llava+7b"  --dataset.type "llava-v1" \
# --pretrained_checkpoint "/scratch/ssrivas9/prismatic-vlms/runs/reproduction-llava-v15+7b+stage-align+x7/checkpoints/latest-checkpoint.pt" \
# --run_id "ptune-stage-0-after-llava-158k" --mitigation "ptune"

# OLF
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=12 torchrun --standalone --nnodes 1 --nproc-per-node 4 scripts/pretrain.py   \
# --stage "finetune" --model.type "stage0-after-llava+7b"  --dataset.type "llava-v1" \
# --pretrained_checkpoint "/localdisk/ssrivas9/prismatic-vlms/runs/reproduction-llava-v15+7b+stage-align+x7/checkpoints/latest-checkpoint.pt" \
# --run_id "olf-stage-0-after-llava-158k" --mitigation "olf"

# IA3 
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=12 torchrun --standalone --nnodes 1 --nproc-per-node 4 scripts/pretrain.py   \
--stage "finetune" --model.type "stage0-after-llava+7b"  --dataset.type "llava-v1" \
--pretrained_checkpoint "/localdisk/ssrivas9/prismatic-vlms/runs/reproduction-llava-v15+7b+stage-align+x7/checkpoints/latest-checkpoint.pt" \
--run_id "ia3-stage-0-after-llava-158k" --mitigation "ia3"


# PTUNE
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=12 torchrun --standalone --nnodes 1 --nproc-per-node 4 scripts/pretrain.py   \
--stage "finetune" --model.type "stage0-after-llava+7b"  --dataset.type "llava-v1" \
--pretrained_checkpoint "/scratch/ssrivas9/prismatic-vlms/runs/reproduction-llava-v15+7b+stage-align+x7/checkpoints/latest-checkpoint.pt" \
--run_id "ptune-stage-0-after-llava-158k" --mitigation "ptune"
# PROMPT
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=12 torchrun --standalone --nnodes 1 --nproc-per-node 4 scripts/pretrain.py   \
--stage "finetune" --model.type "stage0-after-llava+7b"  --dataset.type "llava-v1" \
--pretrained_checkpoint "/scratch/ssrivas9/prismatic-vlms/runs/reproduction-llava-v15+7b+stage-align+x7/checkpoints/latest-checkpoint.pt" \
--run_id "prompt-stage-0-after-llava-158k" --mitigation "prompt"

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=12 torchrun --standalone --nnodes 1 --nproc-per-node 4 scripts/pretrain.py   \
# --stage "finetune" --model.type "stage0-after-llava+7b"  --dataset.type "test" \
# --pretrained_checkpoint "/scratch/ssrivas9/prismatic-vlms/runs/reproduction-llava-v15+7b+stage-align+x7/checkpoints/latest-checkpoint.pt" \
# --run_id "prompt-stage-0-after-llava-158k" --mitigation "prompt"

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=12 torchrun --standalone --nnodes 1 --nproc-per-node 4 scripts/pretrain.py   \
# --stage "finetune" --model.type "stage0-after-llava+7b"  --dataset.type "test" \
# --pretrained_checkpoint "/scratch/ssrivas9/prismatic-vlms/runs/reproduction-llava-v15+7b+stage-align+x7/checkpoints/latest-checkpoint.pt" \
# --run_id "olf-stage-0-after-llava-158k" --mitigation "olf"

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=12 torchrun --standalone --nnodes 1 --nproc-per-node 4 scripts/pretrain.py   \
# --stage "finetune" --model.type "stage0-after-llava+7b"  --dataset.type "llava-v1" \
# --pretrained_checkpoint "/scratch/ssrivas9/prismatic-vlms/runs/reproduction-llava-v15+7b+stage-align+x7/checkpoints/latest-checkpoint.pt" \
# --run_id "ia3-stage-0-after-llava-158k" --mitigation "ia3"


# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=12 torchrun --standalone --nnodes 1 --nproc-per-node 4 scripts/pretrain.py   \
# --stage "finetune" --model.type "stage0-after-llava+7b"  --dataset.type "llava-v1" \
# --pretrained_checkpoint "/localdisk/ssrivas9/prismatic-vlms/runs/reproduction-llava-v15+7b+stage-align+x7/checkpoints/latest-checkpoint.pt" \
# --run_id "lora-stage-0-after-llava-158k" --mitigation "lora"

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=12 torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/pretrain.py   \
# --stage "finetune" --model.type "stage0-after-llava+7b"  --dataset.type "test" \
# --pretrained_checkpoint "/localdisk/ssrivas9/prismatic-vlms/runs/reproduction-llava-v15+7b+stage-align+x7/checkpoints/latest-checkpoint.pt" \
# --run_id "lora-stage-0-after-llava-158k" --mitigation "lora"