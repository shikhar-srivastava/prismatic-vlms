import os

# Define variables
pretrained_checkpoint = "/localdisk/ssrivas9/prismatic-vlms/runs/reproduction-llava-v15+7b+stage-align+x7/checkpoints/latest-checkpoint.pt"
dataset_type = "llava-v1"  # Change this variable based on your task
stage = "finetune"
model_type = "stage0-after-llava+7b"
run_id_prefix = "stage-0-after-llava"
mitigation_methods = ["lora", "olf", "ia3", "soft", "sgm", "qlora"]
{
  "soft": [
    "soft-stage-0-after-llava-vqav2-soft-alpha-0.1",
  ],
  "lora": [
    "lora-stage-0-after-llava-vqav2-peft-policy",
  ],
  "qlora": [
    "qlora-stage-0-after-llava-vqav2-8bit",
  ],
  "msgm": [
    "msgm-stage-0-after-llava-vqav2-8bit",
  ],
  "sgm": [
    "sgm-stage-0-after-llava-vqav2-peft-policy",
  ],
  "ia3": [
    "ia3-stage-0-after-llava-vqav2",
  ],
  "naive-ft": [
    "stage-0-after-llava-vqav2",
  ],
}

# Define mitigation specific arguments
mitigation_args = {
    "lora": "",
    "olf": "",
    "ia3": "",
    "soft": "--soft_alpha 0.01",
    "sgm": "--soft_alpha 0.01 --mitigation lora",
    "qlora": "--load_8bit True"
}

# Define the base command template
base_command = (
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=12 torchrun --standalone --nnodes 1 --nproc-per-node 4 scripts/pretrain.py "
    "--stage \"{stage}\" --model.type \"{model_type}\" --dataset.type \"{dataset_type}\" "
    "--pretrained_checkpoint \"{pretrained_checkpoint}\" --run_id \"{run_id}\" {additional_args}"
)

# Generate and print commands for each mitigation method
for mitigation in mitigation_methods:
    run_id = f"{run_id_prefix}-{dataset_type}-{mitigation}"
    additional_args = f"--mitigation \"{mitigation}\" {mitigation_args[mitigation]}"
    command = base_command.format(
        stage=stage,
        model_type=model_type,
        dataset_type=dataset_type,
        pretrained_checkpoint=pretrained_checkpoint,
        run_id=run_id,
        additional_args=additional_args
    )
    print(command)
