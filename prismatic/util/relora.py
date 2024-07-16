# Taken/Adapted from ReLORA: https://github.com/Guitaricet/relora/blob/176f37633fe02019835387258ddabcf6d91e328d/peft_pretraining/training_utils.py
import os
import json
import math
from functools import partial

import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR

from prismatic.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

def get_cosine_schedule_with_multiple_warmups(
    optimizer,
    *,
    num_training_steps,
    first_warmup_steps,
    restart_warmup_steps,
    restart_every,
    min_lr_ratio=0.1,
    adjust_step=0,
    last_epoch=-1,
):
    if restart_every is None:
        raise ValueError("restart_every must be specified for cosine_restarts scheduler")

    if num_training_steps % restart_every != 0:
        raise ValueError(f"num_training_steps ({num_training_steps}) must be divisible by restart_every ({restart_every})")

    lr_lambda = partial(
        _get_cosine_schedule_with_multiple_warmups_lambda,
        num_training_steps=num_training_steps,
        first_warmup_steps=first_warmup_steps,
        restart_warmup_steps=restart_warmup_steps,
        restart_every=restart_every,
        min_lr_ratio=min_lr_ratio,
        adjust_step=adjust_step,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def _get_cosine_schedule_with_multiple_warmups_lambda(
    current_step,
    *,
    num_training_steps,
    first_warmup_steps,
    restart_warmup_steps,
    restart_every,
    min_lr_ratio,
    adjust_step,
):
    """
    Args:
        adjust_step: useful when continuing training from a warmed up checkpoint,
            it allows to sync the resets by reducing the number of steps
            after the first warmup and before the first reset.
            Thus, your ReLoRA resets can be synced with the optimizer resets.
    """
    assert 0 <= min_lr_ratio <= 1.0, "min_lr_ratio must be in [0,1]"
    assert restart_every > 0, "restart_every must be positive"
    assert adjust_step + first_warmup_steps <= num_training_steps, "warmup + adjust_step is more than full training steps"
    assert adjust_step + first_warmup_steps <= restart_every, "the first reset will happen before the warmup is done"

    if current_step < first_warmup_steps:
        return float(current_step) / float(max(1, first_warmup_steps))

    _current_step = current_step + adjust_step

    restart_step = _current_step % restart_every
    restart_number = _current_step // restart_every

    if restart_step < restart_warmup_steps and current_step >= restart_every:
        # get expected lr multipler at the end of the warmup
        end_of_warmup_progress = (
            float(restart_number * restart_every + restart_warmup_steps - first_warmup_steps) /
            float(max(1, num_training_steps - first_warmup_steps))
        )

        _cosine_decay = 0.5 * (1.0 + math.cos(math.pi * end_of_warmup_progress))
        warmup_lr_multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * _cosine_decay
    
        return float(restart_step) / float(max(1, restart_warmup_steps)) * warmup_lr_multiplier

    progress = float(_current_step - first_warmup_steps) / float(max(1, num_training_steps - first_warmup_steps))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

    return max(0.0, min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay)


@torch.no_grad()
def random_pruning_(tensor, prune_ratio):
    """
    Performs random pruning dimensionality reduction **inplace**.
    Only reduces the inner dimensionality, does not affect the shape of the tensor
    """
    random_pruning_mask = torch.rand_like(tensor) > prune_ratio
    tensor.mul_(random_pruning_mask)


@torch.no_grad()
def magnitude_pruning_(tensor, prune_ratio):
    """
    Performs magnitude pruning dimensionality reduction **inplace**.
    Only reduces the inner dimensionality, does not affect the shape of the tensor
    """
    tensor_magnitude = torch.abs(tensor)
    threshold = torch.quantile(tensor_magnitude.flatten().to(dtype=torch.float32), prune_ratio).to(dtype=tensor.dtype)

    mask = tensor_magnitude > threshold
    tensor.mul_(mask.to(dtype=tensor.dtype))


# def optimizer_reset(
#     optimizer,
#     *,
#     reset_params: list[torch.nn.Parameter],
#     optimizer_state_keys: list[str],
#     reset_optimizer_on_relora: bool,
#     optimizer_random_pruning: float,
#     optimizer_magnitude_pruning: float,
# ):
#     """
#         optimizer_state_keys: e.g., ["exp_avg", "exp_avg_sq"]
#     """
#     n_reset_types = (
#         int(bool(reset_optimizer_on_relora))
#         + int(bool(optimizer_random_pruning))
#         + int(bool(optimizer_magnitude_pruning))
#     )
#     if n_reset_types != 1:
#         overwatch.warning(f"Got {reset_optimizer_on_relora=}, {optimizer_random_pruning=}, "
#                        f"{optimizer_magnitude_pruning=}")
#         raise ValueError(f"Exactly one of reset_optimizer_on_relora, "
#                          f"optimizer_random_pruning, optimizer_magnitude_pruning must be True")

#     # pruning_fn has to be inplace to work with ZeroRedundancyOptimizer
#     if reset_optimizer_on_relora:
#         overwatch.info("Resetting optimizer states to zeros")
#         # looks like zeroing out breaks dictionary in the optimizer
#         # see full error below
#         pruning_fn = partial(random_pruning_, prune_ratio=0.999)
#     elif optimizer_random_pruning:
#         overwatch.info(f"Performing random pruning of optimizer states. "
#                     f"Pruning {optimizer_random_pruning} percent")
#         pruning_fn = partial(random_pruning_, prune_ratio=optimizer_random_pruning)
#     elif optimizer_magnitude_pruning:
#         overwatch.info(f"Performing magnitude pruning of optimizer states. "
#                     f"Pruning {optimizer_magnitude_pruning} percent")
#         pruning_fn = partial(magnitude_pruning_, prune_ratio=optimizer_magnitude_pruning)
#     else:
#         raise ValueError("Unknown pruning type")

#     # ############################################################
#     # A reminder on how optimizer state is structured for regular optimizers:
#     # optimizer.state is a dict[torch.nn.Parameter, dict[str, torch.Tensor]]
#     # optimizer.state[p] is a dict[str, torch.Tensor] where str is
#     # an optimizer state key e.g., "exp_avg", "exp_avg_sq"
#     # Note that none of these tensors has parameter names
#     # and parameter maps to a **dictionary** of opt. states, not a tensor
#     # 
#     # For ZeroRedundancyOptimizer, it works differently.
#     # ZeroRedundancyOptimizer.state always maps to empty dicts.
#     # Instead, it uses optimizer.optim.state for rank-local updates.
#     # 
#     # For some reason, zeroing out a tensor in ZeroRedundancyOptimizer.opt.state
#     # causes an error during state_dict collection.
#     # This is why we use 0.999 pruning ratio for reset_optimizer case.
#     # 
#     # Here's an error that happens:
#     # 
#     # Traceback (most recent call last):
#     # File ".../peft_pretraining/torchrun_main.py", line 866, in <module>
#     #     main(args)
#     # File ".../peft_pretraining/torchrun_main.py", line 715, in main
#     #     save_model(
#     # File ".../peft_pretraining/torchrun_main.py", line 289, in save_model
#     #     save_model_ddp(model, optimizer, scheduler, training_state_checkpoint, run_config, save_dir)
#     # File ".../peft_pretraining/torchrun_main.py", line 224, in save_model_ddp
#     #     optimizer.consolidate_state_dict()
#     # File ".../python3.10/site-packages/torch/distributed/optim/zero_redundancy_optimizer.py", line 565, in consolidate_state_dict
#     #     self.optim.state_dict(),
#     # File ".../python3.10/site-packages/torch/optim/optimizer.py", line 364, in state_dict
#     #     packed_state = {(param_mappings[id(k)] if isinstance(k, torch.Tensor) else k): v
#     # File ".../python3.10/site-packages/torch/optim/optimizer.py", line 364, in <dictcomp>
#     #     packed_state = {(param_mappings[id(k)] if isinstance(k, torch.Tensor) else k): v
#     # KeyError: 140580723685184
#     # 
#     # One one hand, the hypothesis is that making a zero tensor
#     # is implementing by changing the pointer in the memory to
#     # an existing zero-tensor. But on the other hand, we didn't
#     # have issues with that when using regular Adam, without ZeroRedundancyOptimizer wrapper.
#     # ############################################################
#     n_zeros = 0
#     n_total = 0

#     optimizer_state = optimizer.state
#     for p in reset_params:
#         param_state = optimizer_state[p]
#         if len(param_state) == 0: # no state for this param, happens for ZeRo optimizer
#             continue
#         for key in optimizer_state_keys:
#             pruning_fn(param_state[key])  # pruning fn has to be inplace to keep the same keys in the dict
#             n_total += param_state[key].numel()
#             n_zeros += torch.sum(param_state[key] == 0).item()

#     _zeroed = n_zeros / (1e-7 + n_total) * 100
#     overwatch.info(f"Percent of optimizer states zeroed: {_zeroed:.2f}")

# def optimizer_reset(
#     optimizer,
#     *,
#     reset_params: list[torch.nn.Parameter],
#     optimizer_state_keys: list[str],
#     reset_optimizer_on_relora: bool,
#     optimizer_random_pruning: float,
#     optimizer_magnitude_pruning: float,
# ):
#     """
#     optimizer_state_keys: e.g., ["exp_avg", "exp_avg_sq"]
#     """
#     n_reset_types = (
#         int(bool(reset_optimizer_on_relora))
#         + int(bool(optimizer_random_pruning))
#         + int(bool(optimizer_magnitude_pruning))
#     )
#     if n_reset_types != 1:
#         overwatch.warning(f"Got {reset_optimizer_on_relora=}, {optimizer_random_pruning=}, "
#                        f"{optimizer_magnitude_pruning=}")
#         raise ValueError(f"Exactly one of reset_optimizer_on_relora, "
#                          f"optimizer_random_pruning, optimizer_magnitude_pruning must be True")

#     # pruning_fn has to be inplace to work with ZeroRedundancyOptimizer
#     if reset_optimizer_on_relora:
#         overwatch.info("Resetting optimizer states to zeros")
#         # looks like zeroing out breaks dictionary in the optimizer
#         # see full error below
#         pruning_fn = partial(random_pruning_, prune_ratio=0.999)
#     elif optimizer_random_pruning:
#         overwatch.info(f"Performing random pruning of optimizer states. "
#                     f"Pruning {optimizer_random_pruning} percent")
#         pruning_fn = partial(random_pruning_, prune_ratio=optimizer_random_pruning)
#     elif optimizer_magnitude_pruning:
#         overwatch.info(f"Performing magnitude pruning of optimizer states. "
#                     f"Pruning {optimizer_magnitude_pruning} percent")
#         pruning_fn = partial(magnitude_pruning_, prune_ratio=optimizer_magnitude_pruning)
#     else:
#         raise ValueError("Unknown pruning type")

#     n_zeros = 0
#     n_total = 0

#     # Map parameters to their names
#     param_to_name = {param: name for name, param in optimizer.param_groups[0]['params']}

#     optimizer_state = optimizer.state
#     for p in reset_params:
#         param_name = param_to_name.get(p, None)
#         if param_name is None:
#             overwatch.warning(f"Parameter not found in optimizer state")
#             continue

#         if p not in optimizer_state:
#             overwatch.warning(f"Parameter {param_name} not found in optimizer state")
#             continue

#         param_state = optimizer_state[p]
#         if len(param_state) == 0:  # no state for this param, happens for ZeRo optimizer
#             overwatch.warning(f"No state for parameter {param_name}")
#             continue

#         for key in optimizer_state_keys:
#             if key not in param_state:
#                 overwatch.warning(f"State key {key} not found in param state for parameter {param_name}")
#                 continue

#             before_pruning_zeros = torch.sum(param_state[key] == 0).item()
#             pruning_fn(param_state[key])  # pruning fn has to be inplace to keep the same keys in the dict
#             after_pruning_zeros = torch.sum(param_state[key] == 0).item()

#             # Debug prints
#             overwatch.debug(f"Parameter: {param_name}, State key: {key}, Zeros before: {before_pruning_zeros}, Zeros after: {after_pruning_zeros}")

#             n_total += param_state[key].numel()
#             n_zeros += after_pruning_zeros

#     _zeroed = n_zeros / (1e-7 + n_total) * 100
#     overwatch.info(f"Percent of optimizer states zeroed: {_zeroed:.2f}")

def optimizer_reset(optimizer):
    for param in optimizer.state:
        optimizer.state[param] = {k: torch.zeros_like(v) for k, v in optimizer.state[param].items()}