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

def _get_infinite_schedule_with_warmup_rsqrt_cooldown(current_step: int, *, num_warmup_steps: int, decay_steps: int, cooldown_steps:int):

    if current_step < num_warmup_steps:
        # Warmup steps
        return float(current_step) / float(max(1.0, num_warmup_steps))
    elif current_step >= num_warmup_steps and current_step <= (num_warmup_steps + decay_steps):
        # Decay steps
        return math.sqrt(float(num_warmup_steps)/float(max(1.0, current_step)))
    elif current_step > (num_warmup_steps + decay_steps) and current_step < (num_warmup_steps + decay_steps + cooldown_steps):
        # Cooldown steps: linearly decay LR to 0 for the last `cooldown_steps`. Max LR = LR after decay steps.
        return (math.sqrt(float(num_warmup_steps)/float(num_warmup_steps + decay_steps))) * (1.0 - (current_step - (num_warmup_steps + decay_steps)) / cooldown_steps)
    


def get_infinite_schedule_with_warmup_rsqrt_cooldown(
    optimizer,
    *,
    num_warmup_steps,
    decay_steps,
    cooldown_steps,
    last_epoch=-1,
):
    """
    Create an infinite schedule with a warmup period followed by a square root decay, and a cooldown period.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        decay_steps (`int`):
            The number of steps for the decay phase.
        cooldown_steps (`int`):
            The number of steps for the cooldown phase.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    if num_warmup_steps is None or decay_steps is None or cooldown_steps is None:
        raise ValueError("num_warmup_steps, decay_steps, and cooldown_steps must be provided")
    assert num_warmup_steps >= 0, "num_warmup_steps must be non-negative"
    assert decay_steps >= 0, "decay_steps must be non-negative"
    assert cooldown_steps >= 0, "cooldown_steps must be non-negative"

    lr_lambda = partial(
        _get_infinite_schedule_with_warmup_rsqrt_cooldown,
        num_warmup_steps=num_warmup_steps,
        decay_steps=decay_steps,
        cooldown_steps=cooldown_steps
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

