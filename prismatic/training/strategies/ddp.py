"""
ddp.py

Core class definition for a strategy implementing Torch native Distributed Data Parallel Training; note that on most
GPU hardware and LLM backbones >= 5-7B parameters, DDP training will OOM, which is why we opt for FSDP.
"""

import shutil
from pathlib import Path
from typing import Optional

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from torch.optim import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup, \
        get_constant_schedule_with_warmup, get_constant_schedule
from prismatic.util.infinite_schedule import get_infinite_schedule_with_warmup_rsqrt_cooldown

from prismatic.overwatch import initialize_overwatch
from prismatic.training.strategies.base_strategy import TrainingStrategy

from transformers import AutoConfig
from collections import OrderedDict

import schedulefree

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class DDPStrategy(TrainingStrategy):
    @overwatch.rank_zero_only
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None:
        """Save a checkpoint to the `run_dir` only containing the state_dicts for trainable parameters by default."""
        # Remove DDP wrappers only if the LLM backbone is actually wrapped in DDP (i.e., stages
        # other than `align`). For the `align` stage the backbone is kept frozen and *not* wrapped
        # to conserve memory, so we skip unwrapping to avoid disrupting ongoing DDP training on
        # the projector.
        if isinstance(self.vlm.llm_backbone, DDP):
            self.remove_ddp_wrapper()
            wrappers_removed = True
        else:
            wrappers_removed = False
        # # Splinter State Dictionary by Top-Level Submodules (or subset, if `only_trainable`)
        full_vlm_state_dict = self.vlm.state_dict()
        # model_state_dicts = {
        #     mkey: OrderedDict() for mkey in (self.trainable_module_keys if only_trainable else self.all_module_keys)
        # }
        model_state_dicts = {
            mkey: getattr(self.vlm, mkey).state_dict()
            for mkey in (self.trainable_module_keys if only_trainable else self.all_module_keys)
        }
        # Iterate through `full_vlm_state_dict` and split `mkey.{full_dotted_path}` -> `mkey: {full_dotted_path}`
        for key, param in full_vlm_state_dict.items():
            for mkey in model_state_dicts:
                if key.startswith(mprefix := f"{mkey}."):
                    model_state_dicts[mkey][key.removeprefix(mprefix)] = param
        if self.lr_scheduler_type == 'schedule-free':
            self.optimizer.eval()
        optimizer_state_dict = self.optimizer.state_dict()

        # Set Checkpoint Path =>> Embed *minimal* training statistics!
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if train_loss is None:
            checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss=inf.pt"
        else:
            checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss={train_loss:.4f}.pt"

        # Save Checkpoint & Copy Latest to `latest-checkpoint.pt`
        torch.save({"model": model_state_dicts, "optimizer": optimizer_state_dict}, checkpoint_path)
        shutil.copy(checkpoint_path, checkpoint_dir / "latest-checkpoint.pt")

        # Re-wrap the model with DDP so that training can continue seamlessly after the checkpoint
        if wrappers_removed:
            self.rewrap_ddp()
        # supported_classes = (PeftModel,PeftModelForCausalLM)
        # peft_dir = run_dir / "checkpoint_llm_only"
        # peft_dir.mkdir(parents=True, exist_ok=True)
        # llm_backbone = unwrap_model(self.vlm).llm_backbone
        # tokenizer = llm_backbone.tokenizer
        # llm_backbone = llm_backbone.llm
        # # print(llm_backbone)
        # if isinstance(llm_backbone, supported_classes):
        #     overwatch.info(f"Saving LLM Backbone to {peft_dir}")
        #     llm_backbone.save_pretrained(
        #         peft_dir, state_dict=model_state_dicts['llm_backbone'],
        #         safe_serialization=False
        #     )
        #     tokenizer.save_pretrained(peft_dir)
        #     config = llm_backbone.config  # Access the config directly from the model
        #     config.save_pretrained(peft_dir)

    def run_setup(self, run_dir: Path) -> None:
        
        # Gradient Checkpointing Setup
        if self.enable_gradient_checkpointing:
            # For Gradient Checkpointing --> we make the assumption that the "bulk" of activation memory is taken up
            #     by the LLM; because we also make the explicit assumption that each LLM is derived from a HF
            #     pretrained model, the only thing we *need* to do (technically) is call `gradient_checkpoint_enable`
            #     on `self.llm_backbone`.
            #
            # What does it actually do? --> runs the *generic* custom_forward + torch.utils.checkpoint.checkpoint logic
            #   => github.com/huggingface/transformers/.../models/llama/modeling_llama.py#L692-L706
            #
            # Additional Reference (to better understand gradient checkpointing in PyTorch writ large)
            #   => github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb
            overwatch.info("Enabling Gradient Checkpointing on LLM Backbone", ctx_level=1)
            self.vlm.llm_backbone.enable_gradient_checkpointing()

        # Move to Device =>> Note parameters are in full precision (*mixed precision* will only autocast as appropriate)
        overwatch.info("Placing Entire VLM (Vision Backbone, LLM Backbone, Projector Weights) on GPU", ctx_level=1)
        self.vlm.to(self.device_id)

        # Wrap with Distributed Data Parallel
        #   => Note: By default, wrapping naively with DDP(self.vlm) will initialize a *separate* buffer on GPU that
        #            is the same size/dtype as the model parameters; this will *double* GPU memory!
        # - stackoverflow.com/questions/68949954/model-takes-twice-the-memory-footprint-with-distributed-data-parallel
        overwatch.info("Wrapping VLM with Distributed Data Parallel", ctx_level=1)
        #self.vlm = DDP(self.vlm, device_ids=[self.device_id], gradient_as_bucket_view=True, find_unused_parameters=True)
        # Wrap trainable components with Distributed Data Parallel
        if self.cfg.stage != "align":
            self.vlm.llm_backbone = DDP(self.vlm.llm_backbone, device_ids=[self.device_id], gradient_as_bucket_view=True)
        self.vlm.projector = DDP(self.vlm.projector, device_ids=[self.device_id], gradient_as_bucket_view=True)


        # Create Optimizer and LR Scheduler =>> note that most of the LR Schedulers we use require `max_steps/epochs`
        #   => Optimizer should only operate on parameters that are *unfrozen* / trainable!
        trainable_params = [param for param in self.vlm.parameters() if param.requires_grad]
        overwatch.info(f"Number of Trainable Parameters: {len(trainable_params)}")

        if(self.mitigation in ['lora','sgm','ia3']) and self.merges_after_steps > 0:
            assert self.weight_decay == 0, "DDP training does not currently support `weight_decay` > 0!"
            self.optimizer = AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)

            # for param_group in self.optimizer.param_groups:
            #     param_group["lr"] = 0.0
            self.lr_scheduler = None
            if self.max_steps is None:
                num_training_steps = (self.n_train_examples * self.epochs) // self.global_batch_size
            else:
                num_training_steps = self.max_steps
            num_warmup_steps = int(num_training_steps * self.warmup_ratio)
            overwatch.info("Optimizer and LR Scheduler will be initialized in run_training.")

        elif self.lr_scheduler_type == "linear-warmup+cosine-decay":
            if self.max_steps is None:
                num_training_steps = (self.n_train_examples * self.epochs) // self.global_batch_size
            else:
                num_training_steps = self.max_steps

            # Set warmup steps (floor) based on `warmup_ratio` (should be 0.03 - 0.05)
            num_warmup_steps = int(num_training_steps * self.warmup_ratio)

            assert self.weight_decay == 0, "DDP training does not currently support `weight_decay` > 0!"
            self.optimizer = AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
            # Get optimizer state keys 
            self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps)
        elif self.lr_scheduler_type == "schedule-free": # Facebook's https://github.com/facebookresearch/schedule_free
            assert self.weight_decay == 0, "DDP training does not currently support `weight_decay` > 0!"
            if self.max_steps is None:
                num_training_steps = (self.n_train_examples * self.epochs) // self.global_batch_size
            else:
                num_training_steps = self.max_steps

            # Set warmup steps (floor) based on `warmup_ratio` (should be 0.03 - 0.05)
            num_warmup_steps = int(num_training_steps * self.warmup_ratio)

            self.optimizer = schedulefree.AdamWScheduleFree(trainable_params, lr=self.learning_rate, warmup_steps=num_warmup_steps)
            self.lr_scheduler = None # No scheduler 
        elif self.lr_scheduler_type == "linear-warmup+constant":
            assert self.weight_decay == 0, "DDP training does not currently support `weight_decay` > 0!"
            if self.max_steps is None:
                num_training_steps = (self.n_train_examples * self.epochs) // self.global_batch_size
            else:
                num_training_steps = self.max_steps

            # Set warmup steps (floor) based on `warmup_ratio` (should be 0.03 - 0.05)
            num_warmup_steps = int(num_training_steps * self.warmup_ratio)

            self.optimizer = AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
            self.lr_scheduler = get_constant_schedule_with_warmup(self.optimizer, num_warmup_steps)
        elif self.lr_scheduler_type == "constant":
            assert self.weight_decay == 0, "DDP training does not currently support `weight_decay` > 0!"
            self.optimizer = AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)            
            self.lr_scheduler = get_constant_schedule(self.optimizer)
        elif self.lr_scheduler_type == "infinite+rsqrt-cooldown":
            assert self.weight_decay == 0, "DDP training does not currently support `weight_decay` > 0!"
            if self.max_steps is None:
                num_training_steps = (self.n_train_examples * self.epochs) // self.global_batch_size
            else:
                num_training_steps = self.max_steps
            # Set warmup steps (floor) based on `warmup_ratio` (should be 0.03 - 0.05)
            num_warmup_steps = int(num_training_steps * self.warmup_ratio)

            self.optimizer = AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
            self.lr_scheduler = get_infinite_schedule_with_warmup_rsqrt_cooldown(self.optimizer, num_warmup_steps=num_warmup_steps, \
                decay_steps=num_training_steps - 2 * num_warmup_steps, cooldown_steps=num_warmup_steps)
        else:
            raise ValueError(f"Learning Rate Schedule with type `{self.lr_scheduler_type}` is not supported!")

        # Finalize Setup =>> Log
        overwatch.info(
            "DDP Strategy =>> Finalized Training Setup:\n"
            f"         |-> Global (Effective) Batch Size = {self.global_batch_size}\n"
            f"         |-> Per-Device Batch Size = {self.per_device_batch_size}\n"
            f"         |-> Distributed World Size = {overwatch.world_size()}\n"
            f"         |-> Gradient Accumulation Steps = {self.grad_accumulation_steps}\n\n"
            f"         |-> LLM Backbone Gradient Checkpointing = {self.enable_gradient_checkpointing}\n"
            f"         |-> Use Native AMP = {self.enable_mixed_precision_training} ({self.mixed_precision_dtype})\n\n"
            f"         |-> Default AdamW LR = {self.learning_rate}\n"
            f"         |-> AdamW Weight Decay = {self.weight_decay}\n"
            f"         |-> LR Scheduler Type = {self.lr_scheduler_type}\n"
            f"         |-> LR Scheduler Warmup Steps (Ratio) = {num_warmup_steps} ({self.warmup_ratio})\n"
            f"         |-> Dataset Size = {self.n_train_examples} Examples\n"
            f"         |-> Max Steps = {num_training_steps}\n"
        )
    
    def reset_optimizer(self) -> None:
        """Reset Optimizer and LR Scheduler to initial state.
                Assumes that `self.vlm` is already wrapped in DDP!
        """
        trainable_params = [param for param in self.vlm.parameters() if param.requires_grad]
        overwatch.info(f"Number of Trainable Parameters: {len(trainable_params)}")
        assert self.weight_decay == 0, "DDP training does not currently support `weight_decay` > 0!"
        if (self.mitigation in ['lora','sgm','ia3']) and self.merges_after_steps > 0:
            self.optimizer = AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.lr_scheduler_type == "linear-warmup+cosine-decay":
            self.optimizer = AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.lr_scheduler_type == "schedule-free": # Facebook's https://github.com/facebookresearch/schedule_free
            assert self.weight_decay == 0, "DDP training does not currently support `weight_decay` > 0!"
            if self.max_steps is None:
                num_training_steps = (self.n_train_examples * self.epochs) // self.global_batch_size
            else:
                num_training_steps = self.max_steps

            # Set warmup steps (floor) based on `warmup_ratio` (should be 0.03 - 0.05)
            num_warmup_steps = int(num_training_steps * self.warmup_ratio)

            self.optimizer = schedulefree.AdamWScheduleFree(trainable_params, lr=self.learning_rate, warmup_steps=num_warmup_steps)
            self.optimizer.train()
        elif self.lr_scheduler_type == "linear-warmup+constant":
            assert self.weight_decay == 0, "DDP training does not currently support `weight_decay` > 0!"
            if self.max_steps is None:
                num_training_steps = (self.n_train_examples * self.epochs) // self.global_batch_size
            else:
                num_training_steps = self.max_steps
            # Set warmup steps (floor) based on `warmup_ratio` (should be 0.03 - 0.05)
            num_warmup_steps = int(num_training_steps * self.warmup_ratio)
            self.optimizer = AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.lr_scheduler_type == "infinite+rsqrt-cooldown":
            assert self.weight_decay == 0, "DDP training does not currently support `weight_decay` > 0!"
            if self.max_steps is None:
                num_training_steps = (self.n_train_examples * self.epochs) // self.global_batch_size
            else:
                num_training_steps = self.max_steps
            # Set warmup steps (floor) based on `warmup_ratio` (should be 0.03 - 0.05)
            num_warmup_steps = int(num_training_steps * self.warmup_ratio)

            self.optimizer = AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
            

        else:
            raise ValueError(f"Learning Rate Schedule with type `{self.lr_scheduler_type}` is not supported!")
       
    def remove_ddp_wrapper(self) -> None:
        """Remove DDP Wrapping (if present) and reinitialize VLM on CPU."""
        # In some training stages (e.g., `align`) the LLM backbone may not be wrapped
        # in DDP to save memory because its parameters are frozen. Guard against this
        # by only attempting to unwrap when the module is actually an instance of DDP.
        if isinstance(self.vlm.llm_backbone, DDP):
            self.vlm.llm_backbone = self.vlm.llm_backbone.module
        if isinstance(self.vlm.projector, DDP):
            self.vlm.projector = self.vlm.projector.module

    def rewrap_ddp(self) -> None:
        """Wrap VLM with DDP on GPU."""
        self.vlm.llm_backbone = DDP(self.vlm.llm_backbone, device_ids=[self.device_id], gradient_as_bucket_view=True)
        self.vlm.projector = DDP(self.vlm.projector, device_ids=[self.device_id], gradient_as_bucket_view=True)
        
    def clip_grad_norm(self) -> None:
        torch.nn.utils.clip_grad_norm_(self.vlm.parameters(), max_norm=self.max_grad_norm)

def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model