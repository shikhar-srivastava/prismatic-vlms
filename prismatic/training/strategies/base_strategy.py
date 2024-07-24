"""
base_strategy.py

Abstract class definition of a (distributed) training strategy, with full annotations of class methods, utility
functions, and initialization logic.

Training Strategies (DDP, FSDP-Grad, FSDP-Full) tend to have a lot of repeated components; this class does a lot of
heavy lifting.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.training.metrics import Metrics
from prismatic.util import check_bloat16_supported
from prismatic.util.batching_utils import SplitModalitySampler
from prismatic.util.data_utils import PaddedCollatorForLanguageModeling
from prismatic.util.relora import get_cosine_schedule_with_multiple_warmups, optimizer_reset
from prismatic.models.backbones.mitigation import apply_mitigation

from prismatic.util.lora_utils import capture_initial_weights, measure_lora_weight_change

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

# === Abstract Base Class for an arbitrary Training Strategy ===
class TrainingStrategy(ABC):
    def __init__(
        self,
        vlm: PrismaticVLM,
        device_id: int,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        n_train_examples: int = None,
        cfg = None,
        **_: str,
    ) -> None:
        self.soft_alpha = cfg['soft_alpha'] if isinstance(cfg, dict) else getattr(cfg, 'soft_alpha', None)
        self.mitigation = cfg['mitigation'] if isinstance(cfg, dict) else getattr(cfg, 'mitigation', None)
        self.merges_after_steps = cfg['merges_after_steps'] if isinstance(cfg, dict) else getattr(cfg, 'merges_after_steps', 0)
        self.merging_lr_warmup_steps = cfg['merging_lr_warmup_steps'] if isinstance(cfg, dict) else getattr(cfg, 'merging_lr_warmup_steps', 0.0)
        self.track_lora_plasticity = cfg['track_lora_plasticity'] if isinstance(cfg, dict) else getattr(cfg, 'track_lora_plasticity', False)
        self.compare_plasticity_steps = cfg['compare_plasticity_steps'] if isinstance(cfg, dict) else getattr(cfg, 'compare_plasticity_steps', 0)
        self.first_lora_after_warmup = cfg['first_lora_after_warmup'] if isinstance(cfg, dict) else getattr(cfg, 'first_lora_after_warmup', False)
        # Assert that if track_lora_plasticity is True, mitigation is in ['lora', 'qlora', 'sgm', 'msgm']
        assert not self.track_lora_plasticity or self.mitigation in ['lora', 'qlora', 'sgm', 'msgm'], "Plasticity tracking is only supported with LoRA and SGM mitigations."

        self.cfg = cfg
        self.vlm, self.device_id = vlm, device_id

        # Get relevant VLM instance parameters before they get (potentially) wrapped
        self.all_module_keys, self.trainable_module_keys = self.vlm.all_module_keys, self.vlm.trainable_module_keys
        self.llm_transformer_layer_cls = self.vlm.llm_backbone.transformer_layer_cls

        # Optimization Parameters
        self.epochs, self.max_steps = epochs, max_steps
        self.global_batch_size, self.per_device_batch_size, self.n_train_examples = global_batch_size, per_device_batch_size, n_train_examples

        self.learning_rate, self.weight_decay, self.max_grad_norm = learning_rate, weight_decay, max_grad_norm
        self.lr_scheduler_type, self.warmup_ratio = lr_scheduler_type, warmup_ratio

        # Generic Strategy Parameters
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.reduce_in_full_precision = reduce_in_full_precision
        self.mixed_precision_dtype = mixed_precision_dtype

        # DataLoader Parameters
        self.worker_init_fn = worker_init_fn

        # Optimizers & Scheduler (initialized in `run_setup`)
        self.optimizer, self.lr_scheduler = None, None

        # Lightweight Validation
        assert (
            self.global_batch_size % self.per_device_batch_size == 0
        ), "Per-Device Batch Size must evenly divide Global Batch Size!"
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size()
        if self.enable_mixed_precision_training:
            assert self.mixed_precision_dtype == torch.bfloat16, "Only BF16 mixed precision training is supported!"
            assert check_bloat16_supported(), "BFloat16 is not supported on this hardware; unset `mixed_precision`"

    @abstractmethod
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None: ...

    @abstractmethod
    def run_setup(self, run_dir: Path) -> None: ...

    @abstractmethod
    def clip_grad_norm(self) -> None: ...

    def lora_merging(self):
        # Merge the LoRA adapter into the network using the merge_and_unload method.
        # Assess the class of the self.vlm
        assert self.vlm.__class__.__name__ != 'FullyShardedDataParallel', "LoRA Merging is not currently implemented with FSDP."
        
        if isinstance(self.vlm, torch.nn.parallel.DistributedDataParallel):
            if has_lora_module(self.vlm.module.llm_backbone.llm):
                self.vlm.module.llm_backbone.llm = self.vlm.module.llm_backbone.llm.merge_and_unload()
                self.vlm.module.llm_backbone.llm = apply_mitigation(self.vlm.module.llm_backbone.llm, cfg=self.cfg)
            else:
                self.vlm.module.llm_backbone.llm = apply_mitigation(self.vlm.module.llm_backbone.llm, cfg=self.cfg)
            self.vlm.module.llm_backbone.llm.train()
        elif self.vlm.llm_backbone.__class__.__name__ == 'DistributedDataParallel':
            if has_lora_module(self.vlm.llm_backbone.module.llm):
                self.vlm.llm_backbone.module.llm = self.vlm.llm_backbone.module.llm.merge_and_unload()
                self.vlm.llm_backbone.module.llm = apply_mitigation(self.vlm.llm_backbone.module.llm, cfg=self.cfg)
            else:
                self.vlm.llm_backbone.module.llm = apply_mitigation(self.vlm.llm_backbone.module.llm, cfg=self.cfg)
            self.vlm.llm_backbone.module.llm.train()
        else:
            if has_lora_module(self.vlm.llm_backbone.llm):
                self.vlm.llm_backbone.llm = self.vlm.llm_backbone.llm.merge_and_unload()
                self.vlm.llm_backbone.llm = apply_mitigation(self.vlm.llm_backbone.llm, cfg=self.cfg)
            else:
                self.vlm.llm_backbone.llm = apply_mitigation(self.vlm.llm_backbone.llm, cfg=self.cfg)
            self.vlm.llm_backbone.llm.train()
        
        if self.lr_scheduler_type == 'schedule-free':
            self.optimizer.train()

    def run_training(
        self,
        dataset: Dataset,
        collator: PaddedCollatorForLanguageModeling,
        metrics: Metrics,
        stage: str = "finetune",
        batch_construction_strategy: str = "split-modality",
        seed: int = 7,
        cfg = None,
    ) -> None:
        """Run the training loop for the given `dataset` and `collator`; log losses, results to `metrics`"""
        if "finetune" in stage and batch_construction_strategy == "split-modality":
            # Instantiate the split-modality sampler; if you want to extend with other batch construction schemes,
            #   (e.g., grouping by length) =>> can easily add them here!
            modality_lengths = dataset.get_modality_lengths()
            sampler = SplitModalitySampler(
                dataset,
                modality_lengths,
                global_batch_size=self.global_batch_size,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                seed=seed,
                drop_last=False,
            )

        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                shuffle=True,
                seed=seed,
                drop_last=False,
            )

        # Create a DataLoader with the initialized sampler, per-device-bsz, and collator
        dataloader = DataLoader(
            dataset,
            batch_size=self.per_device_batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=2,
            worker_init_fn=self.worker_init_fn,
        )
        initial_weights = None
        last_weights = None
        # Max Steps vs. Epochs Computation
        steps_per_epoch = len(dataloader) // self.grad_accumulation_steps
        
        if ((self.mitigation not in ['lora','sgm','ia3']) or self.vlm.__class__.__name__ == 'FullyShardedDataParallel') and self.merges_after_steps > 0:
            overwatch.error(f"LoRA Merging is not supported with {self.mitigation} mitigation or FSDP. Disabling LoRA Merging.")
            self.merges_after_steps = 0
        if self.merges_after_steps > 0:
            assert self.mitigation == 'lora', "LoRA Merging is only supported with LoRA mitigation."
            num_training_steps = (steps_per_epoch + 2) * self.epochs # Added as a small offset so that num_training_steps % restart_every == 0
            # Set warmup steps (floor) based on `warmup_ratio` (should be 0.03 - 0.05)
            num_warmup_steps = int(num_training_steps * self.warmup_ratio)
        
            self.lr_scheduler = get_cosine_schedule_with_multiple_warmups(self.optimizer,
                            num_training_steps=num_training_steps, 
                            first_warmup_steps=num_warmup_steps,
                            restart_warmup_steps=self.merging_lr_warmup_steps,
                            restart_every=self.merges_after_steps,
                            min_lr_ratio=0.0)
            overwatch.info(f"LoRA Merging Cosine LR with Restarts Scheduler Initialized.\n\
                            Total Training Steps: {num_training_steps}\n\
                            Warmup Steps: {num_warmup_steps}\n\
                            Restart Warmup Steps: {self.merging_lr_warmup_steps}\n\
                            Restart Every: {self.merges_after_steps}")
        else:
            if self.mitigation == 'lora':
                overwatch.info(f"No LoRA Merging.")

        if self.max_steps is not None and steps_per_epoch < self.max_steps:
            # Just set `epochs` to some large number --> we'll short-circuit based on steps anyway
            self.epochs = 100

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(
                (self.epochs * (len(dataloader) // self.grad_accumulation_steps))
                if self.max_steps is None
                else self.max_steps
            ),
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            for epoch in range(self.epochs):
                self.vlm.train()
                if self.lr_scheduler_type == 'schedule-free':
                    self.optimizer.train() 
                sampler.set_epoch(epoch)

                # Zero-Gradients (just in case)
                self.optimizer.zero_grad()

                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                for train_idx, batch in enumerate(dataloader):
                    if self.__class__.__name__ == 'DDPStrategy':
                        # DDP does not automatically move the data to the device.
                        batch = {k: v.to(self.device_id) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    with torch.autocast(
                        "cuda",
                        dtype=self.mixed_precision_dtype,
                        enabled=self.enable_mixed_precision_training,
                    ):  
                        if self.soft_alpha is None:
                            output: CausalLMOutputWithPast = self.vlm(
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                pixel_values=batch["pixel_values"],
                                labels=batch["labels"],
                                multimodal_indices=batch["multimodal_indices"],
                            )
                        else:
                            output, fused_labels = self.vlm(
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                pixel_values=batch["pixel_values"],
                                labels=batch["labels"],
                                multimodal_indices=batch["multimodal_indices"],
                                return_labels=True if self.soft_alpha is not None else False,
                            )

                    if self.soft_alpha is not None:
                        shift_logits = output.logits[:, :-1, :].contiguous()
                        valid_targets = fused_labels[:, 1:].contiguous()

                        num_classes = shift_logits.size(-1)

                        # Handling special value -100 in targets
                        mask = (valid_targets == -100)
                        valid_targets[mask] = 0  # Replace -100 with 0 or another neutral index

                        confidence = 1.0 - self.soft_alpha
                        label_smoothing = self.soft_alpha / (num_classes - 1)
                        targets_smooth = torch.full_like(shift_logits, label_smoothing)

                        # # Ensure all tensors are on the same device and have the same dtype
                        # valid_targets = valid_targets.to(device=shift_logits.device, dtype=torch.int64)

                        # Apply scatter only on valid targets
                        targets_smooth.scatter_(-1, valid_targets.unsqueeze(-1), confidence)

                        # Apply mask to neutralize the effect of -100 in loss calculation
                        targets_smooth[mask.unsqueeze(-1).expand_as(targets_smooth)] = 0
                        # Loss calculation with try-except block
                        loss_fct = torch.nn.CrossEntropyLoss()
                        try:
                            loss = loss_fct(shift_logits.view(-1, num_classes), targets_smooth.view(-1, num_classes))
                        except RuntimeError as e:
                            print("Error during loss calculation:", e)
                            print("Shapes at loss calculation:")
                            print("shift_logits:", shift_logits.view(-1, num_classes).shape)
                            print("targets_smooth:", targets_smooth.view(-1, num_classes).shape)
                            raise e
          
                    else:
                        loss = output.loss

                    # if self.soft_alpha is not None:
                    #     num_classes = output.logits.size(-1)  # Assuming shape [batch_size, seq_length, num_classes]
                    #     valid_mask = fused_labels != IGNORE_INDEX
                    #     valid_labels = fused_labels[valid_mask]

                    #     if valid_labels.numel() > 0:
                    #         # Adjusted smoothing value calculation
                    #         base_smoothing = self.soft_alpha / (num_classes - 1) if num_classes > 1 else 0
                    #         confidence = 1 - self.soft_alpha

                    #         # Initialize the tensor for smoothed labels
                    #         targets_smooth = torch.full((valid_labels.size(0), num_classes), base_smoothing, device=fused_labels.device)
                    #         targets_smooth.scatter_(1, valid_labels.unsqueeze(1), confidence + base_smoothing)

                    #         # Gather logits corresponding to valid labels for loss computation
                    #         logits = output.logits[valid_mask].view(-1, num_classes)
                    #         loss = F.cross_entropy(logits, targets_smooth, reduction='mean')
                    #     else:
                    #         # If there are no valid labels, default to a zero loss
                    #         loss = torch.tensor(0.0, device=output.logits.device)
                    # else:
                    #     # Use the default loss calculated by the model when label smoothing is not applied
                    #     loss = output.loss

                    # Commit Loss (Prior to Gradient Accumulation Normalization)
                    metrics.commit(loss=loss)
                    # Normalize Loss to account for Gradient Accumulation --> Backward!
                    # [IMPORTANT] Technically speaking, doing gradient accumulation in this way is "incorrect"; this is
                    #             because in general, each batch has a *different number of masked out tokens* (because
                    #             we're instruct-tuning). Taking the mean over two unbalanced means != the right thing!
                    #
                    #             HOWEVER -- at least at the 7B scale, the "naive" approach is just as performant as
                    #             the "correct" implementation, without adding extra complexity.
                    #
                    # That being said =>> at the 13B scale, *no matter what we tried, ANY gradient accumulation is just
                    #   really bad for downstream performance. Initial investigation shows that BF16 accumulation
                    #   just really tanks in precision... and don't have a good/clean way to fix this. Would love for
                    #   someone to PR and fix this (and I'd greatly appreciate it!!!)
                    normalized_loss = loss / self.grad_accumulation_steps
                    normalized_loss.backward()

                    # Step =>> Only if Done w/ Gradient Accumulation
                    if (train_idx + 1) % self.grad_accumulation_steps == 0:
                        metrics.commit(update_step_time=True)
                        # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                        self.clip_grad_norm()
                        # Optimizer & LR Scheduler Step
                        if self.lr_scheduler_type == 'schedule-free':
                            self.optimizer.step() 
                        else:
                            self.optimizer.step()
                            self.lr_scheduler.step()
                        self.optimizer.zero_grad()
                        # Push Metrics
                        metrics.commit(global_step=metrics.global_step + 1, \
                                lr=self.optimizer.param_groups[0]['lr'] if self.lr_scheduler_type != 'schedule-free' else self.optimizer.get_lr())

                        # Check for Termination & Save Final Checkpoint (in case `max_steps` is not None)
                        if self.max_steps is not None and metrics.global_step >= self.max_steps:
                            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                            dist.barrier()
                            return

                        if self.track_lora_plasticity and ((train_idx + 1) % self.compare_plasticity_steps == 0):
                            lora_layers = ['lora_']  # Adjust this list based on the actual naming convention of the LoRA layers
                            if isinstance(self.vlm, torch.nn.parallel.DistributedDataParallel):
                                model = self.vlm.module.llm_backbone.llm
                            elif self.vlm.llm_backbone.__class__.__name__ == 'DistributedDataParallel':
                                model = self.vlm.llm_backbone.module.llm
                            elif self.vlm.__class__.__name__ == 'FullyShardedDataParallel':
                                model = self.vlm.llm_backbone.llm
                            else:
                                raise ValueError("No excepted vlm class name in LoRA plasticity tracking")
                            
                            if initial_weights is None:
                                initial_weights = capture_initial_weights(model, lora_layers)
                            if last_weights is None:
                                last_weights = capture_initial_weights(model, lora_layers)
                            else:
                                weight_change_wrt_initial = measure_lora_weight_change(model, initial_weights, lora_layers)
                                weight_change_wrt_last = measure_lora_weight_change(model, last_weights, lora_layers)
                                last_weights = capture_initial_weights(model, lora_layers)
                                metrics.commit(global_step=metrics.global_step + 1, \
                                lora_plasticity = weight_change_wrt_last)
                                metrics.commit(global_step=metrics.global_step + 1, \
                                    lora_plasticity_first = weight_change_wrt_initial)


                        if (self.merges_after_steps > 0):
                            if (self.lr_scheduler.get_last_lr()[0] == 0.0) and ((train_idx + 1)// self.grad_accumulation_steps > num_warmup_steps):
                                # A reset has occured in the LR scheduler
                                overwatch.info(f"Performing LoRA Merging at step {train_idx} or {(train_idx + 1)// self.grad_accumulation_steps} norm step")
                                #self.remove_ddp_wrapper()
                                self.lora_merging()
                                overwatch.info(f"LoRA Merging Complete")
                                #self.rewrap_ddp()
                                self.vlm.train()
                                if self.lr_scheduler_type == 'schedule-free':
                                    self.optimizer.train()
                                else:
                                    trainable_params = [param for param in self.vlm.parameters() if param.requires_grad]
                                    lora_params = [p for n, p in self.vlm.named_parameters() if p.requires_grad and "lora_" in n]
                                    # print(f"LoRA Parameters: {lora_params}")
                                    # optimizer_reset(self.optimizer, 
                                    #             reset_params = lora_params,
                                    #             optimizer_state_keys = ["exp_avg", "exp_avg_sq"],
                                    #             reset_optimizer_on_relora = True,
                                    #             optimizer_random_pruning=0.0,
                                    #             optimizer_magnitude_pruning=0.0)
                                    optimizer_reset(self.optimizer)
                        
                        status = metrics.push()
                        # Update Progress Bar
                        progress.update()
                        progress.set_description(status)

            # Save checkpoint at end each epoch (if `self.max_steps` is None)
            if self.max_steps is None:
                self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                dist.barrier()



def has_lora_module(model):
    """
    Check if the given model contains any LoRA modules.

    Args:
        model (torch.nn.Module): The model to check for LoRA modules.

    Returns:
        bool: True if the model contains LoRA modules, False otherwise.
    """
    for name, module in model.named_modules():
        if 'lora' in name.lower():
            return True
        # If you know specific types or classes that represent LoRA layers,
        # you can check for them directly, e.g.,
        # if isinstance(module, LoraLayer):
        #     return True
    return False