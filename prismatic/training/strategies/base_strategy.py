"""
base_strategy.py

Abstract class definition of a (distributed) training strategy, with full annotations of class methods, utility
functions, and initialization logic.

Training Strategies (DDP, FSDP-Grad, FSDP-Full) tend to have a lot of repeated components; this class does a lot of
heavy lifting.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, Optional

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
from prismatic.preprocessing.validation import load_validation_manager
from prismatic.util.relora import get_cosine_schedule_with_multiple_warmups, optimizer_reset
from prismatic.models.backbones.mitigation import apply_mitigation

from prismatic.util.lora_utils import capture_initial_weights, \
    measure_lora_weight_change, measure_lora_weight_change_per_layer, log_weight_change_detailed

import os
import numpy as np
import wandb

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
        
        ############
        ### Specific configuration related

        # Align Loss
        self.align_loss = cfg['align_loss'] if isinstance(cfg, dict) else getattr(cfg, 'align_loss', False)
        self.align_weight = cfg['align_weight'] if isinstance(cfg, dict) else getattr(cfg, 'align_weight', 0.01)
        self.norm_reg = cfg['norm_reg'] if isinstance(cfg, dict) else getattr(cfg, 'norm_reg', False)
        self.norm_reg_weight = cfg['norm_reg_weight'] if isinstance(cfg, dict) else getattr(cfg, 'norm_reg_weight', 0.01)


        self.track_embeddings = cfg['track_embeddings'] if isinstance(cfg, dict) else getattr(cfg, 'track_embeddings', False)
        self.track_embeddings_values = cfg['track_embeddings_values'] if isinstance(cfg, dict) else getattr(cfg, 'track_embeddings_values', False)
        self.track_covariance = cfg['track_covariance'] if isinstance(cfg, dict) else getattr(cfg, 'track_covariance', False)
        self.use_precomputed_covariance = cfg['use_precomputed_covariance'] if isinstance(cfg, dict) else getattr(cfg, 'use_precomputed_covariance', False)
        self.precomputed_covariance_path = cfg['precomputed_covariance_path'] if isinstance(cfg, dict) else getattr(cfg, 'precomputed_covariance_path', "/home/aac/ssrivas9/prismatic-vlms/text_covariance_186K.pt")
        self.track_avg_rank = cfg['track_avg_rank'] if isinstance(cfg, dict) else getattr(cfg, 'track_avg_rank', False)
        
        # Measure Rank Entropy
        self.measure_rank_entropy = cfg['measure_rank_entropy'] if isinstance(cfg, dict) else getattr(cfg, 'measure_rank_entropy', False)

        # Distillation with Teacher LLM
        self.llm_teacher_checkpoint = cfg['llm_teacher_checkpoint'] if isinstance(cfg, dict) else getattr(cfg, 'llm_teacher_checkpoint', None)
        self.scale_patch_embeddings = cfg['scale_patch_embeddings'] if isinstance(cfg, dict) else getattr(cfg, 'scale_patch_embeddings', False)
        self.stableadam = cfg['stableadam'] if isinstance(cfg, dict) else getattr(cfg, 'stableadam', False)
        
        # Validation Set Management
        self.enable_validation_tracking = cfg['enable_validation_tracking'] if isinstance(cfg, dict) else getattr(cfg, 'enable_validation_tracking', False)
        self.validation_set_dir = cfg['validation_set_dir'] if isinstance(cfg, dict) else getattr(cfg, 'validation_set_dir', None)
        self.validation_frequency = cfg['validation_frequency'] if isinstance(cfg, dict) else getattr(cfg, 'validation_frequency', 100)
        self.validation_manager = None

        # Saving/Loading Logits Utilities
        self.save_logits = cfg['save_logits'] if isinstance(cfg, dict) else getattr(cfg, 'save_logits', False)
        self.save_logits_dir = cfg['save_logits_dir'] if isinstance(cfg, dict) else getattr(cfg, 'save_logits_dir', None)
        self.load_logits = cfg['load_logits'] if isinstance(cfg, dict) else getattr(cfg, 'load_logits', False)
        self.load_logits_dir = cfg['load_logits_dir'] if isinstance(cfg, dict) else getattr(cfg, 'load_logits_dir', None)

        # Soft Targets Variants
        self.soft_alpha = cfg['soft_alpha'] if isinstance(cfg, dict) else getattr(cfg, 'soft_alpha', None)
        self.soft_alpha_masked_interpolation = cfg['soft_alpha_masked_interpolation'] if isinstance(cfg, dict) else getattr(cfg, 'soft_alpha_masked_interpolation', False)
        self.label_smoothing = cfg['label_smoothing'] if isinstance(cfg, dict) else getattr(cfg, 'label_smoothing', 0.0)
        self.add_K = cfg['add_K'] if isinstance(cfg, dict) else getattr(cfg, 'add_K', None)
        self.add_K_percentage = cfg['add_K_percentage'] if isinstance(cfg, dict) else getattr(cfg, 'add_K_percentage', False)
        self.set_to_one = cfg['set_to_one'] if isinstance(cfg, dict) else getattr(cfg, 'set_to_one', False)
        self.max_logit = cfg['max_logit'] if isinstance(cfg, dict) else getattr(cfg, 'max_logit', False)
        self.use_logits_in_max_logit = cfg['use_logits_in_max_logit'] if isinstance(cfg, dict) else getattr(cfg, 'use_logits_in_max_logit', False)

        self.soft_output_logits = cfg['soft_output_logits'] if isinstance(cfg, dict) else getattr(cfg, 'soft_output_logits', True)
        self.interpolation_dtype = cfg['interpolation_dtype'] if isinstance(cfg, dict) else getattr(cfg, 'interpolation_dtype', 'float32')
        self.interpolation_loss = cfg['interpolation_loss'] if isinstance(cfg, dict) else getattr(cfg, 'interpolation_loss', 'cross')
        self.masked_with_logits = cfg['masked_with_logits'] if isinstance(cfg, dict) else getattr(cfg, 'masked_with_logits', False)
        self.masked_with_logits_label_smoothing = cfg['masked_with_logits_label_smoothing'] if isinstance(cfg, dict) else getattr(cfg, 'masked_with_logits_label_smoothing', 0.01)
        self.masked_with_logits_mask_weight = cfg['masked_with_logits_mask_weight'] if isinstance(cfg, dict) else getattr(cfg, 'masked_with_logits_mask_weight', 0.01)
        
        
        # General mitigation methods related
        self.mitigation = cfg['mitigation'] if isinstance(cfg, dict) else getattr(cfg, 'mitigation', None)
        
        # LoRA Merging
        self.merges_after_steps = cfg['merges_after_steps'] if isinstance(cfg, dict) else getattr(cfg, 'merges_after_steps', 0)
        self.merging_lr_warmup_steps = cfg['merging_lr_warmup_steps'] if isinstance(cfg, dict) else getattr(cfg, 'merging_lr_warmup_steps', 0.0)
        
        # Tracking Weight Plasticity of LoRA modules
        self.track_lora_plasticity = cfg['track_lora_plasticity'] if isinstance(cfg, dict) else getattr(cfg, 'track_lora_plasticity', False)
        self.track_ft_plasticity = cfg['track_ft_plasticity'] if isinstance(cfg, dict) else getattr(cfg, 'track_ft_plasticity', False)
        self.compare_plasticity_steps = cfg['compare_plasticity_steps'] if isinstance(cfg, dict) else getattr(cfg, 'compare_plasticity_steps', 0)
        self.first_lora_after_warmup = cfg['first_lora_after_warmup'] if isinstance(cfg, dict) else getattr(cfg, 'first_lora_after_warmup', False)
        # Assert that if track_lora_plasticity is True, mitigation is in ['lora', 'qlora', 'sgm', 'msgm']
        assert not self.track_lora_plasticity or self.mitigation in ['lora', 'adalora', 'qlora', 'sgm', 'msgm'], "Plasticity tracking is only supported with LoRA and SGM mitigations."
        # Assert that if track_ft_plasticity is True, mitigation is None
        assert not self.track_ft_plasticity or self.mitigation is None, "Fine-tuning plasticity tracking is only supported with no mitigation."
        
        # <<< MODIFIED >>> Track Layer Stats Flag
        self.track_layer_stats = cfg['track_layer_stats'] if isinstance(cfg, dict) else getattr(cfg, 'track_layer_stats', False)
        
        # <<< ADDED >>> Track Cosine Layer Stats Flag
        self.track_cosine_layer_stats = cfg['track_cosine_layer_stats'] if isinstance(cfg, dict) else getattr(cfg, 'track_cosine_layer_stats', False)

        # <<< ADDED >>> Random Image Flag
        self.random_image = cfg['random_image'] if isinstance(cfg, dict) else getattr(cfg, 'random_image', False)

        # <<< ADDED >>> Activation Distribution Stats Flag
        self.track_activation_distributions = cfg['track_activation_distributions'] if isinstance(cfg, dict) else getattr(cfg, 'track_activation_distributions', False)

        ################
        # General Configs and Training related

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

    def _compute_validation_metrics(self) -> Optional[Dict[str, float]]:
        """
        Compute validation metrics on the validation sets for both align and finetune stages.
        
        Returns:
            Dictionary of validation metrics or None if computation fails
        """
        if not self.validation_manager:
            return None
        
        validation_metrics = {}
        
        with torch.no_grad():
            for validation_stage in ["align", "finetune"]:
                if not self.validation_manager.has_validation_set(validation_stage):
                    continue
                
                # Get all validation batches for this stage
                validation_batches = self.validation_manager.get_all_validation_batches(validation_stage)
                if not validation_batches:
                    continue
                
                stage_ranks = []
                stage_min_ranks = []
                stage_max_ranks = []
                
                # Process each validation batch
                for batch in validation_batches:
                    try:
                        # Forward pass
                        output, fused_labels = self.vlm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],
                            labels=batch["labels"],
                            multimodal_indices=batch["multimodal_indices"],
                            return_labels=True,
                        )
                        
                        # Calculate ranks for this batch
                        avg_rank, min_rank, max_rank = calculate_average_rank(output, fused_labels)
                        
                        if avg_rank > 0:  # Valid rank computation
                            stage_ranks.append(avg_rank)
                            stage_min_ranks.append(min_rank)
                            stage_max_ranks.append(max_rank)
                    
                    except Exception as e:
                        overwatch.error(f"Error computing validation metrics for {validation_stage}: {e}")
                        continue
                
                # Aggregate metrics for this stage
                if stage_ranks:
                    validation_metrics[f"avg_rank_{validation_stage}"] = sum(stage_ranks) / len(stage_ranks)
                    validation_metrics[f"min_rank_{validation_stage}"] = min(stage_min_ranks)
                    validation_metrics[f"max_rank_{validation_stage}"] = max(stage_max_ranks)
                    validation_metrics[f"avg_min_rank_{validation_stage}"] = sum(stage_min_ranks) / len(stage_min_ranks)
                    validation_metrics[f"avg_max_rank_{validation_stage}"] = sum(stage_max_ranks) / len(stage_max_ranks)
                    
                    # overwatch.info(f"Validation {validation_stage}: avg_rank={validation_metrics[f'avg_rank_{validation_stage}']:.2f}, "
                    #              f"min_rank={validation_metrics[f'min_rank_{validation_stage}']:.1f}, "
                    #              f"max_rank={validation_metrics[f'max_rank_{validation_stage}']:.1f}")
        
        return validation_metrics if validation_metrics else None

    def _compute_embedding_metrics(self, vis_emb, txt_emb, alignment_loss_val=None, reg_loss=None):
        """Helper method to compute embedding metrics and prepare kwargs for metrics.commit."""
        metrics_kwargs = {}
        
        # Only compute stats if tracking embeddings
        if self.track_embeddings:
            with torch.no_grad():
                # ----- Visual embedding stats -----
                if (vis_emb is not None) and (vis_emb.numel() > 0):
                    vis_stats = compute_embedding_stats(vis_emb)
                    metrics_kwargs.update({
                        "vis_embed_mean": vis_stats["embed_mean"],
                        "vis_embed_std": vis_stats["embed_std"],
                        "vis_l2_mean": vis_stats["l2_mean"],
                        "vis_l2_std": vis_stats["l2_std"],
                    })
                else:
                    # If no images were provided, store a default of zero for each metric
                    metrics_kwargs.update({
                        "vis_embed_mean": 0.0,
                        "vis_embed_std": 0.0,
                        "vis_l2_mean": 0.0,
                        "vis_l2_std": 0.0,
                    })

                # ----- Text embedding stats -----
                if (txt_emb is not None) and (txt_emb.numel() > 0):
                    txt_stats = compute_embedding_stats(txt_emb)
                    metrics_kwargs.update({
                        "txt_embed_mean": txt_stats["embed_mean"],
                        "txt_embed_std": txt_stats["embed_std"],
                        "txt_l2_mean": txt_stats["l2_mean"],
                        "txt_l2_std": txt_stats["l2_std"],
                    })
                else:
                    metrics_kwargs.update({
                        "txt_embed_mean": 0.0,
                        "txt_embed_std": 0.0,
                        "txt_l2_mean": 0.0,
                        "txt_l2_std": 0.0,
                    })
                
                # ----- Compute cosine similarity only if both vis and txt embeddings exist -----
                if (
                    (vis_emb is not None) and (vis_emb.numel() > 0) and
                    (txt_emb is not None) and (txt_emb.numel() > 0)
                ):
                    # Flatten and cast
                    vis_flat = vis_emb.reshape(-1, vis_emb.shape[-1]).to(torch.float32)
                    txt_flat = txt_emb.reshape(-1, txt_emb.shape[-1]).to(torch.float32)

                    # Normalize each row so that dot product becomes cosine similarity
                    vis_norm = F.normalize(vis_flat, p=2, dim=-1)
                    txt_norm = F.normalize(txt_flat, p=2, dim=-1)

                    # Cosine similarity matrix: (vis_count, txt_count)
                    cos_sim = torch.matmul(vis_norm, txt_norm.transpose(0, 1))

                    # Grab min, max, and mean over all pairwise cos-sims
                    cos_sim_min = cos_sim.min().item()
                    cos_sim_max = cos_sim.max().item()
                    cos_sim_mean = cos_sim.mean().item()

                    # Log them with descriptive keys
                    metrics_kwargs["vis_txt_cosine_min"] = cos_sim_min
                    metrics_kwargs["vis_txt_cosine_max"] = cos_sim_max
                    metrics_kwargs["vis_txt_cosine_mean"] = cos_sim_mean
                else:
                    # If one is missing, store default zeros
                    metrics_kwargs["vis_txt_cosine_min"] = 0.0
                    metrics_kwargs["vis_txt_cosine_max"] = 0.0
                    metrics_kwargs["vis_txt_cosine_mean"] = 0.0


        # ----- Compute covariance if tracking it -----
        if self.track_covariance and self.align_loss:
            with torch.no_grad():
                # Only compute covariance if vis_emb has data
                if (vis_emb is not None) and (vis_emb.numel() > 0):
                    V = vis_emb.reshape(-1, vis_emb.shape[-1])
                    V_mean = V.mean(dim=0, keepdim=True)
                    Vc = V - V_mean
                    nV = Vc.shape[0]
                    cov_vis = (Vc.t() @ Vc) / max(nV - 1, 1)
                    metrics_kwargs["covariance_matrix"] = cov_vis
                else:
                    metrics_kwargs["covariance_matrix"] = torch.empty(0)

        # ----- Alignment loss -----
        if self.align_loss:
            metrics_kwargs["alignment_loss"] = alignment_loss_val if alignment_loss_val is not None else 0.0

        # ----- Regularization loss -----
        if self.norm_reg:
            metrics_kwargs["reg_loss"] = reg_loss if reg_loss is not None else 0.0

        # ----- Track raw embedding values per-dimension (if requested) -----
        if self.track_embeddings_values:
            with torch.no_grad():
                if (vis_emb is not None) and (vis_emb.numel() > 0):
                    vis_emb_flat = vis_emb.reshape(-1, vis_emb.shape[-1]).to(torch.float32)
                    mean_per_dimension = vis_emb_flat.mean(dim=0)  # [hidden_dim]
                    metrics_kwargs["projected_embeddings_values"] = mean_per_dimension
                else:
                    metrics_kwargs["projected_embeddings_values"] = torch.empty(0)

        return metrics_kwargs

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
            #pin_memory=True,
        )

        # Initialize validation manager if validation tracking is enabled
        if self.enable_validation_tracking and self.validation_set_dir:
            overwatch.info(f"Loading validation sets from {self.validation_set_dir}")
            self.validation_manager = load_validation_manager(self.validation_set_dir, self.device_id)
            if self.validation_manager is None:
                overwatch.error("Failed to load validation manager, disabling validation tracking")
                self.enable_validation_tracking = False
            else:
                overwatch.info(f"Validation manager loaded successfully")
                # Log validation set info
                for stage in ["align", "finetune"]:
                    if self.validation_manager.has_validation_set(stage):
                        val_info = self.validation_manager.get_validation_info(stage)
                        overwatch.info(f"{stage.capitalize()} validation: {val_info['num_samples']} samples, "
                                     f"{val_info['num_batches']} batches")
        
        # Fallback to old batch loading for backward compatibility
        if self.track_avg_rank and not self.enable_validation_tracking:
            overwatch.info("Using legacy single-batch validation (consider upgrading to validation sets)")
            batch_align = torch.load('batch_align.pt')
            batch_align = {k: v.to(self.device_id) if isinstance(v, torch.Tensor) else v for k, v in batch_align.items()}
            batch_finetune = torch.load('batch_finetune.pt')
            batch_finetune = {k: v.to(self.device_id) if isinstance(v, torch.Tensor) else v for k, v in batch_finetune.items()}


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
        if self.save_logits:
            total_steps = self.epochs * len(dataloader)
        else:
            total_steps = (
                (self.epochs * (len(dataloader) // self.grad_accumulation_steps))
                if self.max_steps is None
                else self.max_steps
            )

        # metrics.log(metrics.global_step, {"test_histogram": wandb.Histogram(np.random.randn(1000))})

        status = metrics.get_status()
        with tqdm(
            total=total_steps,
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            for epoch in range(self.epochs):
                if self.save_logits:
                    # Set model to evaluation mode for inference
                    self.vlm.eval()
                    # Disable gradient checkpointing and mixed precision
                    self.vlm.llm_backbone.llm.gradient_checkpointing_disable()
                    self.enable_mixed_precision_training = False
                else:
                    self.vlm.train()
                    if self.lr_scheduler_type == 'schedule-free':
                        self.optimizer.train()
                    sampler.set_epoch(epoch)

                if not self.save_logits:
                    # Zero gradients if training
                    self.optimizer.zero_grad()

                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                for train_idx, batch in enumerate(dataloader):
                    if self.__class__.__name__ == 'DDPStrategy' or self.track_embeddings:
                        # DDP does not automatically move the data to the device.
                        batch = {k: v.to(self.device_id) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    with torch.autocast(
                        "cuda",
                        dtype=self.mixed_precision_dtype,
                        enabled=self.enable_mixed_precision_training,
                    ):  
                        # Extract sample indices
                        if stage == 'finetune' and 'idx' in batch:
                            sample_indices = batch.pop('idx')
                        else:
                            sample_indices = None

                        # <<< ADDED >>> Replace images with random noise if requested
                        if self.random_image and 'pixel_values' in batch and batch['pixel_values'] is not None:
                            pixel_values = batch['pixel_values'] # Assuming shape [B, C, H, W]
                            if 'multimodal_indices' in batch and batch['multimodal_indices'] is not None:
                                multimodal_indices = batch['multimodal_indices']
                                if len(multimodal_indices) > 0:
                                    # Get original images for multimodal samples
                                    original_images = pixel_values[multimodal_indices]
                                    # Calculate mean and std of original images
                                    img_mean = torch.mean(original_images, dim=[0, 2, 3], keepdim=True)
                                    img_std = torch.std(original_images, dim=[0, 2, 3], keepdim=True)
                                    # Generate Gaussian noise with same shape, mean, std
                                    noise = torch.randn_like(original_images) * img_std + img_mean
                                    # Replace the pixel values for multimodal samples
                                    # Ensure noise is same dtype as pixel_values
                                    pixel_values[multimodal_indices] = noise.to(pixel_values.dtype)
                                    batch['pixel_values'] = pixel_values # Update batch dictionary
                            else:
                                # If multimodal_indices is None, assume all are multimodal
                                img_mean = torch.mean(pixel_values, dim=[0, 2, 3], keepdim=True)
                                img_std = torch.std(pixel_values, dim=[0, 2, 3], keepdim=True)
                                noise = torch.randn_like(pixel_values) * img_std + img_mean
                                batch['pixel_values'] = noise.to(pixel_values.dtype)

                        # if self.save_logits:
                        #     # Inference mode: Disable gradient computations
                        #     with torch.no_grad():
                        #         output = self.vlm(
                        #             input_ids=batch["input_ids"],
                        #             attention_mask=batch["attention_mask"],
                        #             pixel_values=batch["pixel_values"] if batch["pixel_values"] is not None else None,
                        #             labels=batch["labels"],
                        #             multimodal_indices=batch["multimodal_indices"],
                        #         )
                        #     # Ensure directory exists
                        #     logits_save_dir = self.save_logits_dir

                        #     # Detach logits and move to CPU
                        #     logits = output.logits.detach().cpu()

                        #     # Save logits for each sample in the batch
                        #     for idx_in_batch, sample_idx in enumerate(sample_indices):
                        #         sample_logits = logits[idx_in_batch]
                        #         filename = f'logits_{sample_idx.item()}.pt'
                        #         filepath = os.path.join(logits_save_dir, filename)
                        #         torch.save(sample_logits, filepath)

                        #     # Release variables to free up memory
                        #     del output, logits, sample_logits
                        #     torch.cuda.empty_cache()

                        #     # Update progress bar
                        #     progress.update()
                        #     progress.set_description(f"Saved logits for batch {train_idx}")
                        #     continue
                            
                        # if self.load_logits:
                        #     # Use the teacher_logits from the batch
                        #     teacher_logits = batch.pop('teacher_logits')
                        #     if teacher_logits is None:
                        #         continue
                        #     else:
                        #         teacher_logits = teacher_logits.to(self.device_id)
                        
                        #     #print(f"Teacher logits shape: {teacher_logits.shape}, Output logits shape: {output.logits.shape}")


                        if (self.soft_alpha is not None) or (self.soft_alpha_masked_interpolation is not None or \
                        self.add_K is not None or self.set_to_one or self.max_logit or (self.label_smoothing > 0.0) or self.measure_rank_entropy):
                            vlm_output = self.vlm(
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                pixel_values=batch["pixel_values"],
                                labels=batch["labels"],
                                multimodal_indices=batch["multimodal_indices"],
                                return_labels=True,
                                align_loss=self.align_loss,
                                # <<< MODIFIED >>> Pass self.track_layer_stats to forward
                                output_layer_stats=self.track_layer_stats,
                            )
                            # If tracking layer stats, unpack the additional return values
                            # <<< MODIFIED >>> Use self.track_layer_stats
                            if self.track_layer_stats:
                                output, fused_labels, hidden_states, vis_indices = vlm_output
                                # hidden_states is a tuple: (embedding_layer_output, layer1_output, ..., layerN_output)
                            else:
                                output, fused_labels = vlm_output
                                hidden_states = None
                                vis_indices = (0,0) # Dummy indices

                            # <<< RESTORED >>> Teacher forward pass block
                            if self.vlm.llm_teacher is not None:
                                with torch.no_grad():
                                    teacher_output, teacher_fused_labels = self.vlm.teacher_forward(
                                        input_ids=batch["input_ids"],
                                        attention_mask=batch["attention_mask"],
                                        pixel_values=batch["pixel_values"],
                                        labels=batch["labels"],
                                        multimodal_indices=batch["multimodal_indices"],
                                        return_labels=True,
                                        align_loss=self.align_loss
                                    )
                        else:
                            vlm_output = self.vlm(
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                pixel_values=batch["pixel_values"],
                                labels=batch["labels"],
                                multimodal_indices=batch["multimodal_indices"],
                                align_loss=self.align_loss,
                                use_precomputed_covariance=self.use_precomputed_covariance,
                                precomputed_covariance_path=self.precomputed_covariance_path,
                                # <<< MODIFIED >>> Pass self.track_layer_stats to forward
                                output_layer_stats=self.track_layer_stats,  # <<< ADDED >>> Pass the flag
                            )

                            # Unpack based on whether align_loss or layer_stats was requested
                            # <<< MODIFIED >>> Use self.track_layer_stats
                            if self.track_layer_stats:
                                output, fused_labels, hidden_states, vis_indices = vlm_output
                                # Note: align_loss (if computed) is now part of output.loss
                                alignment_loss_val = output.alignment_loss if hasattr(output, 'alignment_loss') else None
                            elif self.align_loss:
                                output, alignment_loss_val = vlm_output
                                hidden_states = None
                                vis_indices = (0,0)
                            else:
                                output = vlm_output
                                alignment_loss_val = None
                                hidden_states = None
                                vis_indices = (0,0)

                            # <<< RESTORED >>> Get embeddings if tracking is enabled
                            if self.track_embeddings or self.track_covariance:
                                with torch.no_grad():
                                    vis_emb, txt_emb = self.vlm.get_embeddings(
                                        input_ids=batch["input_ids"],
                                        attention_mask=batch["attention_mask"],
                                        pixel_values=batch["pixel_values"],
                                        labels=batch["labels"],
                                        multimodal_indices=batch["multimodal_indices"]
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

                    elif self.label_smoothing > 0.0:
                        shift_logits = output.logits[:, :-1, :].contiguous()
                        valid_targets = fused_labels[:, 1:].contiguous()

                        num_classes = shift_logits.size(-1)

                        # Loss calculation with try-except block
                        loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
                        try:
                            loss = loss_fct(shift_logits.view(-1, num_classes), valid_targets.view(-1))
                        except RuntimeError as e:
                            print("Error during loss calculation:", e)
                            print("Shapes at loss calculation:")
                            print("shift_logits:", shift_logits.shape)
                            print("valid_targets:", valid_targets.shape)
                            raise e

                    elif self.soft_alpha_masked_interpolation is not None:

                        dtype = torch.float32 if self.interpolation_dtype == 'float32' else torch.bfloat16 # Default
                        shift_logits = output.logits[:, :-1, :].contiguous().to(dtype) # Shape: [batch_size, seq_length-1, num_classes]
                        valid_targets = fused_labels[:, 1:].contiguous().to(dtype)    # Shape: [batch_size, seq_length-1]

                        num_classes = shift_logits.size(-1)
                        mask = (valid_targets != -100)  # Ignored positions are marked with -100

                        if self.load_logits:
                            soft_probs = F.softmax(teacher_logits[:,1:,:].to(dtype), dim=-1).to(dtype).detach()
                        else:
                            # Compute soft probabilities from logits and detach to prevent gradient flow
                            soft_probs = F.softmax(output.logits[:,1:,:].to(dtype), dim=-1).to(dtype).detach()  # Shape: [batch_size, seq_length-1, num_classes]

                        batch_size, shifted_seq_length = shift_logits.size()[:2]

                        # Initialize a zero tensor for one-hot encoding
                        one_hot_targets = torch.zeros(
                            batch_size, 
                            shifted_seq_length, 
                            num_classes, 
                            device=valid_targets.device, 
                            dtype=dtype
                        )  # Shape: [batch_size, seq_length-1, num_classes]

                        # Identify valid target positions
                        valid_positions = mask.nonzero(as_tuple=False)  # Shape: [num_valid_positions, 2]
                        target_indices = valid_targets[mask].long()    # Shape: [num_valid_positions]

                        # Convert target indices to one-hot vectors
                        one_hot_valid = F.one_hot(target_indices, num_classes=num_classes).to(dtype)  # Shape: [num_valid_positions, num_classes]

                        # Assign one-hot vectors to their corresponding positions in the batch
                        batch_indices, seq_indices = valid_positions[:, 0], valid_positions[:, 1]  # Shapes: [num_valid_positions], [num_valid_positions]
                        one_hot_targets[batch_indices, seq_indices] = one_hot_valid  # Broadcasting assignment

                        # Define the interpolation factor
                        alpha = torch.tensor(self.soft_alpha_masked_interpolation, device=one_hot_targets.device, dtype=dtype)

                        # Compute dynamic soft targets without tracking gradients
                        with torch.no_grad():
                            # Interpolate between soft_probs and one_hot_targets
                            dynamic_soft_targets = alpha * soft_probs + (1.0 - alpha) * one_hot_targets  # Shape: [batch_size, seq_length-1, num_classes]
                            
                            # Apply mask based on self.masked_with_logits
                            if self.masked_with_logits:
                                # Assign original soft_probs to masked positions
                                dynamic_soft_targets = dynamic_soft_targets * mask.unsqueeze(-1).float() + soft_probs * (~mask).unsqueeze(-1).float()
                            else:
                                # Keep current behavior: set masked positions to zero
                                dynamic_soft_targets = dynamic_soft_targets * mask.unsqueeze(-1).float()# Shape: [batch_size, seq_length-1, num_classes]
                                                    
                            
                            # Optionally cast to a different dtype if required (ensure compatibility with loss function)
                            dynamic_soft_targets = dynamic_soft_targets.to(torch.bfloat16)

                        # # Verify shapes and properties
                        # assert dynamic_soft_targets.shape == (batch_size, shifted_seq_length, num_classes), \
                        #     f"Expected shape {[batch_size, shifted_seq_length, num_classes]}, but got {dynamic_soft_targets.shape}"
                        # assert torch.all(dynamic_soft_targets[~mask] == 0), "Positions with label -100 are not all zeros in dynamic_soft_targets."

                        # # Verify that valid positions have probability distributions summing to 1
                        # valid_dynamic_sum = dynamic_soft_targets[mask].sum(dim=-1)
                        # assert torch.allclose(valid_dynamic_sum, torch.ones_like(valid_dynamic_sum), atol=1e-2), \
                        #     "Valid positions in dynamic_soft_targets do not sum to 1."

                        # Compute log probabilities from shift_logits for KLDivLoss
                        if not self.soft_output_logits:
                            log_probs = F.log_softmax(shift_logits, dim=-1)  # Shape: [batch_size, seq_length-1, num_classes]

                        # Define the loss function
                        if self.interpolation_loss == 'cross':
                            if self.masked_with_logits:
                                loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=self.masked_with_logits_label_smoothing)
                            elif self.label_smoothing > 0.0:
                                loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
                            else:
                                loss_fct = torch.nn.CrossEntropyLoss()
                        elif self.interpolation_loss == 'kl':
                            loss_fct = torch.nn.KLDivLoss(reduction='batchmean') #TODO: check with reduction='mean'
                        else:
                            raise ValueError(f"Unsupported interpolation loss function: {self.interpolation_loss}")
                        # Compute the loss
                        # Reshape tensors to [batch_size * (seq_length-1), num_classes]
                        if self.soft_output_logits:
                            loss = loss_fct(
                                shift_logits.view(-1, num_classes),                # Predictions
                                dynamic_soft_targets.view(-1, num_classes)       # Targets
                            )
                        else:
                            loss = loss_fct(
                                log_probs.view(-1, num_classes),                # Predictions
                                dynamic_soft_targets.view(-1, num_classes)       # Targets
                            )

                    elif self.add_K is not None: 
                        dtype = torch.float32 if self.interpolation_dtype == 'float32' else torch.bfloat16 # Default
                        shift_logits = output.logits[:, :-1, :].contiguous().to(dtype) # Shape: [batch_size, seq_length-1, num_classes]
                        valid_targets = fused_labels[:, 1:].contiguous().to(dtype)    # Shape: [batch_size, seq_length-1]

                        num_classes = shift_logits.size(-1)
                        mask = (valid_targets != -100)  # Ignored positions are marked with -100
                        
                        if self.load_logits:
                            # Use the teacher_logits from the batch
                            target_aligned_logits = teacher_logits[:, 1:, :].contiguous().to(dtype)
                        elif self.vlm.llm_teacher is not None:
                            target_aligned_logits = teacher_output.logits[:, 1:, :].contiguous().to(dtype)
                        else:
                            target_aligned_logits = output.logits[:, 1:, :].contiguous().to(dtype)  # [batch_size, seq_length - 1, num_classes]

                        # Compute soft probabilities and detach from computational graph
                        soft_probs = F.softmax(target_aligned_logits, dim=-1).detach()  # [batch_size, seq_length - 1, num_classes]

                        # Initialize target distributions
                        targets_add_K = torch.zeros_like(soft_probs)  # [batch_size, seq_length - 1, num_classes]

                        if mask.any():
                            # Extract valid positions
                            valid_indices = valid_targets[mask].long()  # [N]
                            batch_indices, seq_indices = mask.nonzero(as_tuple=True)  # [N], [N]

                            # Gather soft probabilities at valid positions
                            soft_probs_masked = soft_probs[mask]  # [N, num_classes]

                            # Create a mask for the correct tokens
                            correct_token_mask = torch.zeros_like(soft_probs_masked, dtype=torch.bool)
                            correct_token_mask.scatter_(1, valid_indices.unsqueeze(1), True)  # [N, num_classes]

                            # Extract p_correct
                            p_correct = soft_probs_masked[correct_token_mask]  # [N]

                            # Compute delta based on whether K is a percentage
                            if self.add_K_percentage:
                                delta = p_correct * self.add_K  # [N]
                            else:
                                delta = torch.full_like(p_correct, self.add_K)  # [N]

                            # Compute new p_correct, ensuring it does not exceed 1.0
                            p_correct_new = p_correct + delta  # [N]
                            p_correct_new = torch.clamp(p_correct_new, max=1.0)  # [N]

                            # Compute actual delta used (in case p_correct_new was clamped)
                            actual_delta = p_correct_new - p_correct  # [N]

                            # Compute probabilities of other tokens
                            p_others = soft_probs_masked.clone()  # [N, num_classes]
                            p_others[correct_token_mask] = 0.0  # Zero out correct token probabilities

                            # Compute sum of other probabilities
                            sum_p_others = p_others.sum(dim=-1) + 1e-12  # [N]

                            # Adjust other tokens proportionally based on actual_delta
                            adjustment = (p_others / sum_p_others.unsqueeze(-1)) * actual_delta.unsqueeze(-1)  # [N, num_classes]
                            p_others = p_others - adjustment  # [N, num_classes]

                            # Ensure probabilities are non-negative
                            p_others = torch.clamp(p_others, min=0.0)  # [N, num_classes]

                            # Assign adjusted other tokens back
                            soft_probs_masked[~correct_token_mask] = p_others[~correct_token_mask]  # [N, num_classes]

                            # Set correct token's probability to p_correct_new
                            soft_probs_masked[correct_token_mask] = p_correct_new  # [N, num_classes]

                            # Assign to targets_add_K
                            targets_add_K[mask] = soft_probs_masked  # [batch_size, seq_length - 1, num_classes]

                        # Zero out positions where mask is False
                        if self.masked_with_logits:
                            # Assign original soft_probs to masked positions
                            targets_add_K = targets_add_K * mask.unsqueeze(-1).float() + soft_probs * (~mask).unsqueeze(-1).float()
                        else:
                            targets_add_K[~mask] = 0.0

                        # # Assert that probabilities sum to 1.0
                        # probs_sum = targets_add_K.sum(dim=-1)  # [batch_size, seq_length - 1]
                        # assert torch.allclose(probs_sum[mask], torch.ones_like(probs_sum[mask]), atol=1e-6), "Probabilities do not sum to 1.0"
                    
                        if not self.soft_output_logits:
                            log_probs = F.log_softmax(shift_logits, dim=-1)  # Shape: [batch_size, seq_length-1, num_classes]

                        # Define the loss function
                        if self.interpolation_loss == 'cross':
                            if self.masked_with_logits:
                                loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=self.masked_with_logits_label_smoothing)
                            elif self.label_smoothing > 0.0:
                                loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
                            else:
                                loss_fct = torch.nn.CrossEntropyLoss()
                        elif self.interpolation_loss == 'kl':
                            loss_fct = torch.nn.KLDivLoss(reduction='batchmean') #TODO: check with reduction='mean'
                        else:
                            raise ValueError(f"Unsupported interpolation loss function: {self.interpolation_loss}")

                        if self.soft_output_logits:
                            loss = loss_fct(
                                shift_logits.view(-1, num_classes),                # Predictions
                                targets_add_K.view(-1, num_classes)       # Targets
                            )
                        else:
                            loss = loss_fct(
                                log_probs.view(-1, num_classes),                # Predictions
                                targets_add_K.view(-1, num_classes)       # Targets
                            )

                        
                    elif self.set_to_one:
                        dtype = torch.float32 if self.interpolation_dtype == 'float32' else torch.bfloat16 # Default
                        shift_logits = output.logits[:, :-1, :].contiguous().to(dtype) # Shape: [batch_size, seq_length-1, num_classes]
                        valid_targets = fused_labels[:, 1:].contiguous().to(dtype)    # Shape: [batch_size, seq_length-1]

                        num_classes = shift_logits.size(-1)
                        mask = (valid_targets != -100)  # Ignored positions are marked with -100

                        if self.load_logits:
                            # Use the teacher_logits from the batch
                            target_aligned_logits = teacher_logits[:, 1:, :].contiguous().to(dtype)
                        elif self.vlm.llm_teacher is not None:
                            target_aligned_logits = teacher_output.logits[:, 1:, :].contiguous().to(dtype)
                        else:
                            target_aligned_logits = output.logits[:, 1:, :].contiguous().to(dtype)  # [batch_size, seq_length - 1, num_classes]

                        # Compute soft probabilities and detach from computational graph
                        soft_probs = F.softmax(target_aligned_logits, dim=-1).detach()  # [batch_size, seq_length - 1, num_classes]

                        # Initialize target distributions
                        targets_set_to_one = torch.zeros_like(soft_probs)  # [batch_size, seq_length - 1, num_classes]

                        if mask.any():
                            # Extract valid positions
                            valid_indices = valid_targets[mask].long()  # [N]
                            batch_indices, seq_indices = mask.nonzero(as_tuple=True)  # [N], [N]

                            # Gather soft probabilities at valid positions
                            soft_probs_masked = soft_probs[mask]  # [N, num_classes]

                            # Create a mask for the correct tokens
                            correct_token_mask = torch.zeros_like(soft_probs_masked, dtype=torch.bool)
                            correct_token_mask.scatter_(1, valid_indices.unsqueeze(1), True)  # [N, num_classes]

                            # Extract p_correct
                            p_correct = soft_probs_masked[correct_token_mask]  # [N]

                            # Compute delta (amount to set correct token to 1.0)
                            delta = 1.0 - p_correct  # [N]

                            # Compute probabilities of other tokens
                            p_others = soft_probs_masked.clone()  # [N, num_classes]
                            p_others[correct_token_mask] = 0.0  # Zero out correct token probabilities

                            # Compute sum of other probabilities
                            sum_p_others = p_others.sum(dim=-1) + 1e-12  # [N]

                            # Adjust other tokens proportionally
                            adjustment = (p_others / sum_p_others.unsqueeze(-1)) * delta.unsqueeze(-1)  # [N, num_classes]
                            p_others = p_others - adjustment  # [N, num_classes]

                            # Ensure probabilities are non-negative
                            p_others = torch.clamp(p_others, min=0.0)  # [N, num_classes]

                            # Set correct token's probability to 1.0
                            p_correct_new = torch.ones_like(p_correct)  # [N]
                            soft_probs_masked[correct_token_mask] = p_correct_new  # [N, num_classes]

                            # Assign adjusted other tokens back
                            soft_probs_masked[~correct_token_mask] = p_others[~correct_token_mask]  # [N, num_classes]

                            # Assign to target_set_to_one
                            targets_set_to_one[mask] = soft_probs_masked  # [batch_size, seq_length - 1, num_classes]

                        # Zero out positions where mask is False
                        if self.masked_with_logits:
                            # Assign original soft_probs to masked positions
                            targets_set_to_one = targets_set_to_one * mask.unsqueeze(-1).float() + soft_probs * (~mask).unsqueeze(-1).float()
                        else:
                            targets_set_to_one[~mask] = 0.0

                        # # Assert that probabilities sum to 1.0
                        # probs_sum = targets_set_to_one.sum(dim=-1)  # [batch_size, seq_length - 1]
                        # assert torch.allclose(probs_sum[mask], torch.ones_like(probs_sum[mask]), atol=1e-6), "Probabilities do not sum to 1.0"

                        if not self.soft_output_logits:
                            log_probs = F.log_softmax(shift_logits, dim=-1)

                        # Define the loss function
                        if self.interpolation_loss == 'cross':
                            # if self.masked_with_logits:
                            #     loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=self.masked_with_logits_label_smoothing)
                            if self.label_smoothing > 0.0:
                                loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
                            elif self.masked_with_logits:
                                loss_fct = torch.nn.CrossEntropyLoss()
                            else:
                                loss_fct = torch.nn.CrossEntropyLoss()
                        elif self.interpolation_loss == 'kl':
                            loss_fct = torch.nn.KLDivLoss(reduction='batchmean') #TODO: check with reduction='mean'
                        else:
                            raise ValueError(f"Unsupported interpolation loss function: {self.interpolation_loss}")

                        if self.soft_output_logits:
                            loss = loss_fct(
                                shift_logits.view(-1, num_classes),                # Predictions
                                targets_set_to_one.view(-1, num_classes)       # Targets
                            )
                        else:
                            loss = loss_fct(
                                log_probs.view(-1, num_classes),                # Predictions
                                targets_set_to_one.view(-1, num_classes)       # Targets
                            )
                    
                    elif self.max_logit:

                        dtype = torch.float32 if self.interpolation_dtype == 'float32' else torch.bfloat16 # Default
                        shift_logits = output.logits[:, :-1, :].contiguous().to(dtype) # Shape: [batch_size, seq_length-1, num_classes]
                        valid_targets = fused_labels[:, 1:].contiguous().to(dtype)    # Shape: [batch_size, seq_length-1]

                        num_classes = shift_logits.size(-1)
                        mask = (valid_targets != -100)  # Ignored positions are marked with -100
                        if self.load_logits:
                            # Use the teacher_logits from the batch
                            target_aligned_logits = teacher_logits[:, 1:, :].contiguous().to(dtype)
                        elif self.vlm.llm_teacher is not None:
                            target_aligned_logits = teacher_output.logits[:, 1:, :].contiguous().to(dtype)
                        else:
                            target_aligned_logits = output.logits[:, 1:, :].contiguous().to(dtype)  # [batch_size, seq_length - 1, num_classes]

                        # Compute soft probabilities and detach from computational graph
                        soft_probs = F.softmax(target_aligned_logits, dim=-1).detach()  # [batch_size, seq_length - 1, num_classes]
                        # Initialize target distributions
                        targets_max_logit = torch.zeros_like(soft_probs)  # [batch_size, seq_length - 1, num_classes]

                        if mask.any():
                            # Extract valid positions
                            valid_indices = valid_targets[mask].long()  # [N]
                            batch_indices, seq_indices = mask.nonzero(as_tuple=True)  # [N], [N]

                            # Gather soft probabilities at valid positions
                            soft_probs_masked = soft_probs[mask]  # [N, num_classes]

                            # Find p_max for each valid position
                            p_max, _ = soft_probs_masked.max(dim=-1)  # [N]

                            # Create a mask for the correct tokens
                            correct_token_mask = torch.zeros_like(soft_probs_masked, dtype=torch.bool)
                            correct_token_mask.scatter_(1, valid_indices.unsqueeze(1), True)  # [N, num_classes]

                            # Extract p_correct
                            p_correct = soft_probs_masked[correct_token_mask]  # [N]

                            # Compute delta
                            delta = p_max - p_correct  # [N]

                            # Handle cases where p_correct is already the max (delta = 0)
                            # No adjustment needed in these cases

                            # Compute probabilities of other tokens
                            p_others = soft_probs_masked.clone()  # [N, num_classes]
                            p_others[correct_token_mask] = 0.0  # Zero out correct token probabilities

                            # Compute sum of other probabilities
                            sum_p_others = p_others.sum(dim=-1)  # [N]

                            # Avoid division by zero by adding epsilon where sum_p_others is zero
                            epsilon = 1e-12
                            sum_p_others = sum_p_others + epsilon  # [N]

                            # Compute adjustment: subtract delta proportionally from other tokens
                            adjustment = (p_others / sum_p_others.unsqueeze(-1)) * delta.unsqueeze(-1)  # [N, num_classes]
                            p_others = p_others - adjustment  # [N, num_classes]

                            # Ensure probabilities are non-negative
                            p_others = torch.clamp(p_others, min=0.0)  # [N, num_classes]

                            # Set correct token's probability to p_max
                            p_correct_new = p_max  # [N]
                            soft_probs_masked[correct_token_mask] = p_correct_new  # [N, num_classes]

                            # Assign adjusted other tokens back
                            soft_probs_masked[~correct_token_mask] = p_others[~correct_token_mask]  # [N, num_classes]

                            # Assign to targets_max_logit
                            targets_max_logit[mask] = soft_probs_masked  # [batch_size, seq_length - 1, num_classes]

                            # Debugging: Check if any probabilities are negative (they should not be)
                            if (targets_max_logit < 0).any():
                                negative_probs = targets_max_logit < 0
                                print("Negative probabilities detected in max_logit function.")
                                print(targets_max_logit[negative_probs])
                                raise ValueError("Negative probabilities found after adjustment in max_logit.")

                            # Debugging: Check sum of probabilities
                            probs_sum = targets_max_logit.sum(dim=-1)  # [batch_size, seq_length - 1]
                            if not torch.allclose(probs_sum[mask], torch.ones_like(probs_sum[mask]), atol=1e-6):
                                max_diff = torch.abs(probs_sum[mask] - 1.0).max()
                                print(f"Max difference from 1.0: {max_diff.item()}")
                                raise AssertionError("Probabilities do not sum to 1.0 after max_logit adjustments.")

                        # Zero out positions where mask is False
                        if self.masked_with_logits:
                            # Assign original soft_probs to masked positions
                            targets_max_logit = targets_max_logit * mask.unsqueeze(-1).float() + soft_probs * (~mask).unsqueeze(-1).float()
                        else:
                            targets_max_logit[~mask] = 0.0
                        # # Zero out positions where mask is False
                        # targets_max_logit[~mask] = 0.0

                        if not self.soft_output_logits:
                            log_probs = F.log_softmax(shift_logits, dim=-1)

                        # Define the loss function
                        if self.interpolation_loss == 'cross':
                            if self.masked_with_logits:
                                loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=self.masked_with_logits_label_smoothing)
                            elif self.label_smoothing > 0.0:
                                loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
                            else:
                                loss_fct = torch.nn.CrossEntropyLoss()
                        elif self.interpolation_loss == 'kl':
                            loss_fct = torch.nn.KLDivLoss(reduction='batchmean') #TODO: check with reduction='mean'
                        else:
                            raise ValueError(f"Unsupported interpolation loss function: {self.interpolation_loss}")

                        if self.soft_output_logits:
                            loss = loss_fct(
                                shift_logits.view(-1, num_classes),                # Predictions
                                targets_max_logit.view(-1, num_classes)       # Targets
                            )
                        else:
                            loss = loss_fct(
                                log_probs.view(-1, num_classes),                # Predictions
                                targets_max_logit.view(-1, num_classes)       # Targets
                            )

                    else:
                        if self.align_loss:
                            output, alignment_loss_val = output
                        
                        loss = output.loss

                        if (self.align_loss) and (alignment_loss_val is not None):
                            loss = loss + self.align_weight * alignment_loss_val

                        # Add norm-based regularization if enabled
                        if self.norm_reg:
                            # Compute projected visual embeddings
                            vis_emb_for_norm, _ = self.vlm.get_embeddings(
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                pixel_values=batch["pixel_values"],
                                labels=batch["labels"],
                                multimodal_indices=batch["multimodal_indices"],
                            )
                            # Compute L2 norm of projected visual embeddings
                            vis_emb_norm = torch.norm(vis_emb_for_norm, p=2, dim=-1)  # Shape: [bsz, num_patches]
                            # Regularization loss: mean of squared L2 norms
                            reg_loss = (vis_emb_norm ** 2).mean()
                            # Add to the main loss
                            loss = loss + self.norm_reg_weight * reg_loss

                    # === Metric Calculation ===
                    metrics_kwargs = {"loss": loss} # Start with the main loss

                    # --- Layer-wise Stats --- (moved before other embedding metrics)
                    if self.track_layer_stats and hidden_states:
                        with torch.no_grad():
                            # Check if vis_indices indicate a multimodal batch
                            is_multimodal_batch = vis_indices != (0, 0)
                            if is_multimodal_batch:
                                vis_start_idx, vis_end_idx = vis_indices
                            else:
                                # For text-only batches, set indices such that slicing yields empty tensors
                                vis_start_idx, vis_end_idx = 1, 1

                            # Iterate through hidden states (skip embedding layer output at index 0)
                            for layer_idx, layer_hidden_state in enumerate(hidden_states[1:]):
                                # layer_hidden_state shape: [bsz, seq_len, hidden_dim]
                                vis_layer_emb = None
                                txt_layer_emb = None

                                # Extract and compute visual embeddings stats only if it's a multimodal batch
                                if is_multimodal_batch:
                                    if "multimodal_indices" in batch and batch["multimodal_indices"] is not None and len(batch["multimodal_indices"]) > 0:
                                        vis_layer_emb = layer_hidden_state[batch["multimodal_indices"], vis_start_idx:vis_end_idx, :]
                                        if vis_layer_emb.numel() > 0:
                                            vis_layer_stats = compute_embedding_stats(vis_layer_emb)
                                            metrics_kwargs[f"Layer_{layer_idx}/VisEmbedMean"] = vis_layer_stats["embed_mean"]
                                            metrics_kwargs[f"Layer_{layer_idx}/VisEmbedStd"] = vis_layer_stats["embed_std"]
                                            metrics_kwargs[f"Layer_{layer_idx}/VisL2Mean"] = vis_layer_stats["l2_mean"]
                                            metrics_kwargs[f"Layer_{layer_idx}/VisL2Std"] = vis_layer_stats["l2_std"]
                                        else:
                                            vis_layer_emb = None # Ensure it's None if empty

                                # Extract text embeddings (all tokens *excluding* visual tokens if multimodal)
                                txt_mask = torch.ones(layer_hidden_state.shape[1], dtype=torch.bool, device=layer_hidden_state.device)
                                if is_multimodal_batch:
                                    txt_mask[vis_start_idx:vis_end_idx] = False
                                
                                txt_layer_emb = layer_hidden_state[:, txt_mask, :]
                                if txt_layer_emb.numel() > 0:
                                    txt_layer_stats = compute_embedding_stats(txt_layer_emb)
                                    metrics_kwargs[f"Layer_{layer_idx}/TxtEmbedMean"] = txt_layer_stats["embed_mean"]
                                    metrics_kwargs[f"Layer_{layer_idx}/TxtEmbedStd"] = txt_layer_stats["embed_std"]
                                    metrics_kwargs[f"Layer_{layer_idx}/TxtL2Mean"] = txt_layer_stats["l2_mean"]
                                    metrics_kwargs[f"Layer_{layer_idx}/TxtL2Std"] = txt_layer_stats["l2_std"]
                                else:
                                     txt_layer_emb = None # Ensure it's None if empty

                                # <<< ADDED >>> Compute layer-wise cosine similarity
                                if vis_layer_emb is not None and txt_layer_emb is not None:
                                    # <<< MODIFIED >>> Check the dedicated flag
                                    if self.track_cosine_layer_stats:
                                        # Flatten and cast
                                        vis_flat = vis_layer_emb.reshape(-1, vis_layer_emb.shape[-1]).to(torch.float32)
                                        txt_flat = txt_layer_emb.reshape(-1, txt_layer_emb.shape[-1]).to(torch.float32)

                                        # Normalize each row so that dot product becomes cosine similarity
                                        vis_norm = F.normalize(vis_flat, p=2, dim=-1)
                                        txt_norm = F.normalize(txt_flat, p=2, dim=-1)

                                        # Cosine similarity matrix: (vis_count, txt_count)
                                        # Ensure matrix multiplication is possible
                                        if vis_norm.shape[0] > 0 and txt_norm.shape[0] > 0:
                                            cos_sim = torch.matmul(vis_norm, txt_norm.transpose(0, 1))

                                            # Grab min, max, and mean over all pairwise cos-sims
                                            metrics_kwargs[f"Layer_{layer_idx}/VisTxtCosineMin"] = cos_sim.min().item()
                                            metrics_kwargs[f"Layer_{layer_idx}/VisTxtCosineMax"] = cos_sim.max().item()
                                            metrics_kwargs[f"Layer_{layer_idx}/VisTxtCosineMean"] = cos_sim.mean().item()
                                        else:
                                            # Default values if one set is empty after masking/slicing
                                            metrics_kwargs[f"Layer_{layer_idx}/VisTxtCosineMin"] = 0.0
                                            metrics_kwargs[f"Layer_{layer_idx}/VisTxtCosineMax"] = 0.0
                                            metrics_kwargs[f"Layer_{layer_idx}/VisTxtCosineMean"] = 0.0

                                # <<< ADDED >>> Activation Distribution Stats Flag
                                if self.track_activation_distributions:
                                    # Visual token activations
                                    if vis_layer_emb is not None:
                                        vis_dist_stats = compute_activation_distribution(vis_layer_emb)
                                        for stat_k, stat_v in vis_dist_stats.items():
                                            # Separate histogram to avoid commit deque handling
                                            if stat_k == "histogram":
                                                metrics.log(metrics.global_step, {
                                                    f"Layer_{layer_idx}/VisDist/histogram": stat_v
                                                })
                                            else:
                                                metrics_kwargs[f"Layer_{layer_idx}/VisDist/{stat_k}"] = stat_v

                                    # Text token activations
                                    if txt_layer_emb is not None:
                                        txt_dist_stats = compute_activation_distribution(txt_layer_emb)
                                        for stat_k, stat_v in txt_dist_stats.items():
                                            if stat_k == "histogram":
                                                metrics.log(metrics.global_step, {
                                                    f"Layer_{layer_idx}/TxtDist/histogram": stat_v
                                                })
                                            else:
                                                metrics_kwargs[f"Layer_{layer_idx}/TxtDist/{stat_k}"] = stat_v

                    # --- Other Embedding Metrics --- #
                    if self.track_embeddings or self.track_covariance or self.norm_reg:
                        # Get alignment_loss_val if it's available
                        alignment_loss_val_for_metrics = alignment_loss_val if self.align_loss else None
                        reg_loss_for_metrics = reg_loss if self.norm_reg else None

                        # Compute other embedding metrics and update kwargs
                        other_metrics = self._compute_embedding_metrics(vis_emb, txt_emb, alignment_loss_val_for_metrics, reg_loss_for_metrics)
                        metrics_kwargs.update(other_metrics)

                    # If align_loss computed but not tracked via other means, add it explicitly
                    elif self.align_loss and alignment_loss_val is not None:
                        metrics_kwargs["alignment_loss"] = alignment_loss_val

                    # Commit all gathered metrics
                    metrics.commit(**metrics_kwargs)

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

                    # # Go through the modules of vlm.model.llm_backbone, and print which is trainable
                    # for name, module in self.vlm.llm_backbone.named_modules():
                    #     # if the module is the embed_in or embed_out layer
                    #     if 'embed_in' in name or 'embed_out' in name:
                    #         overwatch.info(f"{name}, {module.weight.shape}")
                    #         # print if grad is set to true
                    #         overwatch.info(f"{name}, {module.weight.requires_grad}")
                            
                    # exit()

                    # Step =>> Only if Done w/ Gradient Accumulation
                    if (train_idx + 1) % self.grad_accumulation_steps == 0:
                        metrics.commit(update_step_time=True)
                        # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                        self.clip_grad_norm()

                        # Track validation metrics 
                        should_validate = (
                            (self.enable_validation_tracking and metrics.global_step % self.validation_frequency == 0) or
                            (self.track_avg_rank and not self.enable_validation_tracking)
                        )
                        
                        if should_validate:
                            if self.enable_validation_tracking and self.validation_manager:
                                # Use new validation set approach
                                validation_metrics = self._compute_validation_metrics()
                                if validation_metrics:
                                    metrics.log(metrics.global_step, validation_metrics)
                            elif self.track_avg_rank:
                                # Legacy single-batch approach
                                with torch.no_grad():
                                    # Compute for align batch
                                    output_align, fused_labels_align = self.vlm(
                                        input_ids=batch_align["input_ids"],
                                        attention_mask=batch_align["attention_mask"],
                                        pixel_values=batch_align["pixel_values"],
                                        labels=batch_align["labels"],
                                        multimodal_indices=batch_align["multimodal_indices"],
                                        return_labels=True,
                                    )
                                    avg_rank_align, min_rank_align, max_rank_align = calculate_average_rank(output_align, fused_labels_align)
                                    # Compute for finetune batch
                                    output_finetune, fused_labels_finetune = self.vlm(
                                        input_ids=batch_finetune["input_ids"],
                                        attention_mask=batch_finetune["attention_mask"],
                                        pixel_values=batch_finetune["pixel_values"],
                                        labels=batch_finetune["labels"],
                                        multimodal_indices=batch_finetune["multimodal_indices"],
                                        return_labels=True,
                                    )
                                    avg_rank_finetune, min_rank_finetune, max_rank_finetune = calculate_average_rank(output_finetune, fused_labels_finetune)

                                    # Log metrics with specific names
                                    metrics.log(metrics.global_step, {"avg_rank_align": avg_rank_align})
                                    metrics.log(metrics.global_step, {"avg_rank_finetune": avg_rank_finetune})
                                    metrics.log(metrics.global_step, {"min_rank_align": min_rank_align})
                                    metrics.log(metrics.global_step, {"min_rank_finetune": min_rank_finetune})
                                    metrics.log(metrics.global_step, {"max_rank_align": max_rank_align})
                                    metrics.log(metrics.global_step, {"max_rank_finetune": max_rank_finetune})

                        # Optimizer & LR Scheduler Step
                        if self.lr_scheduler_type == 'schedule-free':
                            self.optimizer.step() 
                        else:
                            self.optimizer.step()
                            self.lr_scheduler.step()
                        self.optimizer.zero_grad()
                        if self.measure_rank_entropy:
                            with torch.no_grad():
                                rank_entropy = calculate_rank_entropy(output, fused_labels)
                                metrics.commit(global_step=metrics.global_step + 1, rank_entropy=rank_entropy)
                        
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
                                weight_change_wrt_last_per_layer = measure_lora_weight_change_per_layer(model, last_weights, lora_layers)
                                weight_change_wrt_initial_per_layer = measure_lora_weight_change_per_layer(model, initial_weights, lora_layers)

                                last_weights = capture_initial_weights(model, lora_layers)                                
                                metrics.commit(global_step=metrics.global_step + 1, \
                                lora_plasticity = weight_change_wrt_last)
                                metrics.commit(global_step=metrics.global_step + 1, \
                                    lora_plasticity_first = weight_change_wrt_initial)
                                metrics.commit(global_step=metrics.global_step + 1, \
                                    lora_weight_changes = weight_change_wrt_last_per_layer)
                                metrics.commit(global_step=metrics.global_step + 1, \
                                    lora_weight_changes_first = weight_change_wrt_initial_per_layer)

                        elif self.track_ft_plasticity and ((train_idx + 1) % self.compare_plasticity_steps == 0):
                            if isinstance(self.vlm, torch.nn.parallel.DistributedDataParallel):
                                model = self.vlm.module.llm_backbone.llm
                            elif self.vlm.llm_backbone.__class__.__name__ == 'DistributedDataParallel':
                                model = self.vlm.llm_backbone.module.llm
                            elif self.vlm.__class__.__name__ == 'FullyShardedDataParallel':
                                model = self.vlm.llm_backbone.llm
                            else:
                                raise ValueError(f"No accepted vlm class name in Full FT plasticity tracking.")

                            # Capture weights if it's the first iteration or if last_weights is not initialized
                            if last_weights is None:
                                last_weights = capture_initial_weights(model)
                            
                            else:
                                # Measure weight changes compared to last captured weights
                                weight_changes = log_weight_change_detailed(model, last_weights)
                                
                                # Update last_weights for the next iteration
                                last_weights = capture_initial_weights(model)

                                # Log weight changes using the metrics class
                                metrics.commit(
                                    global_step=metrics.global_step + 1,
                                    ft_total_weight_change=weight_changes['total'],  # Aggregate total average weight change
                                    ft_layer_weight_changes=weight_changes['layers'],  # Per-layer weight changes (layer, attention, mlp)
                                    ft_parameter_weight_changes=weight_changes['parameters'],  # Per-named parameter weight change
                                )

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
                if self.save_logits:
                    self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, None)
                else:
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

def calculate_rank_entropy(output, fused_labels):
    """
    Calculates the entropy of the distribution of the ranks of correct tokens within the logits.

    Args:
        output (torch.Tensor): The output from the model, containing logits of shape (batch_size, seq_len, vocab_size).
        fused_labels (torch.Tensor): The ground truth labels aligned with the logits of shape (batch_size, seq_len).

    Returns:
        float: The entropy of the rank distribution.
    """
    # Ensure logits and labels are on the same device
    device = output.logits.device
    logits = output.logits
    labels = fused_labels

    # Align logits and labels by shifting
    # Typically, for language modeling, logits are predictions for the next token
    shift_logits = logits[:, :-1, :].contiguous()  # Shape: (batch_size, seq_len - 1, vocab_size)
    valid_targets = labels[:, 1:].contiguous()     # Shape: (batch_size, seq_len - 1)

    # Create mask for valid targets (labels != IGNORE_INDEX)
    mask = (valid_targets != IGNORE_INDEX)  # Shape: (batch_size, seq_len - 1)

    # Replace -100 in labels to avoid indexing errors (temporary placeholder)
    valid_targets_clamped = valid_targets.clone()
    valid_targets_clamped[~mask] = 0  # Any value, since these positions will be masked out later

    # Reshape for easier processing
    batch_size, seq_len_minus_one, vocab_size = shift_logits.size()

    # Flatten the batch and sequence dimensions
    flat_logits = shift_logits.view(-1, vocab_size)           # Shape: (batch_size * (seq_len -1), vocab_size)
    flat_targets = valid_targets_clamped.view(-1)            # Shape: (batch_size * (seq_len -1))
    flat_mask = mask.view(-1)                                # Shape: (batch_size * (seq_len -1))

    # Filter out invalid positions
    valid_logits = flat_logits[flat_mask]                    # Shape: (num_valid, vocab_size)
    valid_targets_final = flat_targets[flat_mask]            # Shape: (num_valid)

    if valid_logits.numel() == 0:
        # If there are no valid logits, return entropy as zero
        return 0.0

    # Compute the ranks of the correct tokens
    # Argsort in descending order to get ranks (highest logit has rank 1)
    sorted_indices = torch.argsort(valid_logits, dim=-1, descending=True)  # Shape: (num_valid, vocab_size)

    # Create a mask where the sorted indices match the target indices
    target_mask = sorted_indices == valid_targets_final.unsqueeze(1)      # Shape: (num_valid, vocab_size)

    # The rank is the index where the target_mask is True, plus 1 (1-based ranking)
    # Since there's exactly one True per row, we can use nonzero and gather
    ranks = torch.nonzero(target_mask, as_tuple=False)[:, 1] + 1        # Shape: (num_valid)

    # Cap the ranks at 1000 to avoid excessively large bins
    ranks_capped = ranks.clamp(max=1000)

    # Determine the maximum rank after capping to define the number of bins
    max_rank = ranks_capped.max()

    # Adjust ranks to start from 0 for bincount
    ranks_adjusted = ranks_capped - 1  # Now ranks start at 0

    # Compute bincount with a minimum length to include all ranks
    # Ensure max_rank is a scalar by calling .item()
    rank_counts = torch.bincount(ranks_adjusted, minlength=max_rank.item())

    # Convert counts to probabilities
    rank_probs = rank_counts.float() / rank_counts.sum()

    # Calculate entropy
    # Add a small epsilon to avoid log(0)
    epsilon = 1e-12
    entropy = -torch.sum(rank_probs * torch.log(rank_probs + epsilon)).item()

    return entropy


# *** UTILITY to compute embedding stats: mean, std, l2-mean, l2-std
def compute_embedding_stats(emb: torch.Tensor) -> dict:
    """
    Given emb shape [B, N, D] or [N, D],
    returns dict with mean, std of values, plus mean, std of L2 norms.
    """
    # flatten to 2D (N, D)
    e2d = emb.reshape(-1, emb.shape[-1])  # [total_vectors, D]
    # coordinate-wise mean, std
    mean_val = e2d.mean().item()
    std_val = e2d.std().item()
    # l2 norm per row
    norms = torch.norm(e2d, p=2, dim=-1)
    mean_norm = norms.mean().item()
    std_norm = norms.std().item()
    return {
        "embed_mean": mean_val,
        "embed_std": std_val,
        "l2_mean": mean_norm,
        "l2_std": std_norm
    }

def calculate_top_n_coverage(output, fused_labels, top_n=1):
    """
    Calculates the cumulative top-n coverage (percentage of correct tokens within top-n predictions).

    Args:
        output: Model output containing logits (batch_size, seq_len, vocab_size).
        fused_labels: Ground truth labels (batch_size, seq_len).
        top_n: Number of top predictions to consider (default: 10).

    Returns:
        float: Percentage of correct tokens within the top-n predictions.
    """
    device = output.logits.device
    logits = output.logits
    labels = fused_labels

    # Align logits and labels by shifting
    shift_logits = logits[:, :-1, :].contiguous()  # (batch_size, seq_len-1, vocab_size)
    valid_targets = labels[:, 1:].contiguous()     # (batch_size, seq_len-1)

    # Create mask for valid targets (labels != -100)
    mask = (valid_targets != -100)

    # Replace -100 to avoid indexing errors
    valid_targets_clamped = valid_targets.clone()
    valid_targets_clamped[~mask] = 0

    # Flatten tensors
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))  # (batch_size*(seq_len-1), vocab_size)
    flat_targets = valid_targets_clamped.view(-1)               # (batch_size*(seq_len-1))
    flat_mask = mask.view(-1)                                   # (batch_size*(seq_len-1))

    # Filter valid positions
    valid_logits = flat_logits[flat_mask]                       # (num_valid, vocab_size)
    valid_targets_final = flat_targets[flat_mask]               # (num_valid)

    # Sort logits and find ranks
    sorted_indices = torch.argsort(valid_logits, dim=-1, descending=True)
    target_mask = (sorted_indices == valid_targets_final.unsqueeze(1))
    ranks = torch.nonzero(target_mask, as_tuple=False)[:, 1] + 1

    # Calculate coverage
    ranks_np = ranks.cpu().numpy()
    top_n_coverage = np.mean(ranks_np <= top_n) * 100.0 if len(ranks_np) > 0 else 0.0
    return top_n_coverage


def calculate_average_rank(output, fused_labels):
    """
    Calculates the average 1-based rank of the correct token within the model's top predictions.
    
    Args:
        output (CausalLMOutputWithPast): Model output containing `.logits` of shape (batch_size, seq_len, vocab_size).
        fused_labels (torch.Tensor): Ground truth labels (batch_size, seq_len).

    Returns:
        float: The average rank of the correct token over all valid positions. 
               (Rank 1 means the token was the top prediction, Rank 2 means it was second, etc.)
               If there are no valid positions, returns 0.0.
    """
    device = output.logits.device
    logits = output.logits
    labels = fused_labels

    # Align logits and labels by shifting. 
    # Typically, for language modeling, logits[:, t, :] is the prediction for labels[:, t+1].
    shift_logits = logits[:, :-1, :].contiguous()  # (batch_size, seq_len-1, vocab_size)
    valid_targets = labels[:, 1:].contiguous()     # (batch_size, seq_len-1)

    # Create mask for valid targets (labels != -100)
    mask = (valid_targets != -100)

    # Replace -100 to avoid indexing errors
    valid_targets_clamped = valid_targets.clone()
    valid_targets_clamped[~mask] = 0

    # Flatten tensors
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))  # (batch_size*(seq_len-1), vocab_size)
    flat_targets = valid_targets_clamped.view(-1)               # (batch_size*(seq_len-1))
    flat_mask = mask.view(-1)                                   # (batch_size*(seq_len-1))

    # Filter valid positions
    valid_logits = flat_logits[flat_mask]                       # (num_valid, vocab_size)
    valid_targets_final = flat_targets[flat_mask]               # (num_valid)

    # If there are no valid positions (e.g., all were -100), return 0.0
    if valid_logits.numel() == 0:
        return 0.0

    # Sort logits to find the rank of the correct token 
    # (descending => highest logit has rank 1)
    sorted_indices = torch.argsort(valid_logits, dim=-1, descending=True)  
    # Create a mask where the sorted indices match the target indices
    target_mask = (sorted_indices == valid_targets_final.unsqueeze(1))  
    # The rank is the index where the target_mask is True, plus 1 (1-based)
    ranks = torch.nonzero(target_mask, as_tuple=False)[:, 1] + 1  # shape: (num_valid,)

    # Compute the average rank
    avg_rank = ranks.float().mean().item()
    min_rank = ranks.float().min().item()
    max_rank = ranks.float().max().item()
    return avg_rank, min_rank, max_rank

# *** UTILITY to compute detailed activation distribution stats (mean/std/min/max/percentiles/skew/kurtosis)
#     This mirrors the statistics computed in the reference ActivationDistributionTracker but omits the histogram for
#     JSON-friendly logging. Tensor is expected to be 2-D or 3-D; it will be flattened prior to processing.
def compute_activation_distribution(tensor: torch.Tensor) -> Dict[str, float]:
    """Return rich distribution statistics for the given tensor (flattened)."""
    if tensor.numel() == 0:
        return {}

    # Convert to float32 for numerical stability & numpy compatibility
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()

    flat = tensor.view(-1).detach().cpu().numpy()

    # Basic moments
    mean = float(np.mean(flat))
    std = float(np.std(flat))
    data_min = float(np.min(flat))
    data_max = float(np.max(flat))
    median = float(np.median(flat))

    # Percentiles
    q25 = float(np.percentile(flat, 25))
    q75 = float(np.percentile(flat, 75))
    q05 = float(np.percentile(flat, 5))
    q95 = float(np.percentile(flat, 95))

    # Higher-order statistics
    if std == 0:
        skewness, kurtosis = 0.0, 0.0
    else:
        normed = (flat - mean) / std
        skewness = float(np.mean(normed ** 3))
        kurtosis = float(np.mean(normed ** 4) - 3.0)

    return {
        "mean": mean,
        "std": std,
        "min": data_min,
        "max": data_max,
        "median": median,
        "q25": q25,
        "q75": q75,
        "q05": q05,
        "q95": q95,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "num_samples": len(flat),
        # Add histogram for WandB visualization
        "histogram": wandb.Histogram(np_histogram=(hist, bin_edges)),
    }