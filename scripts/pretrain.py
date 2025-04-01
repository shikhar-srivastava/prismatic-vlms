"""
pretrain.py

Pretraining script for Prismatic VLM pretraining in native PyTorch, using Fully-Sharded Data Parallel (FSDP) to run
distributed training across GPUs. By default, assumes that CUDA toolkit is >= 11.0 (to support BF16 mixed precision).


Notes & Prerequisites:
    - We're loading LLaMa-2 (and possibly other) gated models from HuggingFace (HF Hub); these require an auth_token.
      For LLaMa-2, make sure to first get Meta approval, then fill out the form at the top of the HF LLaMa-2 page:
        => Link: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
        => Generate Token (from `huggingface.co`): Settings / Access Tokens / New "Read" Token
        => Set `cfg.hf_token` to file path with token (as single line text file) or environment variable name

    - If you want to set a custom location for all HF / TIMM artifacts --> `export HF_HOME="<PATH>"` *before* running!
        => For example (add to end of .bashrc): `export HF_HOME="/mnt/fsx/skaramcheti/cache"`

Run with:
    - [Single Node One-GPU (Debug)] : torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/pretrain.py
    - [Single Node Multi-GPU (= $K)]: torchrun --standalone --nnodes 1 --nproc-per-node $K scripts/pretrain.py
    - [Multi-Node/AWS Sagemaker] Depends on your individual setup; file an issue if you have trouble!
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import draccus
import torch
import torch.distributed as dist
import yaml

from prismatic.conf import DatasetConfig, DatasetRegistry, ModelConfig, ModelRegistry
from prismatic.models import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform, get_vlm
from prismatic.overwatch import initialize_overwatch
from prismatic.preprocessing import get_dataset_and_collator
from prismatic.training import Metrics, get_train_strategy
from prismatic.util import set_global_seed

# Disable Tokenizers Parallelism to Play Nice w/ PyTorch Multiprocessing DataLoaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Added by Shikhar
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Initialize Overwatch =>> Wraps `logging.Logger`
os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'  # not to enforce timeout
os.environ['NCCL_BLOCKING_WAIT'] = '1'

overwatch = initialize_overwatch(__name__)


@dataclass
class PretrainConfig:
    # fmt: off

    # ModelConfig (`prismatic/conf/models.py`); override with --model.type `ModelRegistry.<MODEL>.model_id`
    model: ModelConfig = field(
        default_factory=ModelConfig.get_choice_class(ModelRegistry.PRISM_DINOSIGLIP_7B.model_id)
    )

    # DatasetConfig (`prismatic/conf/datasets.py`); override with --dataset.type `DatasetRegistry.<DATASET>.dataset_id`
    dataset: DatasetConfig = field(
        default_factory=DatasetConfig.get_choice_class(DatasetRegistry.LLAVA_V15.dataset_id)
    )

    # Pretraining Stage in < align (projector-only) | finetune (projector + LLM) | full-finetune (all) >
    # ---
    stage: str = "finetune"                                         # Pretraining Stage in < align | finetune >
    pretrained_checkpoint: Optional[Path] = None                    # Pretrained Checkpoint to Load (for `finetune`)
                                                                    #   if None =>> will match on (run_dir / `align`)

    # Run Arguments
    run_id: Optional[str] = None                                    # Run ID for logging, Weights & Biases
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    seed: int = 7                                                   # Random seed (for reproducibility)

    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = Path(".hf_token")                  # Environment variable or Path to HF Token

    # Tracking Parameters
    trackers: Tuple[str, ...] = ("jsonl", "wandb")                  # Trackers to initialize (if W&B, add config!)
    # wandb_project: str = "prismatic"                                # Name of W&B project (default: `prismatic`)
    # wandb_entity: Optional[str] = None                              # Name of W&B entity (default: None)
    wandb_project: str = "prismatic"
    wandb_entity: str = "klab-shikhar"

    # Mitigation method. Default is None
    mitigation: str = None
    soft_alpha: float = None
    soft_alpha_masked_interpolation: float = None
    soft_output_logits: bool = True
    
    add_K: float = None
    add_K_percentage: bool = False
    set_to_one: bool = False
    max_logit: bool = False
    use_logits_in_max_logit: bool = False

    interpolation_dtype : str = 'float32'
    interpolation_loss: str = 'cross' # or 'kl'
    masked_with_logits: bool = False
    label_smoothing: float = 0.0
    masked_with_logits_label_smoothing: float = 0.01
    masked_with_logits_mask_weight: float = 0.01
    masked_with_logits_lr_increase_factor = 100.0
    olf: bool = False  # Last Transformer Block freezing
    oolf: bool = False # Last Output Layer freezing

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 8
    lora_target_modules: Union[list, str] = 'all-linear' #["q_proj", "v_proj","down_proj"]  #
    reduce_lora_rank_by_factor_of_fullrank: int = 2
    lora_use_2r_heuristic: bool = False
    lora_dropout: float = 0.05

    use_rslora: bool = True
    half_batch_size: bool = False   
    ## LoRA Merging related
    merges_after_steps: int = 0 # 0 means no merging. 1 means merging after every epoch
    ### Resetting lr/optimizer after merging. If non zero, then LR resets to 0 with quick warmup_ratio back to cosine LR after merging.
    merging_lr_warmup_steps: int = 100
    # General Training Parameters
    load_8bit: bool = False
    bigger_batch: bool = False
    hot_fix: int = 0
    epoch_count: int = 1
    ddp: bool = False

    measure_rank_entropy: bool = False

    # Schedulers and Optimizers
    schedule_free : bool = False
    infinite_schedule : bool = False
    constant_lr_with_warmup: bool = False
    constant_lr: bool = False

    track_lora_plasticity : bool = False
    track_ft_plasticity : bool = False

    compare_plasticity_steps: int = 100
    first_lora_after_warmup: bool = False

    # Save logits only
    save_logits: bool = False
    save_logits_dir: str = None
    load_logits: bool = False
    load_logits_dir: str = None

    # Teacher LLM
    llm_teacher_checkpoint: str = None
    stableadam: bool = False
    # Projector Type
    projector_type: str = None

    # Init Projector 
    init_projector: str = None # "ledoitwolf", "ledoitwolf-mlp"
    init_projector_path: str = None

    # Visual embedding scaling 
    scale_patch_embeddings: bool = False # If true, then scale by 1/sqrt(d_model) before projecting 
    pre_projection_layer_norm: bool = False

    # Alignment loss
    align_weight: float = 0.01
    align_loss: bool = False

    # Norm-based regularization
    norm_reg: bool = False
    norm_reg_weight: float = 0.01

    # <<< ADDED >>> Flag to control layer-wise statistics tracking
    track_layer_stats: bool = False

    # <<< ADDED >>> Flag to control layer-wise cosine similarity tracking
    track_cosine_layer_stats: bool = False

    track_embeddings: bool = False
    track_embeddings_histogram: bool = False
    track_embeddings_values: bool = False
    track_covariance: bool = False
    use_precomputed_covariance: bool = False
    precomputed_covariance_path: str = "/home/aac/ssrivas9/prismatic-vlms/text_covariance_186K.pt"
    track_avg_rank: bool = False

    # Mixed precision training
    disable_mixed_precision: bool = False

    # <<< ADDED >>> Flag to replace real images with random noise
    random_image: bool = False

    def __post_init__(self) -> None:
        """Set optimization parameters based on `stage` in {"align", "finetune"}."""
        # assert that load_logits and save_logits are not both true
        assert not (self.load_logits and self.save_logits), "Both load_logits and save_logits cannot be true"
        assert not (self.lora_use_2r_heuristic and self.use_rslora), "Both use_rslora and self.lora_use_2r_heuristic cannot be true"

        # Assert that self.init_projector must be None or 'ledoitwolf' 
        assert self.init_projector is None or self.init_projector == 'ledoitwolf' or self.init_projector == 'ledoitwolf-mlp', "init_projector must be None or 'ledoitwolf' or 'ledoitwolf-mlp'"
        if (self.init_projector is not None):
            if (self.stage != "align"):
                raise ValueError(f"init_projector can only be used in align stage")
            elif (self.init_projector_path is None):
                raise ValueError(f"init_projector_path must be provided if init_projector is not None")
        if self.init_projector == 'ledoitwolf':
            self.model.arch_specifier = 'linear'
            overwatch.critical("Initializing projector with Ledoit-Wolf")
            overwatch.info(f"Projector Type: {self.model.arch_specifier}")
            overwatch.info("Project Init Path: {self.init_projector_path}")
        elif self.init_projector == 'ledoitwolf-mlp':
            self.model.arch_specifier = 'gelu-mlp'
            overwatch.critical("Initializing projector with Ledoit-Wolf MLP")
            overwatch.info(f"Projector Type: {self.model.arch_specifier}")
            overwatch.info("Project Init Path: {self.init_projector_path}")

        if self.scale_patch_embeddings:
            overwatch.critical("Scaling Patch Embeddings by 1/sqrt(d_model)")
        if self.pre_projection_layer_norm:
            overwatch.critical("Using Layer Norm before projection")
        # STAGES 
        if self.stage == "align":
            self.epochs = self.model.align_epochs
            self.max_steps = self.model.align_max_steps
            self.global_batch_size = self.model.align_global_batch_size
            self.per_device_batch_size = self.model.align_per_device_batch_size

            self.learning_rate = self.model.align_learning_rate
            self.weight_decay = self.model.align_weight_decay
            self.max_grad_norm = self.model.align_max_grad_norm
            self.lr_scheduler_type = self.model.align_lr_scheduler_type
            self.warmup_ratio = self.model.align_warmup_ratio

            self.train_strategy = self.model.align_train_strategy

            if self.projector_type is not None:
                self.model.arch_specifier = self.projector_type
                overwatch.info(f"Projector Type: {self.projector_type}")
            if self.bigger_batch is True:
                #self.global_batch_size = self.model.align_global_batch_size * 2 # 128
                self.per_device_batch_size = self.model.align_per_device_batch_size * 4


        elif self.stage.endswith("finetune"):
            self.epochs = self.model.finetune_epochs if self.epoch_count == 1 else self.epoch_count
            self.max_steps = self.model.finetune_max_steps
            self.global_batch_size = self.model.finetune_global_batch_size #if self.mitigation is None else self.model.align_global_batch_size
            self.per_device_batch_size = self.model.finetune_per_device_batch_size
            if self.bigger_batch is True:
                #self.global_batch_size = self.model.align_global_batch_size * 2 # 128
                self.per_device_batch_size = self.model.finetune_per_device_batch_size * 2 # 16 (with 2 gradient accumulations) = 32
            if self.save_logits:
                # Ensure that save_logits_dir is not None 
                assert self.save_logits_dir is not None, "save_logits_dir cannot be None"
                # Create the directory if it does not exist
                os.makedirs(self.save_logits_dir, exist_ok=True)
                if self.per_device_batch_size >= 8:
                    self.per_device_batch_size = int(self.per_device_batch_size/8)
            elif self.load_logits:
                # Ensure that load_logits_dir is not None 
                assert self.load_logits_dir is not None, "load_logits_dir cannot be None"
                if self.per_device_batch_size >= 8:
                    self.per_device_batch_size = int(self.per_device_batch_size/8)
            elif self.soft_alpha is not None:
                # self.global_batch_size = int(self.global_batch_size/2)
                self.per_device_batch_size = int(self.per_device_batch_size/2)
            elif (self.soft_alpha_masked_interpolation is not None) or (self.add_K is not None) or (self.set_to_one) or (self.max_logit):
                self.per_device_batch_size = int(self.per_device_batch_size/4)
            elif self.mitigation =='qlora':
                # self.global_batch_size = int(self.global_batch_size/2)
                self.per_device_batch_size = int(self.per_device_batch_size/2)
            elif self.half_batch_size:
                self.per_device_batch_size = int(self.per_device_batch_size/2)

            self.learning_rate = self.model.finetune_learning_rate
            # if self.masked_with_logits:
            #     self.learning_rate *= self.masked_with_logits_lr_increase_factor

    
            self.max_grad_norm = self.model.finetune_max_grad_norm

            # Add schedule free and constant_lr_with_warmup here
            if self.schedule_free:
                self.lr_scheduler_type = 'schedule-free'
            elif self.infinite_schedule:
                self.lr_scheduler_type = 'infinite+rsqrt-cooldown'
            elif self.constant_lr_with_warmup:
                self.lr_scheduler_type = 'linear-warmup+constant'
            elif self.constant_lr:
                self.lr_scheduler_type = 'constant'
            else:
                self.lr_scheduler_type = self.model.finetune_lr_scheduler_type

            self.warmup_ratio = self.model.finetune_warmup_ratio #if self.schedule_free is False else 0.0

            # if 'align-only' in self.model.model_id:
            #     self.train_strategy = self.model.align_train_strategy
            # else:
            if self.mitigation == 'qlora' or self.merges_after_steps > 0 or self.ddp is True:
                self.train_strategy = "ddp-native"
                self.weight_decay = 0.0
            else:
                self.train_strategy = self.model.finetune_train_strategy
                self.weight_decay = self.model.finetune_weight_decay
            if self.track_ft_plasticity is True:
                self.train_strategy = "ddp-native"
                self.weight_decay = 0.0
            
            if self.llm_teacher_checkpoint is not None:
                self.train_strategy = "ddp-native"
                self.weight_decay = 0.0
                # if self.set_to_one:
                #     self.max_grad_norm = 1.0
            
            if self.projector_type is not None:
                self.model.arch_specifier = self.projector_type
                overwatch.info(f"Projector Type: {self.projector_type}")
                
        else:
            raise ValueError(f"Stage `{self.stage}` is not supported!")

        if self.ddp:
            self.train_strategy = "ddp-native"
            self.weight_decay = 0.0
            overwatch.critical(f"Using DDP with weight decay: {self.weight_decay}")
    # fmt: on


@draccus.wrap()
def pretrain(cfg: PretrainConfig) -> None:
    overwatch.info("Prismatic VLM Training :: Gathering Light")
    overwatch.info(f"track_embeddings_histogram: {cfg.track_embeddings_histogram}")
    overwatch.info(f"track_covariance: {cfg.track_covariance}")
    overwatch.info(f"align_loss: {cfg.align_loss}")
    overwatch.info(f"norm_reg: {cfg.norm_reg}")
    overwatch.info(f"norm_reg_weight: {cfg.norm_reg_weight}")
    # Note => Under `torchrun` initializing `overwatch` will automatically set up `torch.distributed`
    # torch.cuda.set_device(device_id := (overwatch.rank() % torch.cuda.device_count()))
    torch.cuda.set_device(device_id := (overwatch.local_rank()))
    torch.cuda.empty_cache()

    if cfg.disable_mixed_precision:
        overwatch.critical("Disabling Mixed Precision Training")
        cfg.model.enable_mixed_precision_training = False

    # Create Unique Run Name & Save Directory
    model_id = cfg.model.model_id
    if (dataset_id := cfg.dataset.dataset_id) == "llava-v15":
        cfg.run_id = f"{model_id}+stage-{cfg.stage}+x{cfg.seed}" if cfg.run_id is None else cfg.run_id
    else:
        cfg.run_id = f"{dataset_id}+{model_id}+stage-{cfg.stage}+x{cfg.seed}" if cfg.run_id is None else cfg.run_id

    overwatch.critical(f"Alignment Loss: {cfg.align_loss} with Align weight: {cfg.align_weight}")
    overwatch.info(f'Mitigation method: {cfg.mitigation}', ctx_level=1)
    
    if cfg.epoch_count !=1 :
        overwatch.info(f"Raising No of Epochs to : {cfg.epoch_count}",ctx_level=1)
    if cfg.mitigation is not None:
        if cfg.reduce_lora_rank_by_factor_of_fullrank != 1:
            overwatch.info(f"[bold green] Reducing lora rank by factor of full rank: {cfg.reduce_lora_rank_by_factor_of_fullrank} [/]")
        if cfg.use_rslora:
            overwatch.info(f"Using RSLoRA!", ctx_level=2)
        elif cfg.lora_use_2r_heuristic:
            overwatch.info(f"Using 2R Heuristic!", ctx_level = 2)
        # else:
        #     overwatch.info(f'Lora rank and alpha: {cfg.lora_rank} {cfg.lora_alpha}')
        overwatch.info(f"Lora target modules: {cfg.lora_target_modules}")
    if cfg.soft_alpha is not None:
        overwatch.info(f'Soft Alpha: {cfg.soft_alpha}', ctx_level=1)
    elif cfg.soft_alpha_masked_interpolation is not None:
        overwatch.info(f'Soft Alpha Masked Interpolation: {cfg.soft_alpha_masked_interpolation}', ctx_level=1)

    overwatch.info(f"StableAdamW: {cfg.stableadam}")
    # Start =>> Build Directories and Set Randomness
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    os.makedirs(run_dir := (cfg.run_root_dir / cfg.run_id), exist_ok=True)
    os.makedirs(cfg.run_root_dir / cfg.run_id / "checkpoints", exist_ok=True)
    if overwatch.is_rank_zero():
        # Additionally save a JSON version of the config
        draccus.dump(cfg, open(run_dir / "config.yaml", "w"))
        with open(run_dir / "config.yaml", "r") as f_yaml, open(run_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    # Load Vision Backbone --> on CPU, in Full Precision (initializing model, image_transform via TIMM)
    overwatch.info(f"Loading Vision Backbone [bold]{cfg.model.vision_backbone_id}[/] via TIMM ")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        cfg.model.vision_backbone_id, image_resize_strategy=cfg.model.image_resize_strategy
    )

    # Load LLM Backbone --> on CPU, in Full Precision (initializing Tokenizer + handling special tokens if necessary)
    overwatch.info(f"Loading Pretrained LLM [bold]{cfg.model.llm_backbone_id}[/] via HF Transformers")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        cfg.model.llm_backbone_id, llm_max_length=cfg.model.llm_max_length, hf_token=hf_token, cfg=cfg)

    llm_teacher = None
    if cfg.llm_teacher_checkpoint is not None:
        # Only load the model architecture, default weights for the llm.
        overwatch.info(f"Loading Teacher LLM [bold]{cfg.model.llm_backbone_id}[/] via HF Transformers")
        llm_teacher, llm_teacher_tokenizer = get_llm_backbone_and_tokenizer(
            cfg.model.llm_backbone_id, llm_max_length=cfg.model.llm_max_length, hf_token=hf_token, cfg=cfg)
        

    # Create VLM => wraps `vision_backbone` and `llm`
    overwatch.info(f"Instantiating PrismaticVLM `{model_id}` for Training Stage = `{cfg.stage}`")
    vlm = get_vlm(
        model_id,
        cfg.model.arch_specifier,
        vision_backbone,
        llm_backbone,
        enable_mixed_precision_training=cfg.model.enable_mixed_precision_training,
        llm_teacher = llm_teacher,
        init_projector_path = cfg.init_projector_path,
        scale_patch_embeddings=cfg.scale_patch_embeddings,
        pre_projection_layer_norm=cfg.pre_projection_layer_norm
    )

    # Load Weights from Checkpoint (depends on stage, config)
    overwatch.info(f"Invoking `VLM.load_checkpoint()` for `{model_id}` => Training Stage: `{cfg.stage}`")
    # Loads both LLM Backbone (if available) and Projector
    vlm.load_from_checkpoint(cfg.stage, run_dir, \
        pretrained_checkpoint=cfg.pretrained_checkpoint, cfg=cfg, \
        llm_teacher_checkpoint = cfg.llm_teacher_checkpoint)
    
    # [Explicit] Call to `freeze_backbones` here for clarity => will log exactly what is frozen / what's not!
    overwatch.info(f"Invoking `VLM.freeze_backbones()` for `{model_id}` => Training Stage: `{cfg.stage}`")
    vlm.freeze_backbones(cfg.stage, cfg.mitigation)

    # Get count of trainable parameters of the llm model
    trainable_params = sum(p.numel() for p in vlm.llm_backbone.parameters() if p.requires_grad)
    overwatch.info(f"Trainable parameters in LLM model: {trainable_params}")

    # Get Dataset for Specified Stage
    overwatch.info(f"Creating Dataset `{cfg.dataset.dataset_id}` => Stage: `{cfg.stage}`")
    train_dataset, collator = get_dataset_and_collator(
        cfg.stage,
        cfg.dataset,
        image_transform,
        tokenizer,
        prompt_builder_fn=llm_backbone.prompt_builder_fn,
        default_image_resolution=vision_backbone.default_image_resolution,
        padding_side=tokenizer.padding_side,
        load_logits=cfg.load_logits,
        load_logits_dir=cfg.load_logits_dir,
    )

    # Create Train Strategy
    overwatch.info(f"Initializing Train Strategy `{cfg.train_strategy}`")
    train_strategy = get_train_strategy(
        train_strategy=cfg.train_strategy,
        vlm=vlm,
        device_id=device_id,
        epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        global_batch_size=cfg.global_batch_size,
        per_device_batch_size=cfg.per_device_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        enable_gradient_checkpointing=cfg.model.enable_gradient_checkpointing,
        enable_mixed_precision_training=cfg.model.enable_mixed_precision_training,
        reduce_in_full_precision=cfg.model.reduce_in_full_precision,
        worker_init_fn=worker_init_fn,
        n_train_examples=len(train_dataset),
        cfg=cfg,
    )
    train_strategy.run_setup(run_dir=run_dir)

    # Create Metrics =>> Handles on the fly tracking, logging to specified trackers (e.g., JSONL, Weights & Biases)
    overwatch.info(f"Creating Metrics with Active Trackers => `{cfg.trackers}`")
    metrics = Metrics(
        cfg.trackers,
        cfg.run_id,
        run_dir,
        draccus.encode(cfg),
        cfg.stage,
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
        grad_accumulation_steps=int(train_strategy.grad_accumulation_steps),
    )  

    # # Check VLM Trainability
    # vlm.check_trainability()

    # Run Training
    overwatch.info("Starting Training Loop")
    train_strategy.run_training(train_dataset, collator, metrics, stage=cfg.stage, seed=cfg.seed)

    # Finalize
    overwatch.info("Done with Training =>> Finalizing Metrics")
    metrics.finalize()

    # And... we're done!
    overwatch.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    pretrain()
