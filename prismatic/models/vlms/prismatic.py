"""
prismatic.py

PyTorch Module defining a PrismaticVLM, our general interface for defining the various different VLMs in our work.

Notes:
    - For now, we don't subclass `transformers.PretrainedModel` (or CausalLM). Instead, we assume a very limited subset
      of the {Model}ForCausalLM API that enables dispatch to the underlying LLM's `generate` utilities (feeding inputs
      through our custom projection shim).
"""
from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union

import torch
from PIL import Image
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.backbones.llm import LLMBackbone
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import VisionBackbone
from prismatic.models.vlms.base_vlm import VLM
from prismatic.overwatch import initialize_overwatch
from prismatic.util.nn_utils import FusedMLPProjector, LinearProjector, MLPProjector, GLUProjector
from torch.nn.parallel import DistributedDataParallel as DDP

from prismatic.models.backbones.mitigation import apply_mitigation

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

class PrismaticVLM(VLM):
    def __init__(
        self,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        llm_teacher: LLMBackbone = None,
        arch_specifier: str = "gelu-mlp",
        init_projector_path: str = None,
    ) -> None:
        super().__init__(
            "prismatic",
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            llm_teacher=llm_teacher
        )

        # Set Weight Initialization Seed for Projector Consistency
        torch.manual_seed(vision_backbone.embed_dim)

        # Initialize Projection (Adapter) based on `arch_specifier`
        self.arch_specifier = arch_specifier
        if arch_specifier == "linear":
            self.projector = LinearProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
            if llm_teacher is not None:
                self.teacher_projector = LinearProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
        elif arch_specifier.endswith("fused-gelu-mlp"):
            self.projector = FusedMLPProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
            if llm_teacher is not None:
                self.teacher_projector = FusedMLPProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
        elif arch_specifier.endswith("gelu-mlp"):
            self.projector = MLPProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
            if llm_teacher is not None:
                self.teacher_projector = MLPProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
        elif arch_specifier.endswith("glu-zhang"):
            self.projector = GLUProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
            if llm_teacher is not None:
                self.teacher_projector = GLUProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
        else:
            raise ValueError(f"PrismaticVLM with `{arch_specifier = }` is not supported!")
        overwatch.info(f'Dimensions of Projector: \n vision_dim: {vision_backbone.embed_dim}, llm_dim: {llm_backbone.embed_dim}')
        overwatch.info(f'Total Parameters: {vision_backbone.embed_dim * llm_backbone.embed_dim + llm_backbone.embed_dim * llm_backbone.embed_dim }')
        if init_projector_path is not None:
            try:
                self.projector.load_state_dict(torch.load(init_projector_path))
                overwatch.info(f"Loaded Projector from {init_projector_path}")
            except Exception as e:
                overwatch.error(f"Failed to load Projector from {init_projector_path} with error: {e}")
                # Exit the program
                exit(1)
        # Trackers
        self.vision_backbone_requires_grad = False

        # Set Module Keys =>> used in Checkpoint Saving / Model Loading
        self.all_module_keys = ["vision_backbone", "llm_backbone", "projector"]
        self.trainable_module_keys = []

        # === Generation Utilities ===
        #   => For computing likelihoods --> get tokens corresponding to "True", "False" and "Yes", "No"
        self.string2idx = {}
        for trigger_string in ["True", "False", "Yes", "No"] + [chr(ord("A") + i) for i in range(26)]:
            token_idx_list = self.llm_backbone.tokenizer.encode(trigger_string, add_special_tokens=False)
            assert len(token_idx_list) == 1, f'String "{trigger_string}" is tokenized as more than one token!'
            self.string2idx[trigger_string] = token_idx_list[0]

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        llm_teacher: LLMBackbone = None,
    ) -> PrismaticVLM:
        """Initialize a PrismaticVLM from a pretrained checkpoint, freezing all weights, tailored for inference."""
        vlm = cls(
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
            llm_teacher=llm_teacher
        )

        # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
        model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")["model"]

        assert (
            "projector" in model_state_dict or "llm_backbone" in model_state_dict
        ), "PrismaticVLM `from_pretrained` expects checkpoint with keys for `projector` AND `llm_backbone`!"
        #If only projector in model_state_dict, then print that, if both, then print that.
        if "projector" in model_state_dict and "llm_backbone" in model_state_dict:
            overwatch.info("[VLM.from_pretrained] Loading `projector` and `llm_backbone` from checkpoint", ctx_level=1)
            vlm.projector.load_state_dict(model_state_dict["projector"])
            new_model_state_dict = {}
            for k, v in model_state_dict['llm_backbone'].items():
                if k.startswith('llm.'):
                    new_model_state_dict[k.replace('llm.', '')] = v
                else:
                    new_model_state_dict[k] = v
            new_model_state_dict = {k: v.to('cuda') for k, v in new_model_state_dict.items()}
            vlm.llm_backbone.llm.load_state_dict(new_model_state_dict)
            del new_model_state_dict
        elif "projector" in model_state_dict:
            overwatch.error("Loading only `projector` from checkpoint", ctx_level=1)
            vlm.projector.load_state_dict(model_state_dict["projector"])
        
        del model_state_dict # @TODO: Check what consequence this has.
        # Freeze Weights
        vlm.requires_grad_(False)
        vlm.eval()

        return vlm

    def get_prompt_builder(self, system_prompt: Optional[str] = None) -> PromptBuilder:
        prompt_initializer: Type[PromptBuilder] = self.llm_backbone.prompt_builder_fn
        return prompt_initializer(self.model_family, system_prompt=system_prompt)

    def freeze_backbones(self, stage: str, mitigation: str = None) -> None:
        """
        This function sets `requires_grad_` on each of the component modules explicitly, depending on stage.

        We support two separate stages --> "align" and "finetune".
            => "align" --> vision_backbone*, llm_backbone* are frozen; only the `projector` is trained.
            => "finetune" --> vision_backbone* is frozen; both `projector` and `llm_backbone` are trained.

        :param stage: Pretraining stage in < "align" | "finetune" | "full-finetune" >
        """
        if stage == "align":
            self.vision_backbone.requires_grad_(False)
            self.llm_backbone.requires_grad_(False)
            self.projector.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["projector"]

            # Update Trackers
            self.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Trainable Components
            overwatch.info(f"[Frozen]    🥶 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[Frozen]    🥶 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1)

        elif stage == "finetune":
            self.vision_backbone.requires_grad_(False)
            if 'align-only' in self.model_id:
                self.llm_backbone.requires_grad_(False)
            elif mitigation is None:
                overwatch.info(f"LLM Backbone has requires_grad = True. Mitigation is None", ctx_level=1)
                self.llm_backbone.requires_grad_(True)
            self.projector.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            if 'align-only' in self.model_id:
                self.trainable_module_keys = ["projector"]
            else:
                self.trainable_module_keys = ["projector", "llm_backbone"]


            # Update Trackers
            self.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(f"[Frozen]    🥶 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            if 'align-only' in self.model_id:
                overwatch.info(f"[Frozen]    🥶 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            else:
                overwatch.info(f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
                
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1)

            if self.llm_teacher is not None:
                overwatch.info(f"[Frozen] 🥶 =>> Teacher LLM `{self.llm_teacher.identifier}` and Teacher Projector", ctx_level=1)
                self.llm_teacher.requires_grad_(False)

                self.teacher_projector.requires_grad_(False)


        elif stage == "full-finetune":
            self.vision_backbone.dtype = torch.float32
            self.vision_backbone.requires_grad_(True)
            self.llm_backbone.requires_grad_(True)
            self.projector.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["vision_backbone", "projector", "llm_backbone"]

            # Update Trackers
            self.vision_backbone_requires_grad = True

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(f"[TRAINABLE] 🔥 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1)

        else:
            raise ValueError(f"Stage `{stage}` is not supported for LLaVa! Try < align | finetune >")

    def load_from_checkpoint(self, stage: str, \
        run_dir: Path, pretrained_checkpoint: Optional[Path] = None,\
        llm_teacher_checkpoint = None,\
        cfg = None) -> None:
        """Load weights from checkpoint (if required by the given stage)."""
        assert stage in {"align", "finetune", "full-finetune"}, f"Stage {stage} is not supported!"

        # If we're running a `no-align` architecture, we're good!
        if self.arch_specifier.startswith("no-align"):
            overwatch.info(
                f"PrismaticVLM with `{self.arch_specifier = }` does not require pretrained weights!", ctx_level=1
            )
            return

        # Otherwise, handle stage-specific logic!
        if stage == "align":
            overwatch.info("Stage `align` does not require pretrained weights =>> Starting Training", ctx_level=1)
            return

        # Otherwise, load from `pretrained_checkpoint` or match on `run_dir` (s/+stage-finetune/+stage-align/g)
        overwatch.info("Stage `finetune` requires `align` pretrained weights", ctx_level=1)

        # Config specifies path to a checkpoint to load
        if pretrained_checkpoint is not None:
            overwatch.info(f"[VLM.load_from_checkpoint] Loading from Provided Checkpoint `{pretrained_checkpoint}`", ctx_level=1)
            model_state_dict = torch.load(pretrained_checkpoint)["model"]
            self.projector.load_state_dict(model_state_dict["projector"])
            
            if 'llm_backbone' in model_state_dict.keys():
                overwatch.info(f"Loading LLM Backbone from Provided Checkpoint", ctx_level=1)
                new_model_state_dict = {}
                for k, v in model_state_dict['llm_backbone'].items():
                    if k.startswith('llm.'):
                        new_model_state_dict[k.replace('llm.', '')] = v
                    else:
                        new_model_state_dict[k] = v
                new_model_state_dict = {k: v.to('cuda') for k, v in new_model_state_dict.items()}
                
                self.llm_backbone.llm.load_state_dict(new_model_state_dict)
                del new_model_state_dict
                overwatch.info(f"Successfully loaded LLM Backbone from Provided Checkpoint", ctx_level=1)
                # Now importantly, if self.mitigation == 'lora' or 'ia3', then we must merge the weights into itself. 
                # And create a new self.apply_mitigation
                if isinstance(cfg, dict):
                    mitigation = cfg['mitigation'] if 'mitigation' in cfg else None
                    hot_fix = cfg['hot_fix'] if 'hot_fix' in cfg else 0
                else:
                    mitigation = getattr(cfg, 'mitigation', None)
                    hot_fix = getattr(cfg, 'hot_fix', 0)
                if mitigation == 'lora' or mitigation == 'ia3' or mitigation == 'adalora':
                    overwatch.info(f"Removing the shell of: {mitigation}!", ctx_level=1)
                    self.llm_backbone.llm = self.llm_backbone.llm.merge_and_unload()
                    if hot_fix > 0:
                        self.llm_backbone.llm = apply_mitigation(self.llm_backbone.llm, cfg=cfg, hot_fix=hot_fix)
                    else:
                        self.llm_backbone.llm = apply_mitigation(self.llm_backbone.llm, cfg=cfg)
            del model_state_dict
            # return
        else:
            raise ValueError(f"Could not find valid `align` checkpoint at {pretrained_checkpoint}!")

        if llm_teacher_checkpoint is not None:
            overwatch.info(f"[VLM.load_from_checkpoint] Loading LLM Teacher Checkpoint from: {llm_teacher_checkpoint}")            
            teacher_state_dict = torch.load(llm_teacher_checkpoint, map_location="cpu")["model"]
            if "projector" in teacher_state_dict:
                overwatch.info("Teacher: Loading `projector` from checkpoint", ctx_level=1)
                self.teacher_projector.load_state_dict(teacher_state_dict["projector"])
                # new_model_state_dict = {}
                # for k, v in teacher_state_dict['llm_backbone'].items():
                #     if k.startswith('llm.'):
                #         new_model_state_dict[k.replace('llm.', '')] = v
                #     else:
                #         new_model_state_dict[k] = v
                # new_model_state_dict = {k: v.to('cuda') for k, v in new_model_state_dict.items()}
                # vlm.llm_teacher.llm.load_state_dict(new_model_state_dict)
                # del new_model_state_dict
            else:
                raise Exception("LLM Teacher Checkpoint does not have projector in state_dict")
            
            del teacher_state_dict

        # [Contract] If no `pretrained_checkpoint`, assume `align` lives in the run directory; string substitution!
        # print(f"run_name: {run_dir.name}, split: {run_dir.name.split('+')}")
        # # parents.iterdir
        # for d in run_dir.parent.iterdir():
        #     print(d)
        # model, scale, _, seed = run_dir.name.split("+")
        # align_dirs = [
        #     d
        #     for d in run_dir.parent.iterdir()
        #     if (d.name.startswith(f"{model}+{scale}") and d.name.endswith(f"+stage-align+{seed}"))
        # ]
        # assert len(align_dirs) == 1, f"Multiple or No Valid Pretrained Directories Exist -- Double Check `runs`! align_dirs: {align_dirs}"
        # if (pretrained_checkpoint := (align_dirs[0] / "checkpoints" / "latest-checkpoint.pt")).exists():
        #     overwatch.info(f"Loading from Discovered Checkpoint `{pretrained_checkpoint}`", ctx_level=1)
        #     model_state_dict = torch.load(pretrained_checkpoint)["model"]
        #     self.projector.load_state_dict(model_state_dict["projector"])
        # else:
        

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP _or_policy over the policies returned by each individual backbone (and our VLM policy)."""
        vision_fsdp_wrapping_policy = self.vision_backbone.get_fsdp_wrapping_policy()
        llm_fsdp_wrapping_policy = self.llm_backbone.get_fsdp_wrapping_policy() 

        if self.llm_teacher is not None:
            llm_teacher_fsdp_wrapping_policy = self.llm_teacher.get_fsdp_wrapping_policy()
        # Get Prismatic Wrapping Policy =>> just a module wrapping policy around `self.projector`
        prismatic_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={LinearProjector, MLPProjector, FusedMLPProjector},
        )

        # Return union (_or_) over constituent policies
        #   => Note: there is *not* a fall-through policy; any module that isn't covered by the above constituents will
        #            automatically be folded into the root VLM FSDP instance.
        return partial(
            _or_policy,
            policies=[
                vision_fsdp_wrapping_policy,
                llm_fsdp_wrapping_policy,
                llm_teacher_fsdp_wrapping_policy,
                prismatic_fsdp_wrapping_policy,
            ],
        ) if self.llm_teacher is not None else partial(
                _or_policy,
                policies=[
                    vision_fsdp_wrapping_policy,
                    llm_fsdp_wrapping_policy,
                    prismatic_fsdp_wrapping_policy,
                ],
            )

    # Note =>> We're not explicitly subclassing `PreTrainedModel` because we don't need the bloat; however, `forward()`
    #          *must* match the signature of a `{Model}ForCausalLM` so that we can inherit from `GenerationMixin`
    # === GenerationMixin Methods ===

    def get_embeddings(input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        multimodal_indices: Optional[torch.LongTensor] = None,
        return_labels: Optional[bool] = False,):
            
            # Run Visual Feature Extraction
            with torch.set_grad_enabled(self.vision_backbone_requires_grad):
                if isinstance(pixel_values, dict):
                    patch_features = self.vision_backbone({k: pixel_values[k][multimodal_indices] for k in pixel_values})
                else:
                    patch_features = self.vision_backbone(pixel_values[multimodal_indices])

            # Projection Logic :: [bsz, num_patches, llm_embed_dim] =>> num_patches = (2 *) (256 + 1) for ViT-L + CLS
            projector = self.projector.module if isinstance(self.projector, DDP) else self.projector
            projected_patch_embeddings = projector(patch_features)
            projected_patch_attention_mask = None
            if attention_mask is not None:
                projected_patch_attention_mask = torch.full(
                    (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

            # Get Input Embeddings from LLM Backbone :: [bsz, input_seq_len, llm_embed_dim]
            llm_backbone = self.llm_backbone.module if isinstance(self.llm_backbone, DDP) else self.llm_backbone
            input_embeddings = llm_backbone.embed_input_ids(input_ids)
            return projected_patch_embeddings, input_embeddings


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        multimodal_indices: Optional[torch.LongTensor] = None,
        return_labels: Optional[bool] = False,
    ) -> CausalLMOutputWithPast:
        """Run a forward pass through the VLM, returning a CausalLMOutputWithPast instance (contains loss)."""
        
        # # Move input tensors to the correct device
        # device = next(self.parameters()).device
        # input_ids = input_ids.to(device)
        # if attention_mask is not None:
        #     attention_mask = attention_mask.to(device)
        # if pixel_values is not None:
        #     if isinstance(pixel_values, dict):
        #         pixel_values = {k: v.to(device) for k, v in pixel_values.items()}
        #     else:
        #         pixel_values = pixel_values.to(device)
        # if labels is not None:
        #     labels = labels.to(device)
        # if multimodal_indices is not None:
        #     multimodal_indices = multimodal_indices.to(device)

        # Handle Inference (leverage cache, short-circuit on just LLM forward)
        if input_ids.shape[1] == 1 and past_key_values is not None:
            # We're leveraging the cache, so just redirect to `self.llm_backbone` with `input_ids` and `past_key_values`
            llm_backbone = self.llm_backbone.module if isinstance(self.llm_backbone, DDP) else self.llm_backbone
            output = llm_backbone(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            if return_labels:
                return output, None
            return output

        elif input_ids.shape[1] == 1 or pixel_values is None:
            raise RuntimeError("Invalid `forward()` call!")

        # Handle Multimodal Indices is None --> pretend like the batch is fully multimodal (always image + text)!
        if multimodal_indices is None:
            multimodal_indices = torch.arange(len(input_ids), dtype=torch.long, device=input_ids.device)

        # Handle Multimodal Indices is Empty (len == 0) --> simple unimodal forward
        elif len(multimodal_indices) == 0:
            llm_backbone = self.llm_backbone.module if isinstance(self.llm_backbone, DDP) else self.llm_backbone
            output = llm_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            if return_labels:
                return output, labels
            return output

        # Run Visual Feature Extraction
        with torch.set_grad_enabled(self.vision_backbone_requires_grad):
            if isinstance(pixel_values, dict):
                patch_features = self.vision_backbone({k: pixel_values[k][multimodal_indices] for k in pixel_values})
            else:
                patch_features = self.vision_backbone(pixel_values[multimodal_indices])

        # Projection Logic :: [bsz, num_patches, llm_embed_dim] =>> num_patches = (2 *) (256 + 1) for ViT-L + CLS
        projector = self.projector.module if isinstance(self.projector, DDP) else self.projector
        projected_patch_embeddings = projector(patch_features)
        projected_patch_attention_mask = None
        if attention_mask is not None:
            projected_patch_attention_mask = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

        # Get Input Embeddings from LLM Backbone :: [bsz, input_seq_len, llm_embed_dim]
        llm_backbone = self.llm_backbone.module if isinstance(self.llm_backbone, DDP) else self.llm_backbone
        input_embeddings = llm_backbone.embed_input_ids(input_ids)

        # Build Multimodal Embeddings (and build resulting attention mask)
        multimodal_embeddings = torch.cat(
            [
                input_embeddings[multimodal_indices, :1, :],
                projected_patch_embeddings,
                input_embeddings[multimodal_indices, 1:, :],
            ],
            dim=1,
        )
        multimodal_attention_mask = None
        if attention_mask is not None:
            multimodal_attention_mask = torch.cat(
                [
                    attention_mask[multimodal_indices, :1],
                    projected_patch_attention_mask,
                    attention_mask[multimodal_indices, 1:],
                ],
                dim=1,
            )

        # [Contract] We assume the first token of `labels` (associated with <BOS>) is already marked as "IGNORE"
        #   => We'll ignore the per-token outputs for each of the patch embeddings as well!
        multimodal_labels = None
        if labels is not None:
            projected_patch_labels = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            multimodal_labels = torch.cat(
                [labels[multimodal_indices, :1], projected_patch_labels, labels[multimodal_indices, 1:]], dim=1
            )

        # === Add Unimodal Handling ===

        # Create Fused Embeddings, Attention Mask, and Labels by Merging with "unimodal" Inputs (if applicable)
        unimodal_indices = torch.tensor(
            [idx for idx in range(len(input_ids)) if idx not in multimodal_indices],
            dtype=torch.long,
            device=multimodal_indices.device,
        )

        # No "unimodal" data --> Fused == Multimodal
        if len(unimodal_indices) == 0:
            fused_embeddings = multimodal_embeddings
            fused_attention_mask = multimodal_attention_mask
            fused_labels = multimodal_labels

        else:
            # Otherwise --> Merge w/ unimodal data

            # This doesn't matter --> but in the "normal" case this is the embedding of the <PAD> token
            #   => NOTE :: Verified that `zeros/randn/empty/<PAD> embedding` all return the same result!
            unimodal_embeddings_pad = torch.zeros(
                (len(unimodal_indices), projected_patch_embeddings.shape[1], input_embeddings.shape[2]),
                dtype=input_embeddings.dtype,
                device=input_embeddings.device,
            )
            unimodal_attention_pad = torch.full(
                (len(unimodal_indices), projected_patch_embeddings.shape[1]),
                False,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            unimodal_labels_pad = torch.full(
                (len(unimodal_indices), projected_patch_embeddings.shape[1]),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )

            unimodal_embeddings = torch.cat([input_embeddings[unimodal_indices], unimodal_embeddings_pad], dim=1)
            unimodal_attention_mask = torch.cat([attention_mask[unimodal_indices], unimodal_attention_pad], dim=1)
            unimodal_labels = torch.cat([labels[unimodal_indices], unimodal_labels_pad], dim=1)

            # Create "Fused" Tensors by Stacking Multimodal & Unimodal
            fused_embeddings = torch.vstack([multimodal_embeddings, unimodal_embeddings])
            fused_attention_mask = torch.vstack([multimodal_attention_mask, unimodal_attention_mask])
            fused_labels = torch.vstack([multimodal_labels, unimodal_labels])

        # Run LLM Forward --> returns CausalLMOutputWithPast!
        output = llm_backbone(
            input_ids=None,
            attention_mask=fused_attention_mask,
            position_ids=None,
            past_key_values=past_key_values,
            inputs_embeds=fused_embeddings,
            labels=fused_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if return_labels:
            return output, fused_labels
        return output

    #   => Note: The following methods override the functionality of `transformers.GenerationMixin`; these expect the
    #            contract in each of the function signatures, and also expect our `forward` function to roughly take
    #            the same arguments as the underlying LLM (see `LlamaModelForCausalLM` as an example)

    def teacher_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        multimodal_indices: Optional[torch.LongTensor] = None,
        return_labels: Optional[bool] = False,
    ) -> CausalLMOutputWithPast:
        """Run a forward pass through the VLM, returning a CausalLMOutputWithPast instance (contains loss)."""
        
        # # Move input tensors to the correct device
        # device = next(self.parameters()).device
        # input_ids = input_ids.to(device)
        # if attention_mask is not None:
        #     attention_mask = attention_mask.to(device)
        # if pixel_values is not None:
        #     if isinstance(pixel_values, dict):
        #         pixel_values = {k: v.to(device) for k, v in pixel_values.items()}
        #     else:
        #         pixel_values = pixel_values.to(device)
        # if labels is not None:
        #     labels = labels.to(device)
        # if multimodal_indices is not None:
        #     multimodal_indices = multimodal_indices.to(device)

        # Handle Inference (leverage cache, short-circuit on just LLM forward)
        if input_ids.shape[1] == 1 and past_key_values is not None:
            # We're leveraging the cache, so just redirect to `self.llm_backbone` with `input_ids` and `past_key_values`
            llm_backbone = self.llm_teacher.module if isinstance(self.llm_teacher, DDP) else self.llm_teacher
            output = llm_backbone(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            if return_labels:
                return output, None
            return output

        elif input_ids.shape[1] == 1 or pixel_values is None:
            raise RuntimeError("Invalid `forward()` call!")

        # Handle Multimodal Indices is None --> pretend like the batch is fully multimodal (always image + text)!
        if multimodal_indices is None:
            multimodal_indices = torch.arange(len(input_ids), dtype=torch.long, device=input_ids.device)

        # Handle Multimodal Indices is Empty (len == 0) --> simple unimodal forward
        elif len(multimodal_indices) == 0:
            llm_backbone = self.llm_teacher.module if isinstance(self.llm_teacher, DDP) else self.llm_teacher
            output = llm_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            if return_labels:
                return output, labels
            return output

        # Run Visual Feature Extraction
        with torch.set_grad_enabled(self.vision_backbone_requires_grad):
            if isinstance(pixel_values, dict):
                patch_features = self.vision_backbone({k: pixel_values[k][multimodal_indices] for k in pixel_values})
            else:
                patch_features = self.vision_backbone(pixel_values[multimodal_indices])

        # Projection Logic :: [bsz, num_patches, llm_embed_dim] =>> num_patches = (2 *) (256 + 1) for ViT-L + CLS
        projector = self.teacher_projector.module if isinstance(self.teacher_projector, DDP) else self.teacher_projector
        projected_patch_embeddings = projector(patch_features)
        projected_patch_attention_mask = None
        if attention_mask is not None:
            projected_patch_attention_mask = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

        # Get Input Embeddings from LLM Backbone :: [bsz, input_seq_len, llm_embed_dim]
        llm_backbone = self.llm_teacher.module if isinstance(self.llm_teacher, DDP) else self.llm_teacher
        input_embeddings = llm_backbone.embed_input_ids(input_ids)

        # Build Multimodal Embeddings (and build resulting attention mask)
        multimodal_embeddings = torch.cat(
            [
                input_embeddings[multimodal_indices, :1, :],
                projected_patch_embeddings,
                input_embeddings[multimodal_indices, 1:, :],
            ],
            dim=1,
        )
        multimodal_attention_mask = None
        if attention_mask is not None:
            multimodal_attention_mask = torch.cat(
                [
                    attention_mask[multimodal_indices, :1],
                    projected_patch_attention_mask,
                    attention_mask[multimodal_indices, 1:],
                ],
                dim=1,
            )

        # [Contract] We assume the first token of `labels` (associated with <BOS>) is already marked as "IGNORE"
        #   => We'll ignore the per-token outputs for each of the patch embeddings as well!
        multimodal_labels = None
        if labels is not None:
            projected_patch_labels = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            multimodal_labels = torch.cat(
                [labels[multimodal_indices, :1], projected_patch_labels, labels[multimodal_indices, 1:]], dim=1
            )

        # === Add Unimodal Handling ===

        # Create Fused Embeddings, Attention Mask, and Labels by Merging with "unimodal" Inputs (if applicable)
        unimodal_indices = torch.tensor(
            [idx for idx in range(len(input_ids)) if idx not in multimodal_indices],
            dtype=torch.long,
            device=multimodal_indices.device,
        )

        # No "unimodal" data --> Fused == Multimodal
        if len(unimodal_indices) == 0:
            fused_embeddings = multimodal_embeddings
            fused_attention_mask = multimodal_attention_mask
            fused_labels = multimodal_labels

        else:
            # Otherwise --> Merge w/ unimodal data

            # This doesn't matter --> but in the "normal" case this is the embedding of the <PAD> token
            #   => NOTE :: Verified that `zeros/randn/empty/<PAD> embedding` all return the same result!
            unimodal_embeddings_pad = torch.zeros(
                (len(unimodal_indices), projected_patch_embeddings.shape[1], input_embeddings.shape[2]),
                dtype=input_embeddings.dtype,
                device=input_embeddings.device,
            )
            unimodal_attention_pad = torch.full(
                (len(unimodal_indices), projected_patch_embeddings.shape[1]),
                False,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            unimodal_labels_pad = torch.full(
                (len(unimodal_indices), projected_patch_embeddings.shape[1]),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )

            unimodal_embeddings = torch.cat([input_embeddings[unimodal_indices], unimodal_embeddings_pad], dim=1)
            unimodal_attention_mask = torch.cat([attention_mask[unimodal_indices], unimodal_attention_pad], dim=1)
            unimodal_labels = torch.cat([labels[unimodal_indices], unimodal_labels_pad], dim=1)

            # Create "Fused" Tensors by Stacking Multimodal & Unimodal
            fused_embeddings = torch.vstack([multimodal_embeddings, unimodal_embeddings])
            fused_attention_mask = torch.vstack([multimodal_attention_mask, unimodal_attention_mask])
            fused_labels = torch.vstack([multimodal_labels, unimodal_labels])

        # Run LLM Forward --> returns CausalLMOutputWithPast!
        output = llm_backbone(
            input_ids=None,
            attention_mask=fused_attention_mask,
            position_ids=None,
            past_key_values=past_key_values,
            inputs_embeds=fused_embeddings,
            labels=fused_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if return_labels:
            return output, fused_labels
        return output

    #   => Note: The following methods override the functionality of `transformers.GenerationMixin`; these expect the
    #            contract in each of the function signatures, and also expect our `forward` function to roughly take
    #            the same arguments as the underlying LLM (see `LlamaModelForCausalLM` as an example)


    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        **kwargs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Borrowed from `LlamaForCausalLM` --> in general, just handles caching logic during generation."""
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Make sure `pixel_values` are preserved in `model_inputs`
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
            }
        )

        return model_inputs

    @torch.inference_mode()
    def generate_batch(
        self,
        pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]],
        texts: List[str],
        return_string_probabilities: Optional[List[str]] = None,
        **kwargs: str,
    ) -> Union[List[str], List[List[float]]]:
        # For now, only support generation with a batch size of 1 for simplicity
        tokenizer = self.llm_backbone.tokenizer

        # Prepare Inputs
        batch_input_ids = [
            tokenizer(text, truncation=True, return_tensors="pt").input_ids.to(self.device) for text in texts
        ]
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Create Output Lists
        gen_texts, gen_probabilities = [], []

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            for idx, input_ids in enumerate(batch_input_ids):
                if isinstance(pixel_values, torch.Tensor):
                    pixel_values = pixel_values[idx]
                elif isinstance(pixel_values, dict):
                    pixel_values = {k: pixel_values[k][idx] for k in pixel_values}
                else:
                    raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

                # Handle `return_string_probabilities`
                if return_string_probabilities is None:
                    full_out_ids = super().generate(input_ids=input_ids, pixel_values=pixel_values, **kwargs)
                    gen_ids = full_out_ids[0, input_ids.shape[1] :]

                    # Decode `gen_ids` and strip any <EOS> tokens
                    gen_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

                else:
                    full_out_dict = super().generate(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        output_scores=True,
                        return_dict_in_generate=True,
                        **kwargs,
                    )

                    # Generation pattern should usually be [TOKEN] <EOS> for True/False and Yes/No Generations
                    gen_ids = full_out_dict.sequences[0, input_ids.shape[1] :]

                    # [Debug] Verify that the first token generated is in `self.string2idx.values()`
                    # assert gen_ids[0] in self.string2idx.values(), "Generated ID not in mapping!"

                    # Decode `gen_ids` and strip any <EOS> tokens
                    gen_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

                    # Get all token probabilities --> softmax over logits
                    token_probs = torch.softmax(full_out_dict.scores[0][0], dim=0)

                    # Get *normalized* probabilities for all values in `return_token_probabilities`
                    slice_idxs = torch.tensor([self.string2idx[s] for s in return_string_probabilities])
                    string_probs_unnormalized = token_probs[slice_idxs]
                    string_probs = string_probs_unnormalized / string_probs_unnormalized.sum()
                    gen_probabilities.append(string_probs.cpu().numpy().tolist())

        return gen_texts if return_string_probabilities is None else gen_probabilities

    @torch.inference_mode()
    def generate(self, image: Image, prompt_text: str, **kwargs: str) -> str:
        # For now, only support generation with a batch size of 1 for simplicity
        image_transform, tokenizer = self.vision_backbone.image_transform, self.llm_backbone.tokenizer

        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            # fmt: off
            generated_ids = super().generate(
                input_ids=input_ids,            # Shape: [1, seq]
                pixel_values=pixel_values,      # Shape: [1, 3, res, res] or Dict[str, Shape[1, 3, res, res]]
                **kwargs
            )
            # fmt: on

        generated_text = tokenizer.decode(generated_ids[0, input_ids.shape[1] :], skip_special_tokens=True).strip()

        return generated_text

    def check_trainability(self):
        """
        Prints whether the llm_backbone, projector, and visual_backbone are trainable.
        """
        components = {
            "LLM Backbone": self.llm_backbone,
            "Projector": self.projector,
            "Visual Backbone": self.vision_backbone,
        }
        
        for comp_name, comp in components.items():
            print(f"Checking trainability for: {comp_name}")
            trainable = any(param.requires_grad for param in comp.parameters())
            print(f"  Trainable: {trainable}")
            if not trainable:
                print("  All parameters are frozen.")
            else:
                for name, param in comp.named_parameters():
                    if param.requires_grad:
                        print(f"  {name} is trainable")




