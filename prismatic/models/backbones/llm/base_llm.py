"""
base_llm.py

Abstract class definition of a large (autoregressive) language model backbone (LLM), with full annotations of class
methods, utility functions, and initialization logic.

We also define the generic HFLLMBackbone class here, providing a default interface for loading any HF
AutoModelForCausalLM (e.g., LLamaForCausalLM). In general, we make the assumption that any given LLM backbone implements
the AutoModelForCausalLM API (though we may add Seq2Seq models in the future).

We make this assumption to keep the LLM handling in this codebase relatively lightweight, and to inherit all the nice HF
utilities around different types of decoding/generation strategies.
"""
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, List, Optional, Type

import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from peft.utils.other import fsdp_auto_wrap_policy

from transformers import AutoConfig, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.overwatch import initialize_overwatch

# from transformers import BitsAndBytesConfig



# Suppress HF Deprecation Warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === Abstract Base Class for arbitrary HF LLM Backbones ===
class LLMBackbone(nn.Module, ABC):
    def __init__(self, llm_backbone_id: str) -> None:
        super().__init__()
        self.identifier = llm_backbone_id

        # Instance attributes for an LLM Backbone
        self.llm: PreTrainedModel = None
        self.tokenizer: PreTrainedTokenizerBase = None

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        return self.tokenizer

    @abstractmethod
    def get_fsdp_wrapping_policy(self) -> Callable: ...

    @abstractmethod
    def enable_gradient_checkpointing(self) -> None: ...

    @abstractmethod
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        """Run a forward pass through the LLM given targets (labels), returning the scalar Cross-Entropy Loss"""
        raise NotImplementedError

    @abstractmethod
    def embed_input_ids(self, input_ids: torch.LongTensor) -> torch.Tensor: ...

    @property
    @abstractmethod
    def prompt_builder_fn(self) -> Type[PromptBuilder]: ...

    @property
    @abstractmethod
    def transformer_layer_cls(self) -> Type[nn.Module]: ...

    @property
    @abstractmethod
    def half_precision_dtype(self) -> torch.dtype: ...

    @property
    def embed_dim(self) -> int:
        return self.llm.config.hidden_size

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id


# === Abstract Base Class for Arbitrary HF Causal LLMs ===
class HFCausalLLMBackbone(LLMBackbone, ABC):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_family: str,
        llm_cls: Type[PreTrainedModel],
        hf_hub_path: str,
        llm_max_length: int = 2048,
        hf_token: Optional[str] = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = False,
        cfg = None,
    ) -> None:
        super().__init__(llm_backbone_id)
        self.llm_family = llm_family
        self.llm_max_length = llm_max_length
        self.inference_mode = inference_mode
        self.cfg = cfg
        if isinstance(self.cfg, dict):
            self.mitigation = self.cfg['mitigation'] if 'mitigation' in self.cfg else None
            self.stage = self.cfg['stage'] if 'stage' in self.cfg else None
            self.first_lora_after_warmup = self.cfg['first_lora_after_warmup'] if 'first_lora_after_warmup' in self.cfg else None
        else:
            self.mitigation = getattr(self.cfg, 'mitigation', None)
            self.stage = getattr(self.cfg, 'stage', None)
            self.first_lora_after_warmup = getattr(self.cfg, 'first_lora_after_warmup', None)
        # self.bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype="bfloat16",
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_storage="bfloat16",
        # )
        self.load_8bit = False
        if isinstance(self.cfg, dict) and 'load_8bit' in self.cfg:
            self.load_8bit = self.cfg['load_8bit']
        else:
            self.load_8bit = getattr(self.cfg, 'load_8bit', False)
        assert self.load_8bit is False or self.load_8bit is True, "load_8bit must be a boolean"

        use_flash_attention_2 = use_flash_attention_2 if not self.inference_mode else False
        if 'pythia' in self.llm_family:
            use_flash_attention_2 = False

        # Initialize LLM (downloading from HF Hub if necessary) --> `llm_cls` is the actual {Model}ForCausalLM class!
        #   => Note: We're eschewing use of the AutoModel API so that we can be more explicit about LLM-specific details
        if not self.inference_mode:
            overwatch.info(f"Loading [bold]{llm_family}[/] LLM from [underline]`{hf_hub_path}`[/]", ctx_level=1)
            self.llm = llm_cls.from_pretrained(
                hf_hub_path,
                token=hf_token,
                use_flash_attention_2= use_flash_attention_2,
                # The following parameters are set to prevent `UserWarnings` from HF; we want greedy decoding!
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                load_in_4bit= True if (self.mitigation=='qlora' and self.load_8bit is False) else False,
                load_in_8bit= True if (self.mitigation=='qlora' and self.load_8bit is True) else False,
                #torch_dtype = torch.bfloat16 if self.mitigation=='qlora' else None
            )          
        # elif self.load_from_hf_anyway:
        #     overwatch.info(f"Loading [bold]{llm_family}[/] LLM from [underline]`{hf_hub_path}`[/]", ctx_level=1)
        #     self.llm = llm_cls.from_pretrained(
        #         hf_hub_path,
        #         token=hf_token,
        #         use_flash_attention_2=use_flash_attention_2 if not self.inference_mode else False,
        #         # The following parameters are set to prevent `UserWarnings` from HF; we want greedy decoding!
        #         do_sample=False,
        #         temperature=1.0,
        #         top_p=1.0,
        #         load_in_4bit=True if self.mitigation=='qlora' else False,  #quantization_config=self.bnb_config if self.mitigation=='qlora' else None, #
        #         #torch_dtype = torch.bfloat16 if self.mitigation=='qlora' else None
        #     )

        # [Contract] `inference_mode` means we're loading from a pretrained checkpoint; no need to load base weights!
        # [Breaking Contract] we still load base weights, if load_from_hf_anyway is set to True
        else:
            if self.stage == 'align':
                self.llm = llm_cls.from_pretrained(
                    hf_hub_path,
                    token=hf_token,
                    use_flash_attention_2=False,
                    # The following parameters are set to prevent `UserWarnings` from HF; we want greedy decoding!
                    do_sample=True,
                )  
            elif self.mitigation == 'qlora':
                overwatch.info(f"[QLORA BUILD] Building empty [bold]{llm_family}[/] LLM from [underline]`{hf_hub_path}`[/]", ctx_level=1)
                # llm_config = AutoConfig.from_pretrained(hf_hub_path, token=hf_token)
                # llm_config.update({
                #     "do_sample": False,
                #     "max_new_tokens": 2048,
                #     "temperature": None,
                #     "top_p": None,
                # })
                
                # self.llm  = llm_cls.from_pretrained(
                #                 hf_hub_path,
                #                 config=llm_config, 
                #                 use_flash_attention_2=False,
                #                 token=hf_token,
                #                 load_in_4bit=True if self.load_8bit is False else False,
                #                 load_in_8bit=True if self.load_8bit is True else False,
                #             )
                self.llm = llm_cls.from_pretrained(
                    hf_hub_path,
                    token=hf_token,
                    use_flash_attention_2=False,
                    # The following parameters are set to prevent `UserWarnings` from HF; we want greedy decoding!
                    do_sample=True,
                    load_in_4bit=True if self.load_8bit is False else False,
                    load_in_8bit=True if self.load_8bit is True else False,
                ) 
            else:
                overwatch.info(f"Building empty [bold]{llm_family}[/] LLM from [underline]`{hf_hub_path}`[/]", ctx_level=1)
                llm_config = AutoConfig.from_pretrained(hf_hub_path, token=hf_token, 
                )
                self.llm = llm_cls._from_config(llm_config)
                # self.llm = llm_cls.from_pretrained(
                #     hf_hub_path,
                #     token=hf_token,
                #     use_flash_attention_2=False,
                #     # The following parameters are set to prevent `UserWarnings` from HF; we want greedy decoding!
                #     do_sample=True,
                #     load_in_4bit=True if self.mitigation=='qlora' else False,  #quantization_config=self.bnb_config if self.mitigation=='qlora' else None, #
                #     #torch_dtype = torch.bfloat16 if self.mitigation=='qlora' else None
                # )   
            #
            # print("DEBUG EMPTY LLM INITIALIZE")
            # import IPython
            # IPython.embed()
            # exit(0)

        # Lightweight Handling (with extended explanation) for setting some LLM Parameters
        #   => Set `decoder.use_cache = False` --> incompatible with gradient checkpointing (+ training in general)
        #
        #      Reference: https://discuss.huggingface.co/t/what-is-the-purpose-of-use-cache-in-decoder/958
        self.llm.config.use_cache = False if not self.inference_mode else True

        #   => Turns out that when gradient checkpointing is on and the underlying LLM has no "trainable" parameters
        #      (requires_grad is False), backprop will fail; setting `enable_input_requires_grad()` registers a new
        #      forward hook that fixes this =>> also totally safe for the "full finetuning" setting!
        if not self.inference_mode:
            self.llm.enable_input_require_grads()

        # Load (Fast) Tokenizer
        overwatch.info(f"Loading [bold]{llm_family}[/] (Fast) Tokenizer via the AutoTokenizer API", ctx_level=1)
        
        # if self.llm_family == 'pythia':
        #     self.tokenizer = AutoTokenizer.from_pretrained(hf_hub_path, \
        #                     model_max_length=self.llm_max_length, \
        #                     token=hf_token,\
        #                     padding_side="right",
        #                     add_bos_token=False, # GPTNeoXTokenizerFast doesn't add BOS token by default
        #                     add_eos_token=False) # GPTNeoXTokenizerFast doesn't add EOS token by default
        # else:
        self.tokenizer = AutoTokenizer.from_pretrained(hf_hub_path, \
                            model_max_length=self.llm_max_length, \
                            token=hf_token,\
                            padding_side="right")

        # Validation =>> Our VLM logic currently operates under the assumption that the tokenization of a new input
        #                starts with a <BOS> token unless `add_special_tokens = False`; for these models, we empirically
        #                find that adding image patches *after* the BOS leads to much better performance
        #
        # As a result we explicitly validate that a tokenizer conforms to the expected behavior; if you're reading this
        # line, it's probably because you're adding a new LLM with a different tokenizer behavior. If so, feel free to
        # override the `SPECIAL_CASES` set below, but make sure to make the appropriate changes in the `datasets.py`
        # and VLM `forward()` logic!
        SPECIAL_CASES = {
             # Phi-2 Tokenizer doesn't add any BOS tokens by default, and sets BOS == EOS == "<|endoftext|>"
             #   =>> We'll prepend BOS to first input (to play nicely with image token insertion logic; verified that
             #       this works well with base LLM generation.
             #   =>> Like Llama-2 Tokenizers -- we'll add a special PAD token for training purposes.
            "phi-1_5-1b",\
            "phi-2-3b",\
            "pythia-6p9b",\
            "pythia-2p8b",\
            "pythia-1p4b",\
            "pythia-1b",\
            "pythia-410m",\
            "pythia-160m",\
            "pythia-1p4b-instruct",\
        }
        if self.identifier in SPECIAL_CASES:
            return

        # Note =>> this assert should hold for all Llama-derived tokenizers (`LlamaTokenizerFast` ==> includes Mistral!
        assert (self.tokenizer("Test 123", add_special_tokens=True).input_ids[0] == self.tokenizer.bos_token_id) and (
            self.tokenizer("Test 123", add_special_tokens=False).input_ids[0] != self.tokenizer.bos_token_id
        ), (
            f"Default Tokenizer of type `{type(self.tokenizer)}` does not automatically prefix inputs with BOS token!\n"
            "Please read the comment in `base_llm.py` for more information!"
        )

        # Additionally, explicitly verify that Tokenizer padding_side is set to right for training!
        assert self.tokenizer.padding_side == "right", "Tokenizer `padding_side` is not set to `right`!"

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return a `transformer_auto_wrap_policy` where we wrap each instance of `self.transformer_layer_cls`"""
        if self.mitigation is None:
            overwatch.info(f"LLM's FSDP Wrap Policy: [bold]STANDARD[/]", ctx_level=1)
            transformer_block_policy = partial(
                transformer_auto_wrap_policy, transformer_layer_cls={self.transformer_layer_cls}
            )
        else:
            overwatch.info(f"LLM's FSDP Wrap Policy: [bold]PEFT[/]", ctx_level=1)
            transformer_block_policy = fsdp_auto_wrap_policy(self.llm)
        return transformer_block_policy

    def enable_gradient_checkpointing(self) -> None:
        """Dispatch to underlying LLM instance's `gradient_checkpointing_enable`; defined for all `PretrainedModel`."""
        self.llm.gradient_checkpointing_enable()

    def embed_input_ids(self, input_ids: torch.LongTensor) -> torch.Tensor:
        # emb_layer = self.llm.get_input_embeddings()
        # overwatch.info(f"Embedding layer type: {type(emb_layer)}")
        # overwatch.info(f"Embedding weight shape: {emb_layer.weight.shape}")
        # if not isinstance(emb_layer, nn.Embedding):
        #     overwatch.error(f"Embedding layer is not nn.Embedding: {type(emb_layer)}")
        # elif not isinstance(emb_layer.weight, torch.Tensor) or emb_layer.weight.dim() != 2:
        #     overwatch.error(f"Embedding weight is not a 2D tensor: {emb_layer.weight}")
        # return emb_layer(input_ids)
        return self.llm.get_input_embeddings()(input_ids)

    # [Contract] Should match the `forward` call of the underlying `llm` instance!
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        output: CausalLMOutputWithPast = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return output
