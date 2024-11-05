"""
datasets.py

PyTorch Dataset Definitions for Prismatic models; supports processing for both the `align` and `finetune` stages, with
utilities for formatting conversations during the `finetune` stage subject to the given LLM backbone's expected
formatting (e.g., SYS_PROMPT + USER: ... ASSISTANT: ... for Vicuña v1.5 Chat models).

We currently only support Map-style Datasets; assumes that all files (annotations, images) are on local disk, and that
random access image reading is relatively cheap/fast.
"""
import copy
import json
from pathlib import Path
from typing import Dict, List, Tuple, Type

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import GPTNeoXTokenizerFast, CodeGenTokenizerFast, LlamaTokenizerFast, PreTrainedTokenizerBase, GemmaTokenizerFast

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform

import os 

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class AlignDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        chat_json: Path,
        image_dir: Path,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        super().__init__()
        self.chat_json, self.image_dir = chat_json, image_dir
        self.image_transform, self.tokenizer = image_transform, tokenizer
        self.dataset_type = "align"

        # Create Prompt Template
        self.prompt_template = "{caption}" + self.tokenizer.eos_token

        # Load Chat JSON
        with open(self.chat_json, "r") as f:
            self.examples = json.load(f)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Following the *actual* code executed from the LLaVa codebase, during the "align" phase, we actually discard
        the "prompt" from the human, and instead directly predict the caption from the image.

        As a concrete example given the "raw data" for the first example:
            example = self.examples[0]["conversations"]` = {
                [
                    {"from": "human", "value": "Render a clear and concise summary of the photo.\n<image>"},
                    {"from": "gpt", "value": "select luxury furniture 3 - inch gel memory foam mattress topper"}
                ]
            }

        Return =>> self.tokenizer("<image> select luxury furniture 3 - inch gel memory foam mattress topper\n")

        :param idx: Index to retrieve from the dataset.

        :return: Dictionary of {"pixel_values": torch.Tensor, "input_ids": torch.Tensor, "labels": torch.Tensor}
        """
        image_path, conversation = Path(self.examples[idx]["image"]), self.examples[idx]["conversations"]
        assert (len(conversation) == 2) and ("<image>" not in conversation[-1]["value"]), "Unexpected text!"

        # Format Caption --> {caption}{eos_token}
        caption = self.prompt_template.format(caption=conversation[-1]["value"].strip())

        # We treat image patches as "tokens = [p1 p2 p3, ...]"; we need to specify ordering of text/patch tokens.
        #   => Critically, we find that inserting *after* the BOS token leads to the strongest performance!
        #       - input_ids = "<s> p1 p2 p3 ... <caption_text> \n"
        #       - labels = "IGNORE IGNORE ..." (copy `input_ids` replacing <s> and p{1...K} with IGNORE)
        #
        # IMPORTANT => IF WE'RE USING HF LLM.forward(... labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids = self.tokenizer(caption, truncation=True, return_tensors="pt").input_ids[0]
        labels = copy.deepcopy(input_ids)

        # Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches right after)
        labels[0] = IGNORE_INDEX

        # Process Image --> get "pixel_values" (will either be a torch.Tensor OR a Dict[str,torch.Tensor])
        pixel_values = self.image_transform(Image.open(self.image_dir / image_path).convert("RGB"))

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)

    def get_modality_lengths(self, n_image_patches: int) -> List[Tuple[bool, int]]:
        """Get a list of modalities (unimodal / text-only vs. multimodal) and length of conversations per example."""
        modality_lengths = []
        for example in self.examples:
            is_multimodal = "image" in example
            n_words = sum([len(turn["value"].replace("<image>", "").split()) for turn in example["conversations"]])
            modality_lengths.append((is_multimodal, (n_image_patches + n_words) if is_multimodal else n_words))
        return modality_lengths

    def __len__(self) -> int:
        return len(self.examples)


class FinetuneDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        instruct_json: Path,
        image_dir: Path,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
        prompt_builder_fn: Type[PromptBuilder],
        load_logits: bool = False,
        load_logits_dir = None,
    ) -> None:
        super().__init__()
        self.instruct_json, self.image_dir = instruct_json, image_dir
        self.image_transform, self.tokenizer = image_transform, tokenizer
        self.prompt_builder_fn = prompt_builder_fn
        self.dataset_type = "finetune"
        self.load_logits = load_logits
        self.load_logits_dir = load_logits_dir


        # Load Instruct JSON
        with open(self.instruct_json, "r") as f:
            self.examples = json.load(f)

        # If using saved logits, load the mapping of idx to file paths
        if self.load_logits:
            if self.load_logits_dir is None:
                raise ValueError("When load_logits is True, load_logits_dir must be provided.")
            self.teacher_logits_index = self._build_teacher_logits_index()

    def _build_teacher_logits_index(self) -> Dict[int, str]:
        """
        Builds a mapping from sample indices to teacher logits file paths.

        :return: Dictionary mapping idx to teacher logits file paths.
        """
        teacher_logits_index = {}
        for idx in range(len(self.examples)):
            filename = f'logits_{idx}.pt'
            filepath = os.path.join(self.load_logits_dir, filename)
            teacher_logits_index[idx] = filepath
        return teacher_logits_index

        
    # === Unimodal + Multimodal Handling ===
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Unlike the *align* stage handling, for the *finetune* stage, we actually need to handle multiple "turns" of
        dialog grounded in a single image.

        To do this, we leverage the `prompt_builder_fn` which instantiates a PromptBuilder object. By calling the
        methods for adding turns and getting a prompt, we ensure proper formatting and consistency for each example.

        :param idx: Index to retrieve from the dataset.

        :return: Dictionary of {"pixel_values": torch.Tensor, "input_ids": torch.Tensor, "labels": torch.Tensor}
        """
        conversation = self.examples[idx]["conversations"]

        # Create Prompt Builder --> add each message sequentially
        prompt_builder, input_ids, labels = self.prompt_builder_fn(model_family="prismatic"), [], []
        for turn_idx, turn in enumerate(conversation):
            # Get "effective" string added to prompt --> handle whitespace for tokenizer type!
            msg = prompt_builder.add_turn(turn["from"], turn["value"])

            # Llama Tokenizer (Fast) adds extra character if a string ends in whitespace --> strip if non-empty!
            if isinstance(self.tokenizer, LlamaTokenizerFast):
                msg = msg.rstrip()
            # Gemma Tokenizer also adds extra character for trailing whitespace
            elif isinstance(self.tokenizer, GemmaTokenizerFast):
                msg = msg.rstrip()
            elif isinstance(self.tokenizer, GPTNeoXTokenizerFast):
                pass # Similar to CodeGen, handles whitespaces with internal tokens.
            elif isinstance(self.tokenizer, CodeGenTokenizerFast):
                pass
            else:
                raise ValueError(f"Tokenizer of type `{type(self.tokenizer)}` is not explicitly handled!")

            # Tokenize Input IDs
            turn_input_ids = self.tokenizer(msg, add_special_tokens=turn_idx == 0).input_ids

            # [CRITICAL] We do not want to take the loss for the "USER: <msg>" prompts =>> just the responses!
            turn_labels = (
                [IGNORE_INDEX for _ in range(len(turn_input_ids))] if (turn_idx % 2) == 0 else list(turn_input_ids)
            )

            # Add to Trackers
            input_ids.extend(turn_input_ids)
            labels.extend(turn_labels)

        # Tensorize =>> Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches after)
        #   - IMPORTANT => IF WE'RE USING HF LLM.forward(... labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

        # Handle Truncation (if necessary)
        input_ids, labels = input_ids[: self.tokenizer.model_max_length], labels[: self.tokenizer.model_max_length]

        # === Handle "unimodal" (language-only) vs. "multimodal" ===
        if "image" in self.examples[idx]:
            image_path = Path(self.examples[idx]["image"])

            # Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches right after)
            labels[0] = IGNORE_INDEX

            # Process Image --> get "pixel_values" (will either be a torch.Tensor OR a Dict[str,torch.Tensor])
            pixel_values = self.image_transform(Image.open(self.image_dir / image_path).convert("RGB"))

            example = {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "labels": labels,
                "idx": idx,
            }
        else:
            # For unimodal examples, set pixel_values to None
            example = {
                "pixel_values": None,
                "input_ids": input_ids,
                "labels": labels,
                "idx": idx,
            }

        # If load_logits is True, load the teacher logits and include them in the example
        if self.load_logits:
            teacher_logits_path = self.teacher_logits_index[idx]
            if not os.path.exists(teacher_logits_path):
                raise FileNotFoundError(f"Teacher logits file not found: {teacher_logits_path}")
            # Load the teacher logits onto the CPU
            # This avoids unnecessary GPU memory usage in DataLoader workers
            teacher_logits = torch.load(teacher_logits_path)#, map_location="cpu")
            # # Assert that teacher_logits length matches input_ids length
            # try:
            #     assert teacher_logits.size(0) == input_ids.size(0), f"Mismatch in sequence lengths for idx {idx}"
            # except Exception as e:
            #     print(e)
            #     print(f"input_ids.shape: {input_ids.shape}, teacher_logits.shape: {teacher_logits.shape}")
            #     raise
            example["teacher_logits"] = teacher_logits

        return example

    def get_modality_lengths(self) -> List[Tuple[bool, int]]:
        """Get a list of modalities (unimodal / text-only vs. multimodal) and length of conversations per example."""
        modality_lengths = []
        for example in self.examples:
            is_multimodal = "image" in example
            n_words = sum([len(turn["value"].split()) for turn in example["conversations"]])
            modality_lengths.append((is_multimodal, n_words))
        return modality_lengths

    def __len__(self) -> int:
        return len(self.examples)
