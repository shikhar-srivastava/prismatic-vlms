
import argparse
import logging
import sys
import json
import os
from pathlib import Path
from typing import List, Optional, Union

from huggingface_hub import hf_hub_download

from prismatic.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
from prismatic.models.registry import GLOBAL_REGISTRY, MODEL_REGISTRY
from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.models import load

import torch
from lm_eval import evaluator

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === HF Hub Repository ===
# HF_HUB_REPO = "TRI-ML/prismatic-vlms"

SELECTED_NLP_TASKS = ["wsc273","arc_easy","arc_challenge","winogrande","lambada_standard"] #, "webqs"] ["wsc273"]#

# === Available Models ===
def available_models() -> List[str]:
    return list(MODEL_REGISTRY.keys())


def available_model_names() -> List[str]:
    return list(GLOBAL_REGISTRY.items())


def get_model_description(model_id_or_name: str) -> str:
    if model_id_or_name not in GLOBAL_REGISTRY:
        raise ValueError(f"Couldn't find `{model_id_or_name = }; check `prismatic.available_model_names()`")

    # Print Description & Return
    print(json.dumps(description := GLOBAL_REGISTRY[model_id_or_name]["description"], indent=2))

    return description


# === Load Pretrained Model ===


def load_and_write(write_checkpoint_path, model_id_or_path):
    # if model_id_path already contains checkpoint_llm_only folder then skip
    if os.path.exists(os.path.join(write_checkpoint_path, "checkpoint_llm_only")):
        overwatch.info(f"Checkpoint already exists at {write_checkpoint_path}")

        # if checkpoints_base_llm also exists then skip
        if os.path.exists(os.path.join(write_checkpoint_path, "checkpoints_base_llm")):
            overwatch.info(f"Base checkpoint already exists at {write_checkpoint_path}")
            return
        
    vlm = load(model_id_or_path)
    llm_hugface_save_path = os.path.join(
        write_checkpoint_path,
        "checkpoint_llm_only",
    )
    llm_base_path = os.path.join(
        write_checkpoint_path,
        "checkpoints_base_llm",
    )
    overwatch.info(f"Writing LLM & Tokenizer to checkpoint: {llm_hugface_save_path}")
    llm_backbone = vlm.llm_backbone
    llm_backbone.llm.save_pretrained(llm_hugface_save_path)
    llm_backbone.llm.base_model.save_pretrained(llm_base_path)
    llm_backbone.tokenizer.save_pretrained(llm_base_path)
    llm_backbone.tokenizer.save_pretrained(llm_hugface_save_path)
    return llm_hugface_save_path


def nlp_evaluation(llm_checkpoint_path):
    print(f"\n\t NLP Evaluation: {llm_checkpoint_path}")
    task_names = SELECTED_NLP_TASKS
    model_args = f"pretrained={str(llm_checkpoint_path)},trust_remote_code=True"
    print(f"Evaluating on {task_names} with model_args: {model_args}")
    
    results = evaluator.simple_evaluate( 
            model="hf",
            model_args=model_args,
            tasks=task_names,
            device="cuda:0",
            log_samples=True,
            output_path=llm_checkpoint_path
        )
    with open(f"{llm_checkpoint_path}/nlp_evaluation_results.log", "w") as log_file:
        log_file.write(json.dumps(results, indent=4))
    return results


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Process VLM checkpoint for NLP evaluation.")
    # parser.add_argument('--write_path', type=str, required=True, help='Path to write the LLM checkpoint')
    # parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the VLM checkpoint')
    # args = parser.parse_args()
    
    # main(args.write_path, args.checkpoint_path)
    runs_directory = Path("/localdisk/ssrivas9/prismatic-vlms/runs")
    for checkpoint_dir in runs_directory.iterdir():
        if checkpoint_dir.is_dir():  # ensure it's a directory
            write_path = checkpoint_dir
            print(f"\n Processing: {checkpoint_dir}\n")
            load_and_write(write_path, checkpoint_dir)