
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

import torch
from lm_eval import evaluator

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === HF Hub Repository ===
HF_HUB_REPO = "TRI-ML/prismatic-vlms"

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
def load(
    write_checkpoint_path, model_id_or_path: Union[str, Path], cache_dir: Optional[Union[str, Path]] = None,
) -> PrismaticVLM:
    """Loads a pretrained PrismaticVLM from either local disk or the HuggingFace Hub."""
    hf_token: Union[str, Path] = Path(".hf_token") 
    hf_token = hf_token.read_text().strip() if isinstance(hf_token, Path) else os.environ[hf_token]

    if os.path.isdir(model_id_or_path):
        overwatch.info(f"Loading from local path `{(run_dir := Path(model_id_or_path))}`")

        # Get paths for `config.json` and pretrained checkpoint
        assert (config_json := run_dir / "config.json").exists(), f"Missing `config.json` for `{run_dir = }`"
        assert (checkpoint_pt := run_dir / "checkpoints" / "latest-checkpoint.pt").exists(), "Missing checkpoint!"
    else:
        if model_id_or_path not in GLOBAL_REGISTRY:
            raise ValueError(f"Couldn't find `{model_id_or_path = }; check `prismatic.available_model_names()`")

        overwatch.info(f"Downloading `{(model_id := GLOBAL_REGISTRY[model_id_or_path]['model_id'])} from HF Hub")
        config_json = hf_hub_download(repo_id=HF_HUB_REPO, filename=f"{model_id}/config.json", cache_dir=cache_dir)
        checkpoint_pt = hf_hub_download(
            repo_id=HF_HUB_REPO, filename=f"{model_id}/checkpoints/latest-checkpoint.pt", cache_dir=cache_dir
        )

    # Load Model Config from `config.json`
    with open(config_json, "r") as f:
        model_cfg = json.load(f)["model"]

    # = Load Individual Components necessary for Instantiating a VLM =
    #   =>> Print Minimal Config
    overwatch.info(
        f"Found Config =>> Loading & Freezing [bold blue]{model_cfg['model_id']}[/] with:\n"
        f"             Vision Backbone =>> [bold]{model_cfg['vision_backbone_id']}[/]\n"
        f"             LLM Backbone    =>> [bold]{model_cfg['llm_backbone_id']}[/]\n"
        f"             Arch Specifier  =>> [bold]{model_cfg['arch_specifier']}[/]\n"
        f"             Checkpoint Path =>> [underline]`{checkpoint_pt}`[/]"
    )

    # Load Vision Backbone
    overwatch.info(f"Loading Vision Backbone [bold]{model_cfg['vision_backbone_id']}[/]")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        model_cfg["vision_backbone_id"],
        model_cfg["image_resize_strategy"],
    )

    # Check if VLM checkpoint does not contain llm base weights. If not, then get_llm_backbone_and_tokenizer must load the default LLama2/Vicuna weights
    model_state_dict = torch.load(checkpoint_pt, map_location="cpu")["model"]
    load_from_hf_anyway = False
    if ("projector" in model_state_dict) and ("llm_backbone" not in model_state_dict):
        load_from_hf_anyway = True
    del model_state_dict

    # Load LLM Backbone --> note `inference_mode = True` by default when calling `load()`
    overwatch.info(f"Loading Pretrained LLM [bold]{model_cfg['llm_backbone_id']}[/] via HF Transformers")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        model_cfg["llm_backbone_id"],
        llm_max_length=model_cfg.get("llm_max_length", 2048),
        hf_token=hf_token,
        inference_mode=True,
        load_from_hf_anyway = load_from_hf_anyway
    )

    llm_hugface_save_path = os.path.join(
        write_checkpoint_path,
        "checkpoint_llm_only",
    )
    print(f'Saving to {llm_hugface_save_path}')
    
    llm_backbone.llm.save_pretrained(llm_hugface_save_path)
    tokenizer.save_pretrained(llm_hugface_save_path)
    overwatch.info(f"Writing LLM to checkpoint")

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

def main(write_path, checkpoint_path):
    llm_path = load(write_path, model_id_or_path=checkpoint_path)
    results = nlp_evaluation(llm_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process VLM checkpoint for NLP evaluation.")
    parser.add_argument('--write_path', type=str, required=True, help='Path to write the LLM checkpoint')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the VLM checkpoint')
    args = parser.parse_args()
    
    main(args.write_path, args.checkpoint_path)