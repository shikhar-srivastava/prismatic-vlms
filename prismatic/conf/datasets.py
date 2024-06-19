"""
datasets.py

Draccus Dataclass Definition for a DatasetConfig object, with various registered subclasses for each dataset variant
and processing scheme. A given dataset variant (e.g., `llava-lightning`) configures the following attributes:
    - Dataset Variant (Identifier) --> e.g., "llava-v15"
    - Align Stage Dataset Components (annotations, images)
    - Finetune Stage Dataset Components (annotations, images)
    - Dataset Root Directory (Path)
"""
from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import Tuple

from draccus import ChoiceRegistry


@dataclass
class DatasetConfig(ChoiceRegistry):
    # fmt: off
    dataset_id: str                                 # Unique ID that fully specifies a dataset variant

    # Dataset Components for each Stage in < align | finetune >
    align_stage_components: Tuple[Path, Path]       # Path to annotation file and images directory for `align` stage
    finetune_stage_components: Tuple[Path, Path]    # Path to annotation file and images directory for `finetune` stage

    dataset_root_dir: Path                          # Path to dataset root directory; others paths are relative to root
    # fmt: on


# [Reproduction] LLaVa-v15 (exact dataset used in all public LLaVa-v15 models)
@dataclass
class LLaVa_V15_Config(DatasetConfig):
    dataset_id: str = "llava-v15"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_mix665k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path("data")


# [Multimodal-Only] LLava-v15 WITHOUT the Language-Only ShareGPT Data (No Co-Training)
@dataclass
class LLaVa_Multimodal_Only_Config(DatasetConfig):
    dataset_id: str = "llava-multimodal"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_stripped625k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path("/mnt/fsx/skaramcheti/datasets/prismatic-vlms")


# LLaVa-v15 + LVIS-Instruct-4V
@dataclass
class LLaVa_LVIS4V_Config(DatasetConfig):
    dataset_id: str = "llava-lvis4v"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_lvis4v_mix888k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path("/mnt/fsx/skaramcheti/datasets/prismatic-vlms")


# LLaVa-v15 + LRV-Instruct
@dataclass
class LLaVa_LRV_Config(DatasetConfig):
    dataset_id: str = "llava-lrv"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_lrv_mix1008k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path("/mnt/fsx/skaramcheti/datasets/prismatic-vlms")


# LLaVa-v15 + LVIS-Instruct-4V + LRV-Instruct
@dataclass
class LLaVa_LVIS4V_LRV_Config(DatasetConfig):
    dataset_id: str = "llava-lvis4v-lrv"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_lvis4v_lrv_mix1231k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path("/mnt/fsx/skaramcheti/datasets/prismatic-vlms")



@dataclass
class LLaVa_V1_Config(DatasetConfig):
    dataset_id: str = "llava-v1"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("continual/llava-158k-instruct/llava_instruct_150k-cleaned.json"), # Path to json with annotations
        Path("download/llava-v1.5-instruct/"), # Base path to image directories
    )
    dataset_root_dir: Path = Path("data")

@dataclass
class LLAVA_V1_VQAV2_Config(DatasetConfig):
    dataset_id: str = "llava-vqav2"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("continual/splits/vqa-v2_82787k.json"), # Path to json with annotations
        Path("download/llava-v1.5-instruct/"), # Base path to image directories
    )
    dataset_root_dir: Path = Path("data")


@dataclass
class LLAVA_V1_INSTRUCT_VQA_ALL_Config(DatasetConfig):
    dataset_id: str = "llava-instruct-vqa-all"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("continual/combined/instruct_vqa_all.json"), # Path to json with annotations
        Path("download/llava-v1.5-instruct/"), # Base path to image directories
    )
    dataset_root_dir: Path = Path("data")

@dataclass
class LLAVA_V1_VQA_ALL_Config(DatasetConfig):
    dataset_id: str = "llava-vqa-all"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("continual/combined/vqa_all.json"), # Path to json with annotations
        Path("download/llava-v1.5-instruct/"), # Base path to image directories
    )
    dataset_root_dir: Path = Path("data")


# ==========================================
## ========= LLAVA CL DATASETS ===========
@dataclass
class INSTRUCT_Config(DatasetConfig):
    dataset_id: str = "instruct"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("continual/combined/instruct.json"), # Path to json with annotations
        Path("download/llava-v1.5-instruct/"), # Base path to image directories
    )
    dataset_root_dir: Path = Path("data")

@dataclass
class VQA_Config(DatasetConfig):

    dataset_id: str = "vqa"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("continual/combined/vqa.json"), # Path to json with annotations
        Path("download/llava-v1.5-instruct/"), # Base path to image directories
    )
    dataset_root_dir: Path = Path("data")

@dataclass
class OCR_Config(DatasetConfig):
    dataset_id: str = "ocr"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("continual/combined/ocr.json"), # Path to json with annotations
        Path("download/llava-v1.5-instruct/"), # Base path to image directories
    )
    dataset_root_dir: Path = Path("data")


@dataclass
class Ref_Config(DatasetConfig):
    dataset_id: str = "ref"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("continual/combined/ref.json"), # Path to json with annotations
        Path("download/llava-v1.5-instruct/"), # Base path to image directories
    )
    dataset_root_dir: Path = Path("data")


## ========= REHEARSAL DATASETS ===========
# ==========================================
# REHEARSAL SEQEUENCE (Version 1: Partial LLaVA CL)
@dataclass
class VO_Rehearse_R_Train_p1_Config(DatasetConfig):
    dataset_id: str = "vo_rehearse_r_train_p1"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("continual/rehearsal/rehearse_vqa-v2_82787k_rehearse_ocrvqa_80000k_train_refcoco_48447k_0.1.json"), # Path to json with annotations
        Path("download/llava-v1.5-instruct/"), # Base path to image directories
    )
    dataset_root_dir: Path = Path("data")

@dataclass
class VO_Rehearse_R_Train_1_Config(DatasetConfig):
    dataset_id: str = "vo_rehearse_r_train_1"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("continual/rehearsal/rehearse_vqa-v2_82787k_rehearse_ocrvqa_80000k_train_refcoco_48447k_1.json"), # Path to json with annotations
        Path("download/llava-v1.5-instruct/"), # Base path to image directories
    )
    dataset_root_dir: Path = Path("data")


@dataclass
class VO_Rehearse_R_Train_10_Config(DatasetConfig):
    dataset_id: str = "vo_rehearse_r_train_10"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("continual/rehearsal/rehearse_vqa-v2_82787k_rehearse_ocrvqa_80000k_train_refcoco_48447k_10.json"), # Path to json with annotations
        Path("download/llava-v1.5-instruct/"), # Base path to image directories
    )
    dataset_root_dir: Path = Path("data")

@dataclass
class V_Rehearse_O_Train_p1_Config(DatasetConfig):
    dataset_id: str = "v_rehearse_o_train_p1"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("continual/rehearsal/rehearse_vqa-v2_82787k_train_ocrvqa_80000k_0.1.json"), # Path to json with annotations
        Path("download/llava-v1.5-instruct/"), # Base path to image directories
    )
    dataset_root_dir: Path = Path("data")

# @dataclass
# class V_Rehearse_O_Train_1_Config(DatasetConfig):
#     dataset_id: str = "v_rehearse_o_train_1"

#     align_stage_components: Tuple[Path, Path] = (
#         Path("download/llava-laion-cc-sbu-558k/chat.json"),
#         Path("download/llava-laion-cc-sbu-558k/"),
#     )
#     finetune_stage_components: Tuple[Path, Path] = (
#         Path("continual/rehearsal/rehearse_vqa-v2_82787k_train_ocrvqa_80000k_1.json"), # Path to json with annotations
#         Path("download/llava-v1.5-instruct/"), # Base path to image directories
#     )
#     dataset_root_dir: Path = Path("data")

@dataclass
class V_Rehearse_O_Train_10_Config(DatasetConfig):
    dataset_id: str = "v_rehearse_o_train_10"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("continual/rehearsal/rehearse_vqa-v2_82787k_train_ocrvqa_80000k_10.json"), # Path to json with annotations
        Path("download/llava-v1.5-instruct/"), # Base path to image directories
    )
    dataset_root_dir: Path = Path("data")

# ========================================
## == Rehearsal Datasets for FULL LLaVA CL

# ======= Datasets for rehearsals ========
#
#   Naming convention: \
#       Current Sequence of Datasets: IVOR [I(Instruct)V(VQA)O(OCR)R(Referential-Expression)]
#       Dataset: {Datasets rehearsed}_Rehearse_{Current dataset trained on}_Train_{sampling rate %}_Config

#  Rehearsal Sampling Rate: 0.1% ('p1')
@dataclass
class I_Rehearse_V_Train_p1_Config(DatasetConfig):
    dataset_id: str = "i_rehearse_v_train_p1"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("continual/rehearsal/ivor_0.1/rehearse_instruct_train_vqa_0.1.json"), # Path to json with annotations
        Path("download/llava-v1.5-instruct/"), # Base path to image directories
    )
    dataset_root_dir: Path = Path("data")

@dataclass
class IV_Rehearse_O_Train_p1_Config(DatasetConfig):
    dataset_id: str = "iv_rehearse_o_train_p1"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("continual/rehearsal/ivor_0.1/rehearse_instruct_vqa_train_ocr_0.1.json"), # Path to json with annotations
        Path("download/llava-v1.5-instruct/"), # Base path to image directories
    )
    dataset_root_dir: Path = Path("data")

@dataclass
class IVO_Rehearse_R_Train_p1_Config(DatasetConfig):
    dataset_id: str = "ivo_rehearse_r_train_p1"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("continual/rehearsal/ivor_0.1/rehearse_instruct_vqa_ocr_train_ref_0.1.json"), # Path to json with annotations
        Path("download/llava-v1.5-instruct/"), # Base path to image directories
    )
    dataset_root_dir: Path = Path("data")

# Rehearsal Sampling Rate: 1% ('1')
@dataclass
class I_Rehearse_V_Train_1_Config(DatasetConfig):
    dataset_id: str = "i_rehearse_v_train_1"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("continual/rehearsal/ivor_1/rehearse_instruct_train_vqa_1.json"), # Path to json with annotations
        Path("download/llava-v1.5-instruct/"), # Base path to image directories
    )
    dataset_root_dir: Path = Path("data")

@dataclass
class IV_Rehearse_O_Train_1_Config(DatasetConfig):
    dataset_id: str = "iv_rehearse_o_train_1"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("continual/rehearsal/ivor_1/rehearse_instruct_vqa_train_ocr_1.json"), # Path to json with annotations
        Path("download/llava-v1.5-instruct/"), # Base path to image directories
    )
    dataset_root_dir: Path = Path("data")

@dataclass
class IVO_Rehearse_R_Train_1_Config(DatasetConfig):
    dataset_id: str = "ivo_rehearse_r_train_1"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("continual/rehearsal/ivor_1/rehearse_instruct_vqa_ocr_train_ref_1.json"), # Path to json with annotations
        Path("download/llava-v1.5-instruct/"), # Base path to image directories
    )
    dataset_root_dir: Path = Path("data")

# Rehearsal Sampling Rate: 10% ('10')
@dataclass
class I_Rehearse_V_Train_10_Config(DatasetConfig):
    dataset_id: str = "i_rehearse_v_train_10"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("continual/rehearsal/ivor_10/rehearse_instruct_train_vqa_10.json"), # Path to json with annotations
        Path("download/llava-v1.5-instruct/"), # Base path to image directories
    )
    dataset_root_dir: Path = Path("data")

@dataclass
class IV_Rehearse_O_Train_10_Config(DatasetConfig):
    dataset_id: str = "iv_rehearse_o_train_10"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("continual/rehearsal/ivor_10/rehearse_instruct_vqa_train_ocr_10.json"), # Path to json with annotations
        Path("download/llava-v1.5-instruct/"), # Base path to image directories
    )
    dataset_root_dir: Path = Path("data")

@dataclass
class IVO_Rehearse_R_Train_10_Config(DatasetConfig):
    dataset_id: str = "ivo_rehearse_r_train_10"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("continual/rehearsal/ivor_10/rehearse_instruct_vqa_ocr_train_ref_10.json"), # Path to json with annotations
        Path("download/llava-v1.5-instruct/"), # Base path to image directories
    )
    dataset_root_dir: Path = Path("data")


# Rehearsal Sampling Rate: 10% ('10')
@dataclass
class I_Rehearse_V_Train_20_Config(DatasetConfig):
    dataset_id: str = "i_rehearse_v_train_20"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("continual/rehearsal/ivor_20/rehearse_instruct_train_vqa_20.json"), # Path to json with annotations
        Path("download/llava-v1.5-instruct/"), # Base path to image directories
    )
    dataset_root_dir: Path = Path("data")

@dataclass
class IV_Rehearse_O_Train_20_Config(DatasetConfig):
    dataset_id: str = "iv_rehearse_o_train_20"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("continual/rehearsal/ivor_20/rehearse_instruct_vqa_train_ocr_20.json"), # Path to json with annotations
        Path("download/llava-v1.5-instruct/"), # Base path to image directories
    )
    dataset_root_dir: Path = Path("data")

@dataclass
class IVO_Rehearse_R_Train_20_Config(DatasetConfig):
    dataset_id: str = "ivo_rehearse_r_train_20"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("continual/rehearsal/ivor_20/rehearse_instruct_vqa_ocr_train_ref_20.json"), # Path to json with annotations
        Path("download/llava-v1.5-instruct/"), # Base path to image directories
    )
    dataset_root_dir: Path = Path("data")


# Current Sequence of Datasets: IORV [I(Instruct)O(OCR)R(Referential-Expression)V(VQA)]
# 1 (1%)
 
@dataclass
class I_Rehearse_O_Train_1_Config(DatasetConfig):
    dataset_id: str = "i_rehearse_o_train_1"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("continual/rehearsal/iorv_1/rehearse_instruct_train_ocr_1.json"), # Path to json with annotations
        Path("download/llava-v1.5-instruct/"), # Base path to image directories
    )
    dataset_root_dir: Path = Path("data")

@dataclass
class IO_Rehearse_R_Train_1_Config(DatasetConfig):
    dataset_id: str = "io_rehearse_r_train_1"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("continual/rehearsal/iorv_1/rehearse_instruct_ocr_train_ref_1.json"), # Path to json with annotations
        Path("download/llava-v1.5-instruct/"), # Base path to image directories
    )
    dataset_root_dir: Path = Path("data")

@dataclass
class IOR_Rehearse_V_Train_1_Config(DatasetConfig):
    dataset_id: str = "ior_rehearse_v_train_1"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("continual/rehearsal/iorv_1/rehearse_instruct_ocr_ref_train_vqa_1.json"), # Path to json with annotations
        Path("download/llava-v1.5-instruct/"), # Base path to image directories
    )
    dataset_root_dir: Path = Path("data")


# # Current Sequence of Datasets: VOIR [V(VQA)O(OCR)I(Instruct)R(Referential-Expression)]
# # 1 (1%)
    
@dataclass
class V_Rehearse_O_Train_1_Config(DatasetConfig):
    dataset_id: str = "v_rehearse_o_train_1"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("continual/rehearsal/voir_1/rehearse_vqa_train_ocr_1.json"), # Path to json with annotations
        Path("download/llava-v1.5-instruct/"), # Base path to image directories
    )
    dataset_root_dir: Path = Path("data")

@dataclass
class VO_Rehearse_I_Train_1_Config(DatasetConfig):
    dataset_id: str = "vo_rehearse_i_train_1"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("continual/rehearsal/voir_1/rehearse_vqa_ocr_train_instruct_1.json"), # Path to json with annotations
        Path("download/llava-v1.5-instruct/"), # Base path to image directories
    )
    dataset_root_dir: Path = Path("data")

@dataclass
class VOI_Rehearse_R_Train_1_Config(DatasetConfig):
    dataset_id: str = "voi_rehearse_r_train_1"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("continual/rehearsal/voir_1/rehearse_vqa_ocr_instruct_train_ref_1.json"), # Path to json with annotations
        Path("download/llava-v1.5-instruct/"), # Base path to image directories
    )
    dataset_root_dir: Path = Path("data")

# ========================================
@dataclass
class Test(DatasetConfig):
    dataset_id: str = "test"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("llava_instruct_150k-cleaned-256.json"), # Path to json with annotations
        Path("download/llava-v1.5-instruct/"), # Base path to image directories
    )
    dataset_root_dir: Path = Path("data")



# === Define a Dataset Registry Enum for Reference & Validation =>> all *new* datasets must be added here! ===
@unique
class DatasetRegistry(Enum):

    # LLAVA FULL CL 
    INSTRUCT = INSTRUCT_Config
    VQA = VQA_Config
    REF = Ref_Config
    OCR = OCR_Config
    # LLAVA PARTIAL CL    
    LLAVA_VQA_ALL  = LLAVA_V1_VQA_ALL_Config
    LLAVA_INSTRUCT_VQA_ALL = LLAVA_V1_INSTRUCT_VQA_ALL_Config
    LLAVA_VQA_V2 = LLAVA_V1_VQAV2_Config

    # === Rehearsal Datasets ===
    VO_REHEARSE_R_TRAIN_P1 = VO_Rehearse_R_Train_p1_Config
    VO_REHEARSE_R_TRAIN_1 = VO_Rehearse_R_Train_1_Config
    VO_REHEARSE_R_TRAIN_10 = VO_Rehearse_R_Train_10_Config
    
    V_REHEARSE_O_TRAIN_P1 = V_Rehearse_O_Train_p1_Config
    #V_REHEARSE_O_TRAIN_1 = V_Rehearse_O_Train_1_Config
    V_REHEARSE_O_TRAIN_10 = V_Rehearse_O_Train_10_Config

    # === Rehearsal Datasets for FULL LLaVA CL ===
    I_REHEARSE_V_TRAIN_P1 = I_Rehearse_V_Train_p1_Config
    IV_REHEARSE_O_TRAIN_P1 = IV_Rehearse_O_Train_p1_Config
    IVO_REHEARSE_R_TRAIN_P1 = IVO_Rehearse_R_Train_p1_Config

    I_REHEARSE_V_TRAIN_1 = I_Rehearse_V_Train_1_Config
    IV_REHEARSE_O_TRAIN_1 = IV_Rehearse_O_Train_1_Config
    IVO_REHEARSE_R_TRAIN_1 = IVO_Rehearse_R_Train_1_Config

    I_REHEARSE_V_TRAIN_10 = I_Rehearse_V_Train_10_Config
    IV_REHEARSE_O_TRAIN_10 = IV_Rehearse_O_Train_10_Config
    IVO_REHEARSE_R_TRAIN_10 = IVO_Rehearse_R_Train_10_Config


    I_REHEARSE_V_TRAIN_20 = I_Rehearse_V_Train_20_Config
    IV_REHEARSE_O_TRAIN_20 = IV_Rehearse_O_Train_20_Config
    IVO_REHEARSE_R_TRAIN_20 = IVO_Rehearse_R_Train_20_Config


    # IORV 
    I_REHEARSE_O_TRAIN_1 = I_Rehearse_O_Train_1_Config
    IO_REHEARSE_R_TRAIN_1 = IO_Rehearse_R_Train_1_Config
    IOR_REHEARSE_V_TRAIN_1 = IOR_Rehearse_V_Train_1_Config

    # VOIR
    V_REHEARSE_O_TRAIN_1 = V_Rehearse_O_Train_1_Config
    VO_REHEARSE_I_TRAIN_1 = VO_Rehearse_I_Train_1_Config
    VOI_REHEARSE_R_TRAIN_1 = VOI_Rehearse_R_Train_1_Config



    TEST_DATA = Test
    # === LLaVa v1 ===
    LLAVA_V1 = LLaVa_V1_Config

    # === LLaVa v1.5 ===
    LLAVA_V15 = LLaVa_V15_Config

    LLAVA_MULTIMODAL_ONLY = LLaVa_Multimodal_Only_Config

    LLAVA_LVIS4V = LLaVa_LVIS4V_Config
    LLAVA_LRV = LLaVa_LRV_Config

    LLAVA_LVIS4V_LRV = LLaVa_LVIS4V_LRV_Config

    @property
    def dataset_id(self) -> str:
        return self.value.dataset_id


# Register Datasets in Choice Registry
for dataset_variant in DatasetRegistry:
    DatasetConfig.register_subclass(dataset_variant.dataset_id, dataset_variant.value)
