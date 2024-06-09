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

@dataclass
class V_Rehearse_O_Train_1_Config(DatasetConfig):
    dataset_id: str = "v_rehearse_o_train_1"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("continual/rehearsal/rehearse_vqa-v2_82787k_train_ocrvqa_80000k_1.json"), # Path to json with annotations
        Path("download/llava-v1.5-instruct/"), # Base path to image directories
    )
    dataset_root_dir: Path = Path("data")

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

    # === Rehearsal Datasets ===
    VO_REHEARSE_R_TRAIN_P1 = VO_Rehearse_R_Train_p1_Config
    VO_REHEARSE_R_TRAIN_1 = VO_Rehearse_R_Train_1_Config
    VO_REHEARSE_R_TRAIN_10 = VO_Rehearse_R_Train_10_Config
    
    V_REHEARSE_O_TRAIN_P1 = V_Rehearse_O_Train_p1_Config
    V_REHEARSE_O_TRAIN_1 = V_Rehearse_O_Train_1_Config
    V_REHEARSE_O_TRAIN_10 = V_Rehearse_O_Train_10_Config

    # CL 
    INSTRUCT = INSTRUCT_Config
    VQA = VQA_Config
    REF = Ref_Config
    OCR = OCR_Config

    LLAVA_VQA_ALL  = LLAVA_V1_VQA_ALL_Config
    LLAVA_INSTRUCT_VQA_ALL = LLAVA_V1_INSTRUCT_VQA_ALL_Config
    LLAVA_VQA_V2 = LLAVA_V1_VQAV2_Config




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
