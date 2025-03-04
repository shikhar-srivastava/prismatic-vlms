import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Tuple, Union
import jsonlines
import numpy as np
import torch
import wandb
from prismatic.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

# === Define Tracker Interface ===
class Tracker(Protocol):
    def write_hyperparameters(self) -> None: ...
    def write(self, global_step: int, metrics: Dict[str, Union[int, float]]) -> None: ...
    def finalize(self) -> None: ...

# === Individual Tracker Definitions ===
class JSONLinesTracker:
    def __init__(self, run_id: str, run_dir: Path, hparams: Dict[str, Any]) -> None:
        self.run_id, self.run_dir, self.hparams = run_id, run_dir, hparams

    @overwatch.rank_zero_only
    def write_hyperparameters(self) -> None:
        with jsonlines.open(self.run_dir / "run-metrics.jsonl", mode="w", sort_keys=True) as js_tracker:
            js_tracker.write({"run_id": self.run_id, "hparams": self.hparams})

    @overwatch.rank_zero_only
    def write(self, _: int, metrics: Dict[str, Union[int, float]]) -> None:
        with jsonlines.open(self.run_dir / f"{self.run_id}.jsonl", mode="a", sort_keys=True) as js_tracker:
            js_tracker.write(metrics)

    def finalize(self) -> None:
        return

class WeightsBiasesTracker:
    def __init__(
        self,
        run_id: str,
        run_dir: Path,
        hparams: Dict[str, Any],
        project: str = "prismatic",
        entity: Optional[str] = None,
        group: str = "align",
    ) -> None:
        self.run_id, self.run_dir, self.hparams = run_id, run_dir, hparams
        self.project, self.entity, self.group, self.wandb_dir = project, entity, group, self.run_dir
        self.initialize()

    @overwatch.rank_zero_only
    def initialize(self) -> None:
        wandb.init(
            name=self.run_id,
            dir=self.wandb_dir,
            config=self.hparams,
            project=self.project,
            entity=self.entity,
            group=self.group,
        )

    @overwatch.rank_zero_only
    def write_hyperparameters(self) -> None:
        wandb.config = self.hparams

    @overwatch.rank_zero_only
    def write(self, global_step: int, metrics: Dict[str, Union[int, float]]) -> None:
        wandb.log(metrics, step=global_step)

    @staticmethod
    def finalize() -> None:
        if overwatch.is_rank_zero():
            wandb.finish()
        time.sleep(30)

# === Core Metrics Container ===
class Metrics:
    def __init__(
        self,
        active_trackers: Tuple[str, ...],
        run_id: str,
        run_dir: Path,
        hparams: Dict[str, Any],
        stage: str,
        wandb_project: str = "prismatic",
        wandb_entity: Optional[str] = None,
        grad_accumulation_steps: int = 1,
        window_size: int = 128,
    ) -> None:
        self.run_id, self.run_dir, self.hparams, self.stage = run_id, run_dir, hparams, stage

        # Initialize Trackers
        self.trackers = []
        for tracker_type in active_trackers:
            if tracker_type == "jsonl":
                tracker = JSONLinesTracker(run_id, run_dir, hparams)
            elif tracker_type == "wandb":
                tracker = WeightsBiasesTracker(
                    run_id, run_dir, hparams, project=wandb_project, entity=wandb_entity, group=self.stage
                )
            else:
                raise ValueError(f"Tracker with type `{tracker_type}` is not supported!")

            tracker.write_hyperparameters()
            self.trackers.append(tracker)

        # Create Universal Metrics Buffers
        self.global_step, self.start_time, self.step_start_time = 0, time.time(), time.time()
        self.state = {
            "loss_raw": deque(maxlen=grad_accumulation_steps),
            "loss": deque(maxlen=window_size),
            "step_time": deque(maxlen=window_size),
            "lr": [],
            "lora_plasticity": None,
            "lora_plasticity_first": None,
            "lora_weight_changes": {},
            "lora_weight_changes_first": {},
            "ft_total_weight_change": None,
            "ft_layer_weight_changes": {},
            "ft_parameter_weight_changes": {},
            "rank_entropy": deque(maxlen=window_size),
            # *** NEW *** embedding stats
            "vis_embed_mean": deque(maxlen=window_size),
            "vis_embed_std": deque(maxlen=window_size),
            "vis_l2_mean": deque(maxlen=window_size),
            "vis_l2_std": deque(maxlen=window_size),
            "txt_embed_mean": deque(maxlen=window_size),
            "txt_embed_std": deque(maxlen=window_size),
            "txt_l2_mean": deque(maxlen=window_size),
            "txt_l2_std": deque(maxlen=window_size),
        }

    def log(self, global_step: int, metrics: Dict[str, Union[int, float]]) -> None:
        for tracker in self.trackers:
            tracker.write(global_step, metrics)

    def get_status(self, loss: Optional[torch.Tensor] = None) -> str:
        lr = self.state["lr"][-1] if len(self.state["lr"]) > 0 else 0
        if loss is None:
            return f"=>> [Global Step] {self.global_step:06d} =>> LR :: {lr:.6f}"
        return f"=>> [Global Step] {self.global_step:06d} =>> LR :: {lr:.6f} -- Loss :: {loss:.4f}"

    def commit(
        self,
        *,
        global_step: Optional[int] = None,
        lr: Optional[float] = None,
        update_step_time: bool = False,
        lora_plasticity: Optional[float] = None,
        lora_plasticity_first: Optional[float] = None,
        lora_weight_changes: Optional[Dict[str, float]] = None,
        lora_weight_changes_first: Optional[Dict[str, float]] = None,
        ft_total_weight_change: Optional[float] = None,
        ft_layer_weight_changes: Optional[Dict[str, Dict[str, float]]] = None,
        ft_parameter_weight_changes: Optional[Dict[str, float]] = None,
        rank_entropy: Optional[float] = None,
        # *** NEW *** optional embedding stats
        vis_embed_mean: Optional[float] = None,
        vis_embed_std: Optional[float] = None,
        vis_l2_mean: Optional[float] = None,
        vis_l2_std: Optional[float] = None,
        txt_embed_mean: Optional[float] = None,
        txt_embed_std: Optional[float] = None,
        txt_l2_mean: Optional[float] = None,
        txt_l2_std: Optional[float] = None,
        **kwargs
    ) -> None:
        if global_step is not None:
            self.global_step = global_step

        # Only track on rank zero
        if not overwatch.is_rank_zero():
            return

        # Basic logging
        if lr is not None:
            self.state["lr"].append(lr)

        if update_step_time:
            self.state["step_time"].append(time.time() - self.step_start_time)
            self.step_start_time = time.time()

        # Lora / FT tracking
        if lora_plasticity is not None:
            self.state["lora_plasticity"] = lora_plasticity
        if lora_plasticity_first is not None:
            self.state["lora_plasticity_first"] = lora_plasticity_first
        if lora_weight_changes is not None:
            self.state["lora_weight_changes"] = lora_weight_changes
        if lora_weight_changes_first is not None:
            self.state["lora_weight_changes_first"] = lora_weight_changes_first

        if ft_total_weight_change is not None:
            self.state["ft_total_weight_change"] = ft_total_weight_change
        if ft_layer_weight_changes is not None:
            self.state["ft_layer_weight_changes"] = ft_layer_weight_changes
        if ft_parameter_weight_changes is not None:
            self.state["ft_parameter_weight_changes"] = ft_parameter_weight_changes
        if rank_entropy is not None:
            self.state["rank_entropy"].append(rank_entropy)

        # NEW: embedding stats
        if vis_embed_mean is not None:
            self.state["vis_embed_mean"].append(torch.tensor(vis_embed_mean))
        if vis_embed_std is not None:
            self.state["vis_embed_std"].append(torch.tensor(vis_embed_std))
        if vis_l2_mean is not None:
            self.state["vis_l2_mean"].append(torch.tensor(vis_l2_mean))
        if vis_l2_std is not None:
            self.state["vis_l2_std"].append(torch.tensor(vis_l2_std))

        if txt_embed_mean is not None:
            self.state["txt_embed_mean"].append(torch.tensor(txt_embed_mean))
        if txt_embed_std is not None:
            self.state["txt_embed_std"].append(torch.tensor(txt_embed_std))
        if txt_l2_mean is not None:
            self.state["txt_l2_mean"].append(torch.tensor(txt_l2_mean))
        if txt_l2_std is not None:
            self.state["txt_l2_std"].append(torch.tensor(txt_l2_std))

        # Generic key-value arguments for e.g. "loss"
        for key, value in kwargs.items():
            if key == "loss":
                loss_val = value.detach()
                self.state["loss_raw"].append(loss_val)
                self.state["loss"].append(loss_val)
            else:
                self.state[key].append(value.detach())

    @overwatch.rank_zero_only
    def push(self) -> str:
        prefix = self.stage.capitalize()

        # 1) Loss stats
        loss_raw = torch.stack(list(self.state["loss_raw"])).mean().item() if len(self.state["loss_raw"]) > 0 else 0.0
        loss = torch.stack(list(self.state["loss"])).mean().item() if len(self.state["loss"]) > 0 else 0.0
        step_time = np.mean(list(self.state["step_time"])) if len(self.state["step_time"]) > 0 else 0.0
        lr = self.state["lr"][-1] if len(self.state["lr"]) > 0 else 0.0

        status = self.get_status(torch.tensor(loss))

        metrics = {
            f"{prefix}/Step": self.global_step,
            f"{prefix}/Loss": loss,
            f"{prefix}/Loss (Raw)": loss_raw,
            f"{prefix}/Learning Rate": lr,
            f"{prefix}/Step Time": step_time,
        }

        # 2) rank entropy
        if len(self.state["rank_entropy"]) > 0:
            avg_rank_entropy = np.mean(self.state["rank_entropy"])
            metrics[f"{prefix}/Rank Entropy"] = avg_rank_entropy
            self.state["rank_entropy"].clear()

        # 3) log LoRA plasticity
        if self.state["lora_plasticity"] is not None:
            metrics[f"{prefix}/LoRA Plasticity"] = self.state["lora_plasticity"]
            self.state["lora_plasticity"] = None
        if self.state["lora_plasticity_first"] is not None:
            metrics[f"{prefix}/LoRA Plasticity First"] = self.state["lora_plasticity_first"]
            self.state["lora_plasticity_first"] = None

        if self.state["lora_weight_changes"]:
            for layer, avg_change in self.state["lora_weight_changes"].items():
                metrics[f"{prefix}/LoRA Weight Change Layer {layer}"] = avg_change
            self.state["lora_weight_changes"] = {}
        if self.state["lora_weight_changes_first"]:
            for layer, avg_change in self.state["lora_weight_changes_first"].items():
                metrics[f"{prefix}/LoRA Weight Change First Layer {layer}"] = avg_change
            self.state["lora_weight_changes_first"] = {}

        # 4) log FT changes
        if self.state["ft_total_weight_change"] is not None:
            metrics[f"{prefix}/FT Total Weight Change"] = self.state["ft_total_weight_change"]
            self.state["ft_total_weight_change"] = None
        if self.state["ft_layer_weight_changes"]:
            for layer, changes in self.state["ft_layer_weight_changes"].items():
                # changes is usually { 'layer': float, 'attention': float, 'mlp': float }
                metrics[f"{prefix}/FT Weight Change Layer {layer}"] = changes['layer']
                metrics[f"{prefix}/FT Weight Change Attention Layer {layer}"] = changes['attention']
                metrics[f"{prefix}/FT Weight Change MLP Layer {layer}"] = changes['mlp']
            self.state["ft_layer_weight_changes"] = {}
        if self.state["ft_parameter_weight_changes"]:
            for param_name, param_change in self.state["ft_parameter_weight_changes"].items():
                metrics[f"{prefix}/FT Weight Change Param {param_name}"] = param_change
            self.state["ft_parameter_weight_changes"] = {}

        # *** NEW *** 5) Log embedding stats
        def safe_mean_of_deque(dq):
            if len(dq) == 0:
                return 0.0
            vals = torch.stack(list(dq))
            return vals.mean().item()

        # Visual
        vm = safe_mean_of_deque(self.state["vis_embed_mean"])
        vs = safe_mean_of_deque(self.state["vis_embed_std"])
        vl2m = safe_mean_of_deque(self.state["vis_l2_mean"])
        vl2s = safe_mean_of_deque(self.state["vis_l2_std"])
        metrics[f"{prefix}/VisEmbedMean"] = vm
        metrics[f"{prefix}/VisEmbedStd"] = vs
        metrics[f"{prefix}/VisL2Mean"] = vl2m
        metrics[f"{prefix}/VisL2Std"] = vl2s
        self.state["vis_embed_mean"].clear()
        self.state["vis_embed_std"].clear()
        self.state["vis_l2_mean"].clear()
        self.state["vis_l2_std"].clear()

        # Text
        tm = safe_mean_of_deque(self.state["txt_embed_mean"])
        ts = safe_mean_of_deque(self.state["txt_embed_std"])
        tl2m = safe_mean_of_deque(self.state["txt_l2_mean"])
        tl2s = safe_mean_of_deque(self.state["txt_l2_std"])
        metrics[f"{prefix}/TxtEmbedMean"] = tm
        metrics[f"{prefix}/TxtEmbedStd"] = ts
        metrics[f"{prefix}/TxtL2Mean"] = tl2m
        metrics[f"{prefix}/TxtL2Std"] = tl2s
        self.state["txt_embed_mean"].clear()
        self.state["txt_embed_std"].clear()
        self.state["txt_l2_mean"].clear()
        self.state["txt_l2_std"].clear()

        # Actually log them to trackers
        self.log(self.global_step, metrics)
        return status

    def finalize(self) -> None:
        for tracker in self.trackers:
            tracker.finalize()