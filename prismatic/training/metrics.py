import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, Deque
from collections import deque
import jsonlines
import numpy as np
import torch
import wandb
from prismatic.overwatch import initialize_overwatch
import matplotlib.pyplot as plt

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

# === Define Tracker Interface ===
class Tracker(Protocol):
    def write_hyperparameters(self) -> None: ...
    def write(self, global_step: int, metrics: Dict[str, Union[int, float, torch.Tensor]]) -> None: ...
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
    def write(self, _: int, metrics: Dict[str, Union[int, float, torch.Tensor]]) -> None:
        # Process any tensor objects for JSON serialization
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                # Skip complex tensor objects that can't be easily serialized
                if key in ["projected_embeddings_histogram", "covariance_matrix"]:
                    continue
                
                # Handle scalar tensors
                if value.numel() == 1:
                    serializable_metrics[key] = value.item()
                else:
                    # For multi-dimensional tensors, we'll just store stats to avoid huge files
                    serializable_metrics[f"{key}_shape"] = list(value.shape)
                    serializable_metrics[f"{key}_mean"] = float(value.mean().item())
                    serializable_metrics[f"{key}_std"] = float(value.std().item())
            else:
                serializable_metrics[key] = value
        
        with jsonlines.open(self.run_dir / f"{self.run_id}.jsonl", mode="a", sort_keys=True) as js_tracker:
            js_tracker.write(serializable_metrics)

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
    def write(self, global_step: int, metrics: Dict[str, Union[int, float, torch.Tensor]]) -> None:
        processed_metrics = {}
        
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
                
                if key == "covariance_matrix":
                    try:
                        # Convert to float32 to avoid unsupported type error
                        matrix = value.to(torch.float32).numpy()
                        
                        # If the matrix is too large, downsample it
                        max_size = 100  # Maximum size for visualization
                        if matrix.shape[0] > max_size:
                            indices = np.linspace(0, matrix.shape[0]-1, max_size, dtype=int)
                            matrix = matrix[indices][:, indices]
                        
                        # Create heatmap
                        fig, ax = plt.subplots(figsize=(16, 12))
                        im = ax.imshow(matrix, cmap="viridis")
                        plt.colorbar(im, ax=ax)
                        ax.set_title("Embedding Covariance Matrix")
                        
                        # Log as a WandB image
                        processed_metrics["covariance_matrix_heatmap"] = wandb.Image(fig)
                        plt.close(fig)
                        
                        # Log statistics about the covariance matrix
                        processed_metrics["covariance_matrix_mean"] = float(matrix.mean())
                        processed_metrics["covariance_matrix_std"] = float(matrix.std())
                        processed_metrics["covariance_matrix_min"] = float(matrix.min())
                        processed_metrics["covariance_matrix_max"] = float(matrix.max())
                    except Exception as e:
                        overwatch.warning(f"Failed to log covariance matrix: {e}")
                elif key == "projected_embeddings_histogram":
                    try:
                        data = value.numpy()  # 1D array of mean values
                        processed_metrics["projected_embeddings_histogram"] = wandb.Histogram(data)
                    except Exception as e:
                        overwatch.warning(f"Failed to log histogram for {key}: {e}")
                
                elif key == "projected_embeddings_values":
                    try:
                        values = value.numpy()
                        fig, ax = plt.subplots(figsize=(16, 12))
                        ax.plot(values, linewidth=1.0)
                        ax.set_xlabel("Dimension Index")
                        ax.set_ylabel("Batch-Averaged Value")
                        ax.set_title("Batch-Averaged Projected Visual Embeddings per Dimension")
                        ax.grid(True, linestyle="--", alpha=0.7)
                        processed_metrics["projected_embeddings_values_plot"] = wandb.Image(fig)
                        plt.close(fig)
                    except Exception as e:
                        overwatch.warning(f"Failed to log plot for {key}: {e}")

                # Handle other tensor cases (e.g., histograms) as needed
                else:
                    if value.numel() == 1:
                        processed_metrics[key] = value.item()
                    else:
                        try:
                            if value.numel() <= 1000000:
                                value_np = value.numpy()
                                processed_metrics[f"{key}_mean"] = float(np.mean(value_np))
                                processed_metrics[f"{key}_std"] = float(np.std(value_np))
                                processed_metrics[f"{key}_min"] = float(np.min(value_np))
                                processed_metrics[f"{key}_max"] = float(np.max(value_np))
                            else:
                                overwatch.warning(f"Tensor {key} too large to log: {value.shape}")
                        except Exception as e:
                            overwatch.warning(f"Failed to process tensor {key}: {e}")
            else:
                processed_metrics[key] = value
        
        wandb.log(processed_metrics, step=global_step)


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
            "vis_embed_mean": deque(maxlen=window_size),
            "vis_embed_std": deque(maxlen=window_size),
            "vis_l2_mean": deque(maxlen=window_size),
            "vis_l2_std": deque(maxlen=window_size),
            "txt_embed_mean": deque(maxlen=window_size),
            "txt_embed_std": deque(maxlen=window_size),
            "txt_l2_mean": deque(maxlen=window_size),
            "txt_l2_std": deque(maxlen=window_size),
            "vis_txt_cosine_min": deque(maxlen=window_size),
            "vis_txt_cosine_max": deque(maxlen=window_size),
            "vis_txt_cosine_mean": deque(maxlen=window_size),
            "alignment_loss": deque(maxlen=window_size),
            "reg_loss": deque(maxlen=window_size),
            "projected_embeddings_histogram": None,
            "covariance_matrix": None,
        }

    def log(self, global_step: int, metrics: Dict[str, Union[int, float, torch.Tensor]]) -> None:
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
        vis_embed_mean: Optional[float] = None,
        vis_embed_std: Optional[float] = None,
        vis_l2_mean: Optional[float] = None,
        vis_l2_std: Optional[float] = None,
        txt_embed_mean: Optional[float] = None,
        txt_embed_std: Optional[float] = None,
        txt_l2_mean: Optional[float] = None,
        txt_l2_std: Optional[float] = None,
        vis_txt_cosine_min: Optional[float] = None,
        vis_txt_cosine_max: Optional[float] = None,
        vis_txt_cosine_mean: Optional[float] = None,
        alignment_loss: Optional[torch.Tensor] = None,
        reg_loss: Optional[torch.Tensor] = None,
        projected_embeddings_histogram: Optional[torch.Tensor] = None,
        covariance_matrix: Optional[torch.Tensor] = None,
        **kwargs
    ) -> None:
        if global_step is not None:
            self.global_step = global_step

        if not overwatch.is_rank_zero():
            return

        if lr is not None:
            self.state["lr"].append(lr)

        if alignment_loss is not None:
            alignment_loss_val = alignment_loss.detach()
            self.state["alignment_loss"].append(alignment_loss_val)

        if reg_loss is not None:
            reg_loss_val = reg_loss.detach()
            self.state["reg_loss"].append(reg_loss_val)

        if update_step_time:
            self.state["step_time"].append(time.time() - self.step_start_time)
            self.step_start_time = time.time()

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

        if vis_txt_cosine_min is not None:
            self.state["vis_txt_cosine_min"].append(torch.tensor(vis_txt_cosine_min))
        if vis_txt_cosine_max is not None:
            self.state["vis_txt_cosine_max"].append(torch.tensor(vis_txt_cosine_max))
        if vis_txt_cosine_mean is not None:
            self.state["vis_txt_cosine_mean"].append(torch.tensor(vis_txt_cosine_mean))

        if projected_embeddings_histogram is not None:
            self.state["projected_embeddings_histogram"] = projected_embeddings_histogram.detach()

        if covariance_matrix is not None:
            self.state["covariance_matrix"] = covariance_matrix.detach()

        # Handle additional kwargs, ensuring proper initialization for new keys
        for key, value in kwargs.items():
            if key == "loss":
                loss_val = value.detach()
                self.state["loss_raw"].append(loss_val)
                self.state["loss"].append(loss_val)
            elif key in ["projected_embeddings_histogram", "covariance_matrix", "projected_embeddings_values"]:
                self.state[key] = value.detach()
            else:
                if key not in self.state:
                    self.state[key] = deque(maxlen=window_size)
                self.state[key].append(value.detach())

    @overwatch.rank_zero_only
    def push(self) -> str:
        prefix = self.stage.capitalize()

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

        if len(self.state["alignment_loss"]) > 0:
            alignment_loss = torch.stack(list(self.state["alignment_loss"])).mean().item()
            metrics[f"{prefix}/Alignment Loss"] = alignment_loss
            self.state["alignment_loss"].clear()

        if len(self.state["reg_loss"]) > 0:
            reg_loss = torch.stack(list(self.state["reg_loss"])).mean().item()
            metrics[f"{prefix}/Reg Loss"] = reg_loss
            self.state["reg_loss"].clear()

        if len(self.state["rank_entropy"]) > 0:
            avg_rank_entropy = np.mean(self.state["rank_entropy"])
            metrics[f"{prefix}/Rank Entropy"] = avg_rank_entropy
            self.state["rank_entropy"].clear()

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

        if self.state["ft_total_weight_change"] is not None:
            metrics[f"{prefix}/FT Total Weight Change"] = self.state["ft_total_weight_change"]
            self.state["ft_total_weight_change"] = None
        if self.state["ft_layer_weight_changes"]:
            for layer, changes in self.state["ft_layer_weight_changes"].items():
                metrics[f"{prefix}/FT Weight Change Layer {layer}"] = changes['layer']
                metrics[f"{prefix}/FT Weight Change Attention Layer {layer}"] = changes['attention']
                metrics[f"{prefix}/FT Weight Change MLP Layer {layer}"] = changes['mlp']
            self.state["ft_layer_weight_changes"] = {}
        if self.state["ft_parameter_weight_changes"]:
            for param_name, param_change in self.state["ft_parameter_weight_changes"].items():
                metrics[f"{prefix}/FT Weight Change Param {param_name}"] = param_change
            self.state["ft_parameter_weight_changes"] = {}

        def safe_mean_of_deque(dq):
            if len(dq) == 0:
                return 0.0
            vals = torch.stack(list(dq))
            return vals.mean().item()

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

        if len(self.state["vis_txt_cosine_min"]) > 0:
            metrics[f"{prefix}/VisTxtCosineMin"] = safe_mean_of_deque(self.state["vis_txt_cosine_min"])
            self.state["vis_txt_cosine_min"].clear()
        if len(self.state["vis_txt_cosine_max"]) > 0:
            metrics[f"{prefix}/VisTxtCosineMax"] = safe_mean_of_deque(self.state["vis_txt_cosine_max"])
            self.state["vis_txt_cosine_max"].clear()
        if len(self.state["vis_txt_cosine_mean"]) > 0:
            metrics[f"{prefix}/VisTxtCosineMean"] = safe_mean_of_deque(self.state["vis_txt_cosine_mean"])
            self.state["vis_txt_cosine_mean"].clear()

        # Include histogram and covariance matrix in metrics if available
        if "projected_embeddings_histogram" in self.state and self.state["projected_embeddings_histogram"] is not None:
            metrics["projected_embeddings_histogram"] = self.state["projected_embeddings_histogram"]
            self.state["projected_embeddings_histogram"] = None

        if "covariance_matrix" in self.state and self.state["covariance_matrix"] is not None:
            metrics["covariance_matrix"] = self.state["covariance_matrix"]
            self.state["covariance_matrix"] = None
        
        if "projected_embeddings_values" in self.state and self.state["projected_embeddings_values"] is not None:
            metrics["projected_embeddings_values"] = self.state["projected_embeddings_values"]
            self.state["projected_embeddings_values"] = None

        self.log(self.global_step, metrics)
        return status

    def finalize(self) -> None:
        for tracker in self.trackers:
            tracker.finalize()