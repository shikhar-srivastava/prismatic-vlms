"""
nn_utils.py

Utility functions and PyTorch submodule definitions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 


# === Definitions for Various Projection Modules, with Signature :: [..., in_dim] --> [..., out_dim] ===
class LinearProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.projector = nn.Linear(vision_dim, llm_dim, bias=True)

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(img_patches)


class MLPProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int, mlp_type: str = "gelu-mlp") -> None:
        super().__init__()
        if mlp_type == "gelu-mlp":
            self.projector = nn.Sequential(
                nn.Linear(vision_dim, llm_dim, bias=True),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim, bias=True),
            )
        else:
            raise ValueError(f"Projector with `{mlp_type = }` is not supported!")

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(img_patches)


class FusedMLPProjector(nn.Module):
    def __init__(self, fused_vision_dim: int, llm_dim: int, mlp_type: str = "fused-gelu-mlp") -> None:
        super().__init__()
        self.initial_projection_dim = fused_vision_dim * 4
        if mlp_type == "fused-gelu-mlp":
            self.projector = nn.Sequential(
                nn.Linear(fused_vision_dim, self.initial_projection_dim, bias=True),
                nn.GELU(),
                nn.Linear(self.initial_projection_dim, llm_dim, bias=True),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim, bias=True),
            )
        else:
            raise ValueError(f"Fused Projector with `{mlp_type = }` is not supported!")

    def forward(self, fused_img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(fused_img_patches)


# === New GLUProjector Class based on suggested modifications ===
# Based on : https://arxiv.org/pdf/2412.04616 (Zhang et al. 2024)
class GLUProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int, mlp_type: str = "glu-zhang") -> None:
        super().__init__()
        assert mlp_type == "glu-zhang", f"GLU Projector with `{mlp_type = }` is not supported!"
        # Single linear layer that outputs 2 * llm_dim for gating
        self.linear = nn.Linear(vision_dim, 2 * llm_dim, bias=True)
        self.relu = nn.ReLU()

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        x = self.linear(img_patches)
        # Apply GLU: splits the output into two halves and applies a sigmoid gate
        x = F.glu(x, dim=-1)
        x = self.relu(x)
        return x