"""
compare_activations.py
----------------------
Visual comparison of Lap4-PW-C¹ (continuous capped squared-ReLU) with several
standard activations.

Author: ChatGPT • 2025
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------
# 1.  Lap4-PW-C¹ activation (piece-wise: ReLU² up to T, Laplace tail beyond)
# -------------------------------------------------------------------------

class Lap4PWC1(nn.Module):
    """
    Continuous C¹ activation that is:
      * f(x) = x²           for 0 < x <= T
      * f(x) = 4 * Laplace  for       x  > T
      * f(x) = 0            for  x <= 0
    with value AND slope matching at x = T.
    Constants below are pre-solved for cap A = 4, switch T = 1.9.
    """

    __constants__ = ("A", "T", "mu", "sigma")

    def __init__(self):
        super().__init__()
        # Positive cap
        self.A = 4.0
        # Switch point (must have T² < A)
        self.T = 1.9
        # Pre-computed Laplace parameters (see derivation)
        self.mu = 1.665       # matches value & slope at x = T
        self.sigma = 0.1813

        # Helpful scalar tensors for broadcasting
        self._A = torch.tensor(self.A)
        self._T = torch.tensor(self.T)
        self._mu = torch.tensor(self.mu)
        self._sig = torch.tensor(self.sigma)
        self._sqrt2 = torch.tensor(1.4142135623730951)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # region masks
        pos_sq  = (x > 0) & (x <= self._T)
        pos_lap =  x > self._T

        y = torch.zeros_like(x)

        # squared-ReLU branch
        if pos_sq.any():
            y[pos_sq] = x[pos_sq] ** 2

        # Laplace tail (already multiplied by cap A = 4)
        if pos_lap.any():
            z = (x[pos_lap] - self._mu) / (self._sig * self._sqrt2)
            y[pos_lap] = 0.5 * self._A * (1.0 + torch.erf(z))

        # negatives stay 0
        return y


# -------------------------------------------------------------------------
# 2.  Additional activations for comparison
# -------------------------------------------------------------------------

class ReLU2(nn.Module):
    """Squared ReLU (x > 0 ? x² : 0)."""
    def forward(self, x):                      # noqa: D401
        return torch.clamp_min(x, 0.0) ** 2


class Rational4(nn.Module):
    """Smooth cap via rational function: 4 x² / (x² + 4)."""
    def __init__(self, A=4.0):
        super().__init__()
        self.A = A

    def forward(self, x):
        return self.A * x.pow(2) / (x.pow(2) + self.A)


# HardTanh with lower = 0, upper = 4 ("ReLU6-style" with 4 instead of 6)
HardTanh4 = nn.Hardtanh(min_val=0.0, max_val=4.0)


# -------------------------------------------------------------------------
# 3.  Test & visualise
# -------------------------------------------------------------------------

def main():
    # Grid for plotting
    x = torch.linspace(-0.5, 4.0, 2000, dtype=torch.float32)

    activations = {
        "ReLU":        nn.ReLU(),
        "ReLU$^2$":    ReLU2(),
        "Lap4-PW-C¹":  Lap4PWC1(),
        "Rational-4":  Rational4(),
        "HardTanh-4":  HardTanh4,
    }

    plt.figure(figsize=(8, 4.5))
    for name, act in activations.items():
        y = act(x)
        plt.plot(x.numpy(), y.numpy(), label=name, linewidth=1.5)

    plt.title("Activation comparison", fontsize=13)
    plt.xlabel("x")
    plt.ylabel("activation output")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()