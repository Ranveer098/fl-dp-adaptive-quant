"""Laplacian local differential privacy (paper Section III-F, Eqs. 11-13, Algorithm 1 lines 21-26)."""
from __future__ import annotations

import math

import torch


def clip_grad_l1_(params, xi: float) -> float:
    """Scale gradients so that their total L1 norm is <= xi. Returns the pre-clip L1 norm.

    Algorithm 1 line 21 prints  grad / max(||grad||_1, xi);  read literally, xi = inf would
    stop learning. The official code uses torch.nn.utils.clip_grad_norm_(params, xi, norm_type=1),
    i.e. standard clipping  grad * min(1, xi / ||grad||_1), which is what we do."""
    grads = [p.grad for p in params if p.grad is not None]
    total = torch.stack([g.abs().sum() for g in grads]).sum()
    if not math.isinf(xi):
        coef = torch.clamp(xi / (total + 1e-12), max=1.0)
        for g in grads:
            g.mul_(coef)
    return float(total)


class LipschitzTracker:
    """Eq.(13): lambda_i = ||grad f(theta) - grad f(theta')||_1 / ||theta - theta'||_1.

    "We track the gradient and parameter differences for each batch across successive
    epochs ... and use the largest value" (Sec. III-F). As in the official trainer, the
    gradients recorded are the CLIPPED ones and the weights are those before the SGD step.
    Batch composition is fixed within a round so batch j is the same data in every epoch.
    The official code keeps a model copy for every batch of every epoch; we keep only the
    previous epoch (same result, far less memory). `max_batches` limits the tracked batch
    slots (-1 = all, as in the paper)."""

    def __init__(self, max_batches: int = -1):
        self.max_batches = max_batches
        self.prev = {}
        self.lam = 0.0

    def wants(self, j: int) -> bool:
        return self.max_batches < 0 or j < self.max_batches

    @torch.no_grad()
    def update(self, j: int, params):
        theta = torch.cat([p.detach().reshape(-1) for p in params])
        grad = torch.cat([p.grad.detach().reshape(-1) for p in params])
        if j in self.prev:
            th0, g0 = self.prev[j]
            d = (theta - th0).abs().sum()
            if d > 0:
                self.lam = max(self.lam, float((grad - g0).abs().sum() / d))
        self.prev[j] = (theta, grad)


def l1_sensitivity(xi: float, eta: float, E: int, n: int, lam: float) -> float:
    """Eq.(12). The paper defines E0 as the smallest INTEGER with (1 + lam*eta)^E0 >= 1 + n;
    the official code (core/dp/mechanisms/laplace.py) uses the real value log(1+n)/log(1+lam*eta)
    and treats lam <= 1e-10 as 0. We follow the code."""
    if lam <= 1e-10:
        return 2 * xi * E * eta / n
    E0 = math.log(1 + n) / math.log(1 + lam * eta)
    if E < E0:
        return 2 * xi / (lam * n) * ((1 + lam * eta) ** E - 1)
    return 2 * xi + 2 * eta * xi * (E - E0)


def laplace_scale(eps: float, sens: float, P: int, T: int, N: int, E: int) -> float:
    """Noise scale  b = T_i * Xi / eps  with  T_i = P T / (N E)  (Sec. III-F)."""
    return (P * T / (N * E)) * sens / eps


@torch.no_grad()
def add_laplace_noise_(tensors, scale: float):
    """Algorithm 1 lines 25-26: theta <- theta + Lap(0, scale), element-wise.
    The official code adds noise to every entry of the state_dict (weights AND BatchNorm
    running statistics); the caller passes those tensors."""
    if scale <= 0:
        return
    for p in tensors:
        u = torch.rand_like(p).clamp_(1e-12, 1 - 1e-12) - 0.5
        p.add_(-scale * torch.sign(u) * torch.log1p(-2 * u.abs()))
