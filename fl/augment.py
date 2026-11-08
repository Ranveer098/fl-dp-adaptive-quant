"""Batched version of the authors' CIFAR10 training augmentation.

Official (fedml/data/cifar10/data_loader.py): for every image, ONE of
  RandomCrop(32, pad 4) | RandomHorizontalFlip | Cutout(10) | ColorJitter(0.8, 0.8, 0.8, 0.2)
  | RandomRotation(45) | RandomAffine(45, scale=(0.5, 1.5))
is chosen at random. The paper does not mention augmentation, but it affects CIFAR10 accuracy.
This implementation is vectorized over the batch (fast on CPU). ColorJitter's hue component is
omitted; brightness/contrast/saturation are applied in a fixed order.
"""
import math

import torch
import torch.nn.functional as F


def _rotate_scale(x, max_deg, scale_range=None):
    n = x.shape[0]
    ang = torch.empty(n, device=x.device).uniform_(-max_deg, max_deg) * math.pi / 180
    sc = torch.ones(n, device=x.device) if scale_range is None else \
        torch.empty(n, device=x.device).uniform_(*scale_range)
    cos, sin, zero = torch.cos(ang) / sc, torch.sin(ang) / sc, torch.zeros(n, device=x.device)
    theta = torch.stack([torch.stack([cos, -sin, zero], 1), torch.stack([sin, cos, zero], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


def _gray(x):
    return (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)


def official_cifar_augment(x: torch.Tensor) -> torch.Tensor:
    """x: float batch in [0, 1], shape (B, 3, H, W)."""
    B, _, H, W = x.shape
    dev = x.device
    choice = torch.randint(0, 6, (B,), device=dev)
    out = x.clone()

    m = choice == 0  # random crop with 4-pixel zero padding
    if m.any():
        xs = F.pad(x[m], (4, 4, 4, 4))
        n = xs.shape[0]
        i = torch.randint(0, 9, (n,)).tolist()
        j = torch.randint(0, 9, (n,)).tolist()
        out[m] = torch.stack([xs[k, :, i[k]:i[k] + H, j[k]:j[k] + W] for k in range(n)])

    m = choice == 1  # horizontal flip with p = 0.5
    if m.any():
        xs = x[m]
        flip = torch.rand(xs.shape[0], device=dev) < 0.5
        xs[flip] = xs[flip].flip(-1)
        out[m] = xs

    m = choice == 2  # Cutout: one 10x10 hole, centre anywhere (clipped at the border)
    if m.any():
        n = int(m.sum())
        cy = torch.randint(0, H, (n, 1, 1), device=dev)
        cx = torch.randint(0, W, (n, 1, 1), device=dev)
        yy = torch.arange(H, device=dev).view(1, H, 1)
        xx = torch.arange(W, device=dev).view(1, 1, W)
        hole = (yy >= cy - 5) & (yy < cy + 5) & (xx >= cx - 5) & (xx < cx + 5)
        out[m] = x[m] * (~hole).unsqueeze(1).to(x.dtype)

    m = choice == 3  # colour jitter (brightness, contrast, saturation factors in [0.2, 1.8])
    if m.any():
        xs = x[m]
        n = xs.shape[0]

        def f():
            return torch.empty(n, 1, 1, 1, device=dev).uniform_(0.2, 1.8)

        xs = (xs * f()).clamp(0, 1)
        c = f()
        xs = (c * xs + (1 - c) * _gray(xs).mean(dim=(2, 3), keepdim=True)).clamp(0, 1)
        s = f()
        xs = (s * xs + (1 - s) * _gray(xs)).clamp(0, 1)
        out[m] = xs

    m = choice == 4  # rotation in [-45, 45] degrees
    if m.any():
        out[m] = _rotate_scale(x[m], 45)

    m = choice == 5  # affine: rotation [-45, 45] and scale [0.5, 1.5]
    if m.any():
        out[m] = _rotate_scale(x[m], 45, (0.5, 1.5))
    return out
