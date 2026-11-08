"""Stochastic uniform quantization (Eqs. 3-7) and bit-length schedulers (Eqs. 8-9)."""
from __future__ import annotations

import math

import torch


_EPS = 2.220446049250313e-16  # sys.float_info.epsilon, as in the official code


def stochastic_quantize(x: torch.Tensor, bits: int):
    """Symmetric quantization with stochastic rounding.

    Eq.(4)  s = (2^(b-1) - 1) / alpha,  alpha = max|x|
    Eq.(6)  rho(y) = floor(y) w.p. ceil(y)-y, else ceil(y)   (unbiased)
    Eq.(5)  clip to [-(2^(b-1)-1), 2^(b-1)-1]
    Granularity follows the authors' "per_batch" setting (fedml/utils/quantize.py): every 3-D
    sub-tensor of a 4-D conv weight (one output filter) gets its own scale; tensors with < 4 dims
    use one scale ("each three-dimensional tensor in each layer", Sec. III-D).
    Returns integer codes (stored as float for simulation) and the scale(s) s.
    """
    if bits < 2:
        raise ValueError("bits must be >= 2")
    qmax = 2 ** (bits - 1) - 1
    if x.dim() >= 4:
        alpha = x.abs().amax(dim=tuple(range(1, x.dim())), keepdim=True) + _EPS
    else:
        alpha = x.abs().max() + _EPS
    s = qmax / alpha
    y = x * s
    low = torch.floor(y)
    q = low + (torch.rand_like(y) < (y - low)).to(y.dtype)
    return q.clamp_(-qmax, qmax), s


def dequantize(q: torch.Tensor, s) -> torch.Tensor:
    """Eq.(7)."""
    return q / s


def fake_quantize_state(state: dict, bits: int) -> dict:
    """Quantize->dequantize every floating tensor of a state_dict (what the receiver sees)."""
    if bits >= 32:
        return {k: v.detach().clone() for k, v in state.items()}
    out = {}
    for k, v in state.items():
        if v.is_floating_point():
            q, s = stochastic_quantize(v.detach(), bits)
            out[k] = dequantize(q, s)
        else:
            out[k] = v.detach().clone()
    return out


def payload_bits(state: dict, bits: int, count_scales: bool = False) -> int:
    """Bits needed to transmit all floating tensors (weights + BN statistics) at `bits`
    precision. Like the official code (stats.get_state_dict_size), scale factors are not counted.
    Matches paper Table 6 (e.g. MNIST CNN, 50 clients, FP32, 1000 rounds -> 61.97 GiB)."""
    floats = [v for v in state.values() if v.is_floating_point()]
    n = sum(v.numel() for v in floats)
    extra = 32 * len(floats) if (bits < 32 and count_scales) else 0
    return n * bits + extra


# ----------------------------------------------------------------------------- schedulers

def cosine_bits(t: int, T: int, bmin: int, bmax: int, nu: float = 1.0) -> int:
    """Eq.(8): b = bmin + nu (bmax - bmin) (1 + cos(pi t / T)) / 2, rounded to an integer.

    Uses T as in Eq. 8 and the official code (stats.consine_schedule); Alg. 1 line 5 prints T-1."""
    b = bmin + nu * (bmax - bmin) * (1 + math.cos(math.pi * t / max(T, 1))) / 2
    return int(min(max(round(b), bmin), bmax))


def client_importance(norm_entropy: float, n_i: int, n_max: int, lambda_h: float) -> float:
    """Eq.(9): nu_i = lambda_h * H_i / log2(K) + (1 - lambda_h) * |D_i| / n_max."""
    return float(lambda_h * norm_entropy + (1 - lambda_h) * n_i / max(n_max, 1))


def parse_quant(mode: str):
    """'fp32' | 'intK' (static K bits) | 'cosine' | 'dynamic'."""
    mode = mode.lower()
    if mode == "fp32":
        return "static", 32
    if mode.startswith("int"):
        return "static", int(mode[3:])
    if mode in ("cosine", "dynamic"):
        return mode, None
    raise ValueError(f"unknown quant mode {mode}")


def server_bits(mode: str, t: int, T: int, bmin: int, bmax: int) -> int:
    kind, b = parse_quant(mode)
    return b if kind == "static" else cosine_bits(t, T, bmin, bmax)


def client_bits(mode: str, t: int, T: int, bmin: int, bmax: int, nu: float) -> int:
    kind, b = parse_quant(mode)
    if kind == "static":
        return b
    return cosine_bits(t, T, bmin, bmax, nu if kind == "dynamic" else 1.0)
