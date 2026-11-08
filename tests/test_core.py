"""Sanity checks for the paper's equations.  Run:  python -m pytest -q"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl.augment import official_cifar_augment  # noqa: E402
from fl.data import merge_users, normalized_entropy, partition_dirichlet, synthetic_mnist_users  # noqa: E402
from fl.dp import add_laplace_noise_, l1_sensitivity, laplace_scale  # noqa: E402
from fl.models import MNISTCNN, VGG7  # noqa: E402
from fl.quant import (client_importance, cosine_bits, dequantize, fake_quantize_state,  # noqa: E402
                      payload_bits, stochastic_quantize)

GIB = 8 * 1024 ** 3


def test_quantization_is_unbiased_and_in_range():
    torch.manual_seed(0)
    x = torch.randn(1000)
    for b in (2, 4, 8, 16):
        q, s = stochastic_quantize(x, b)
        qmax = 2 ** (b - 1) - 1
        assert q.abs().max() <= qmax and torch.all(q == q.round())
        mean = torch.stack([dequantize(*stochastic_quantize(x, b)) for _ in range(400)]).mean(0)
        # one quantization step is 1/s; the 400-sample mean should be within 0.2 steps of x
        assert (mean - x).abs().max() < 0.2 / s


def test_per_filter_scales_for_conv_weights():
    torch.manual_seed(0)
    w = torch.randn(8, 3, 3, 3)
    w[0] *= 100  # one large filter must not destroy the precision of the others
    q, s = stochastic_quantize(w, 4)
    assert s.shape == (8, 1, 1, 1)
    err = (dequantize(q, s) - w).abs()
    assert err[1:].max() < 1.0


def test_augmentation_shapes():
    x = torch.rand(64, 3, 32, 32)
    y = official_cifar_augment(x)
    assert y.shape == x.shape and torch.isfinite(y).all() and y.min() >= 0 and y.max() <= 1


def test_fp32_is_identity():
    m = MNISTCNN()
    st = fake_quantize_state(m.state_dict(), 32)
    assert all(torch.equal(st[k], v) for k, v in m.state_dict().items())


def test_cosine_schedule_endpoints():
    T = 100
    assert cosine_bits(0, T, 8, 32) == 32
    assert cosine_bits(T - 1, T, 8, 32) == 8
    assert cosine_bits(0, T, 8, 32, nu=0.5) == 20
    bits = [cosine_bits(t, T, 8, 32) for t in range(T)]
    assert all(a >= b for a, b in zip(bits, bits[1:]))  # descending


def test_importance_range():
    assert client_importance(1.0, 100, 100, 0.75) == 1.0
    assert 0 <= client_importance(0.0, 1, 100, 0.75) <= 1


def test_param_counts_and_comm_cost_match_paper():
    mn, vg = MNISTCNN().state_dict(), VGG7().state_dict()
    # Table 6, 50 clients (5 per round), FP32, 1000 rounds, up + down
    mnist_gib = payload_bits(mn, 32) * 5 * 2 * 1000 / GIB
    cifar_gib = payload_bits(vg, 32) * 5 * 2 * 1000 / GIB
    assert abs(mnist_gib - 61.97) < 0.05, mnist_gib
    assert abs(cifar_gib - 12.13) < 0.05, cifar_gib


def test_sensitivity_cases():
    assert math.isclose(l1_sensitivity(100, 0.1, 5, 500, 0.0), 2 * 100 * 5 * 0.1 / 500)
    s_small = l1_sensitivity(100, 0.1, 5, 500, 0.5)       # E < E0
    assert 0 < s_small < 1
    assert l1_sensitivity(100, 0.1, 5, 500, 1000.0) >= 200  # E >= E0 -> huge
    assert l1_sensitivity(100, 0.1, 5, 500, 1e-12) == l1_sensitivity(100, 0.1, 5, 500, 0.0)
    assert math.isclose(laplace_scale(1e4, 0.2, 10, 1000, 100, 5), 20 * 0.2 / 1e4)


def test_laplace_noise_scale():
    torch.manual_seed(0)
    p = torch.zeros(200_000)
    add_laplace_noise_([p], 0.5)
    assert abs(p.mean()) < 0.01
    assert abs(p.abs().mean() - 0.5) < 0.01  # E|Lap(0,b)| = b


def test_partitions():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(10), 6000)
    users = synthetic_mnist_users(labels, rng)
    assert len(users) == 1000 and sum(len(u) for u in users) == len(labels)
    assert all(len(np.unique(labels[u])) <= 2 for u in users)
    merged = merge_users(users, 100)  # official rule: user u -> client u % 100
    assert len(merged) == 100 and sum(len(u) for u in merged) == len(labels)
    assert len(merged[3]) == sum(len(users[u]) for u in range(3, 1000, 100))
    d = partition_dirichlet(labels, 100, 0.5, np.random.default_rng(0))
    assert sum(len(u) for u in d) == len(labels) and min(len(u) for u in d) >= 10
    H = normalized_entropy(labels, users, 10)
    assert H.max() <= 1 / math.log2(10) + 1e-9  # two classes at most -> <= log2(2)/log2(10)
