"""Datasets and non-IID partitioning (paper Sec. III-B), matched to the authors' FedML code.

Images are kept in memory as uint8 tensors and scaled to [0, 1] per batch (the official code
feeds [0, 1] inputs without mean/std normalization). This avoids DataLoader overhead.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch


# ----------------------------------------------------------------------------- loading

def _torchvision_mnist(root):
    from torchvision import datasets
    tr = datasets.MNIST(root, train=True, download=True)
    te = datasets.MNIST(root, train=False, download=True)
    return (tr.data.unsqueeze(1).contiguous(), tr.targets.long().clone(),
            te.data.unsqueeze(1).contiguous(), te.targets.long().clone())


def load_leaf_mnist(leaf_dir: str, root: str):
    """The authors' MNIST: LEAF/FedProx JSON with 1000 users x 2 digits (61,664 train / 7,371 test).

    Download https://fedcv.s3.us-west-1.amazonaws.com/MNIST.zip and unzip so that
    <leaf_dir>/train/*.json and <leaf_dir>/test/*.json exist. Values in these files are
    standardized per pixel; like the authors' renormalize.py we map them back to [0, 1]
    with the per-pixel mean/std of the full 70k MNIST. The result is cached as .npz.
    Returns x_train_u8, y_train, x_test_u8, y_test, user_parts (list of index arrays).
    """
    cache = os.path.join(leaf_dir, "leaf_mnist_cache.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        bounds = z["bounds"]
        parts = [np.arange(a, b) for a, b in zip(bounds[:-1], bounds[1:])]
        return (torch.from_numpy(z["xtr"]), torch.from_numpy(z["ytr"]).long(),
                torch.from_numpy(z["xte"]), torch.from_numpy(z["yte"]).long(), parts)

    def read(split):
        data = {}
        d = os.path.join(leaf_dir, split)
        for f in sorted(os.listdir(d)):
            if f.endswith(".json"):
                with open(os.path.join(d, f)) as fh:
                    data.update(json.load(fh)["user_data"])
        return data

    print("[leaf-mnist] reading JSON (first time only, needs ~3 GB RAM) ...", flush=True)
    train, test = read("train"), read("test")
    users = sorted(train.keys())  # the official loader iterates over sorted user ids

    xtr_raw, _, xte_raw, _ = _torchvision_mnist(root)
    full = torch.cat([xtr_raw, xte_raw]).reshape(-1, 784).float().numpy()
    mu, sigma = full.mean(0), full.std(0)

    def to_u8(xs):
        x = np.asarray(xs, dtype=np.float32)
        if x.min() < 0:  # standardized -> back to [0, 1] (renormalize.py)
            x = (x * sigma + mu) / 255.0
        return np.clip(np.round(x * 255), 0, 255).astype(np.uint8).reshape(-1, 1, 28, 28)

    xs, ys, bounds = [], [], [0]
    for u in users:
        xs.append(to_u8(train[u]["x"]))
        ys.append(np.asarray(train[u]["y"], dtype=np.int64))
        bounds.append(bounds[-1] + len(ys[-1]))
    xte = np.concatenate([to_u8(test[u]["x"]) for u in users if u in test])
    yte = np.concatenate([np.asarray(test[u]["y"], dtype=np.int64) for u in users if u in test])
    xtr, ytr, bounds = np.concatenate(xs), np.concatenate(ys), np.array(bounds)
    np.savez(cache, xtr=xtr, ytr=ytr, xte=xte, yte=yte, bounds=bounds)
    parts = [np.arange(a, b) for a, b in zip(bounds[:-1], bounds[1:])]
    print(f"[leaf-mnist] {len(users)} users, {len(ytr)} train / {len(yte)} test samples", flush=True)
    return torch.from_numpy(xtr), torch.from_numpy(ytr), torch.from_numpy(xte), torch.from_numpy(yte), parts


def load_dataset(name: str, root: str = "./data", mnist_source: str = "torchvision", leaf_dir: str = ""):
    """Return (x_train_u8, y_train, x_test_u8, y_test, user_parts_or_None)."""
    if name == "mnist":
        if mnist_source == "leaf":
            return load_leaf_mnist(leaf_dir or os.path.join(root, "MNIST"), root)
        return (*_torchvision_mnist(root), None)
    if name == "cifar10":
        from torchvision import datasets
        tr = datasets.CIFAR10(root, train=True, download=True)
        te = datasets.CIFAR10(root, train=False, download=True)
        xtr = torch.from_numpy(tr.data).permute(0, 3, 1, 2).contiguous()
        xte = torch.from_numpy(te.data).permute(0, 3, 1, 2).contiguous()
        return (xtr, torch.tensor(tr.targets, dtype=torch.long), xte,
                torch.tensor(te.targets, dtype=torch.long), None)
    raise ValueError(f"unknown dataset {name}")


def to_unit(x_u8: torch.Tensor) -> torch.Tensor:
    """uint8 batch -> float in [0, 1] (official preprocessing: ToTensor, no Normalize)."""
    return x_u8.float().div_(255.0)


# ----------------------------------------------------------------------------- partitions

def synthetic_mnist_users(labels: np.ndarray, rng: np.random.Generator, n_users: int = 1000,
                          classes_per_user: int = 2, sigma: float = 1.0, min_per_class: int = 5):
    """Stand-in for the LEAF MNIST users when only torchvision MNIST is available:
    1000 users, 2 digits each, heavy-tailed (log-normal) sizes."""
    K = int(labels.max()) + 1
    user_classes = [rng.choice(K, classes_per_user, replace=False) for _ in range(n_users)]
    weights = rng.lognormal(mean=0.0, sigma=sigma, size=n_users)
    user_idx = [[] for _ in range(n_users)]
    for k in range(K):
        pool = rng.permutation(np.where(labels == k)[0])
        owners = np.array([u for u in range(n_users) if k in user_classes[u]])
        if len(owners) == 0:
            continue
        avail = len(pool) - min_per_class * len(owners)
        if avail < 0:
            raise ValueError("too many owners for class pool; lower min_per_class")
        share = weights[owners] / weights[owners].sum()
        counts = np.floor(share * avail).astype(int) + min_per_class
        counts[np.argmax(share)] += len(pool) - counts.sum()
        for u, chunk in zip(owners, np.split(pool, np.cumsum(counts)[:-1])):
            user_idx[u].extend(chunk.tolist())
    return [np.array(u, dtype=np.int64) for u in user_idx]


def merge_users(user_parts, n_clients: int):
    """Official rule (fedml/data/MNIST/data_loader.py): user u >= N is merged into client u % N."""
    if n_clients >= len(user_parts):
        return list(user_parts)
    groups = [[] for _ in range(n_clients)]
    for u, p in enumerate(user_parts):
        groups[u % n_clients].append(p)
    return [np.concatenate(g) for g in groups]


def partition_dirichlet(labels: np.ndarray, n_clients: int, alpha: float, rng: np.random.Generator,
                        min_size: int = 10, max_tries: int = 1000):
    """FedML 'hetero' label-skew Dirichlet(alpha) partition (alpha=0.5 for CIFAR10), including
    its balancing trick: clients above the average size receive no further samples."""
    K = int(labels.max()) + 1
    N = len(labels)
    for _ in range(max_tries):
        parts = [[] for _ in range(n_clients)]
        for k in range(K):
            idx = rng.permutation(np.where(labels == k)[0])
            p = rng.dirichlet(np.full(n_clients, alpha))
            sizes = np.array([len(c) for c in parts])
            p = p * (sizes < N / n_clients)
            p = p / p.sum()
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for i, chunk in enumerate(np.split(idx, cuts)):
                parts[i].extend(chunk.tolist())
        if min(len(c) for c in parts) >= min_size:
            return [np.array(c, dtype=np.int64) for c in parts]
    raise RuntimeError("could not satisfy min_size; lower it or raise alpha")


def partition_iid(labels: np.ndarray, n_clients: int, rng: np.random.Generator):
    return [np.sort(p) for p in np.array_split(rng.permutation(len(labels)), n_clients)]


def make_partition(dataset: str, labels: np.ndarray, n_clients: int, scheme: str, alpha: float,
                   rng: np.random.Generator, user_parts=None):
    """Returns (client index arrays, merged_flag). merged_flag is True when MNIST users were merged
    into fewer clients - the official code then trains each merged client as ONE full batch."""
    if scheme == "auto":
        scheme = "users" if dataset == "mnist" else "dirichlet"
    if scheme == "users":
        users = user_parts if user_parts is not None else synthetic_mnist_users(labels, rng)
        return merge_users(users, n_clients), n_clients < len(users)
    if scheme == "dirichlet":
        return partition_dirichlet(labels, n_clients, alpha, rng), False
    if scheme == "iid":
        return partition_iid(labels, n_clients, rng), False
    raise ValueError(f"unknown partition {scheme}")


def normalized_entropy(labels: np.ndarray, parts, K: int) -> np.ndarray:
    """First term of Eq. (9): Shannon entropy of each client's label histogram / log2(K)."""
    out = np.zeros(len(parts))
    for i, p in enumerate(parts):
        h = np.bincount(labels[p], minlength=K).astype(float)
        q = h[h > 0] / h.sum()
        out[i] = float(-(q * np.log2(q)).sum() / np.log2(K))
    return out
