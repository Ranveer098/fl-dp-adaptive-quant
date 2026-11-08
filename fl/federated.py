"""Algorithm 1: FedAvg + Laplacian local DP + adaptive quantization (single-process simulation)."""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .augment import official_cifar_augment
from .data import load_dataset, make_partition, normalized_entropy, to_unit
from .dp import LipschitzTracker, add_laplace_noise_, clip_grad_l1_, l1_sensitivity, laplace_scale
from .models import build_model
from .quant import client_bits, client_importance, fake_quantize_state, payload_bits, server_bits

GIB = 8 * 1024 ** 3  # bits per GiB (the paper's "GBs" match GiB)


@dataclass
class Config:
    dataset: str = "mnist"           # mnist | cifar10
    n_clients: int = 100             # N
    frac: float = 0.1                # P / N (paper keeps 0.1)
    rounds: int = 1000               # T
    local_epochs: int = 5            # E
    batch_size: int = 64             # B
    lr: float = 0.1                  # eta
    weight_decay: float = 0.0        # paper says 1e-3, but the official SGD optimizer ignores it
    xi: float = 100.0                # L1 gradient clipping bound (inf = no clipping)
    epsilon: float = 1e4             # privacy budget (inf = no DP noise)
    lipschitz: str = "paper"         # paper (Eq. 13) | zero (lambda=0 -> first case of Eq. 12)
    lip_batches: int = -1            # batch slots tracked for Eq. 13 (-1 = all)
    quant: str = "fp32"              # fp32 | int4 | int8 | int16 | cosine | dynamic
    bmax: int = 32
    bmin: int = 8
    lambda_h: float = 0.75
    partition: str = "auto"          # auto | users | dirichlet | iid
    alpha: float = 0.5               # Dirichlet concentration
    mnist_source: str = "torchvision"  # torchvision (synthetic 1000 users) | leaf (authors' MNIST JSON)
    leaf_dir: str = ""               # folder with train/ and test/ JSON (default: <data_root>/MNIST)
    batch_mode: str = "auto"         # auto: official behaviour (merged MNIST clients = one full batch) | fixed
    augment: str = "auto"            # auto: official CIFAR10 augmentation, none for MNIST | official | none
    input_norm: str = "unit"         # unit: pixels in [0, 1] | standardize: per-pixel z-score, i.e. the
                                     # LEAF MNIST JSON as stored if renormalize.py was never run
    seed: int = 0
    eval_every: int = 10
    device: str = "auto"
    threads: int = 0                 # torch CPU threads (0 = torch default)
    data_root: str = "./data"
    out_dir: str = "./results"
    run_name: str = ""
    resume: bool = True
    max_minutes: float = 0.0         # stop cleanly after this wall time (0 = no limit)

    def name(self) -> str:
        if self.run_name:
            return self.run_name
        eps = "inf" if math.isinf(self.epsilon) else f"{self.epsilon:g}"
        xi = "inf" if math.isinf(self.xi) else f"{self.xi:g}"
        s = (f"{self.dataset}_N{self.n_clients}_{self.quant}_eps{eps}_xi{xi}"
             f"_T{self.rounds}_E{self.local_epochs}_s{self.seed}")
        if self.quant == "dynamic":
            s += f"_lh{self.lambda_h:g}"
        if self.partition not in ("auto",):
            s += f"_{self.partition}{self.alpha:g}" if self.partition == "dirichlet" else f"_{self.partition}"
        return s


def _device(cfg):
    if cfg.device != "auto":
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, state, x, y, prep=to_unit, bs=2000):
    model.load_state_dict(state)
    model.eval()
    correct, loss = 0, 0.0
    for i in range(0, len(y), bs):
        out = model(prep(x[i:i + bs]))
        loss += F.cross_entropy(out, y[i:i + bs], reduction="sum").item()
        correct += (out.argmax(1) == y[i:i + bs]).sum().item()
    return correct / len(y), loss / len(y)


def local_train(model, opt, x_all, y_all, idx, cfg, full_batch=False, augment=False, post=None):
    """Algorithm 1 lines 18-24: E epochs of clipped SGD; returns the Lipschitz estimate lambda_i."""
    model.train()
    params = [p for p in model.parameters()]
    perm = idx[torch.randperm(idx.numel(), device=idx.device)]
    batches = torch.split(perm, idx.numel() if full_batch else cfg.batch_size)
    track = cfg.lipschitz == "paper" and not math.isinf(cfg.epsilon)
    tracker = LipschitzTracker(cfg.lip_batches) if track else None
    for _ in range(cfg.local_epochs):
        for j, b in enumerate(batches):
            opt.zero_grad(set_to_none=True)
            xb = to_unit(x_all[b])
            if augment:
                xb = official_cifar_augment(xb)
            if post is not None:
                xb = post(xb)
            loss = F.cross_entropy(model(xb), y_all[b])
            loss.backward()
            clip_grad_l1_(params, cfg.xi)
            if tracker is not None and tracker.wants(j):   # official: clipped grads, pre-step weights
                tracker.update(j, params)
            opt.step()
    return tracker.lam if tracker is not None else 0.0


def run(cfg: Config) -> dict:
    os.makedirs(cfg.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(cfg.out_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    name = cfg.name()
    res_path = os.path.join(cfg.out_dir, f"{name}.json")
    ckpt_path = os.path.join(ckpt_dir, f"{name}.pt")
    if cfg.threads > 0:
        torch.set_num_threads(cfg.threads)
    device = _device(cfg)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    # ---- data + non-IID split
    xtr, ytr, xte, yte, user_parts = load_dataset(cfg.dataset, cfg.data_root, cfg.mnist_source, cfg.leaf_dir)
    labels = ytr.numpy()
    K = int(labels.max()) + 1
    parts, merged = make_partition(cfg.dataset, labels, cfg.n_clients, cfg.partition, cfg.alpha, rng, user_parts)
    full_batch = cfg.batch_mode == "auto" and merged
    augment = cfg.augment == "official" or (cfg.augment == "auto" and cfg.dataset == "cifar10")
    sizes = np.array([len(p) for p in parts])
    H = normalized_entropy(labels, parts, K)
    post = None  # optional transform applied after [0, 1] scaling (and augmentation)
    if cfg.input_norm == "standardize":
        allx = torch.cat([xtr, xte]).float()
        mu = allx.mean(0, keepdim=True).to(device)
        sd = allx.std(0, unbiased=False, keepdim=True).to(device) + 1e-3  # FedProx: (x - mu) / (sigma + 0.001)
        del allx

        def post(x01):
            return (x01 * 255.0 - mu) / sd
    xtr, ytr, xte, yte = xtr.to(device), ytr.to(device), xte.to(device), yte.to(device)
    parts_t = [torch.as_tensor(p, device=device) for p in parts]

    # ---- model
    model = build_model(cfg.dataset).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    global_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    N, T, E = cfg.n_clients, cfg.rounds, cfg.local_epochs
    P = max(1, int(round(cfg.frac * N)))
    dp_on = not math.isinf(cfg.epsilon)

    hist = {"round": [], "test_acc": [], "test_loss": [], "gib": [], "avg_up_bits": [],
            "down_bits": [], "mean_lambda": [], "mean_noise_scale": [], "mean_nu": [], "sec": []}
    start_round, total_bits, elapsed_before = 0, 0, 0.0
    round_log = {"up_bits": [], "lam": [], "noise": [], "nu": []}

    if cfg.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        global_state = {k: v.to(device) for k, v in ck["global_state"].items()}
        start_round, total_bits, hist = ck["round"], ck["total_bits"], ck["hist"]
        elapsed_before = ck.get("elapsed", 0.0)
        rng.bit_generator.state = ck["np_rng"]
        torch.set_rng_state(ck["torch_rng"])
        if device.type == "cuda" and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"].cpu())
        print(f"[resume] {name} from round {start_round}")

    print(f"[{name}] device={device} N={N} P={P} T={T} E={E} | train {len(labels)} / test {len(yte)} | "
          f"client sizes min={sizes.min()} median={int(np.median(sizes))} max={sizes.max()} | "
          f"mean norm. entropy={H.mean():.3f} | full-batch clients={full_batch} augment={augment} "
          f"input_norm={cfg.input_norm}", flush=True)

    t0 = time.time()
    for t in range(start_round, T):
        # official sampling: np.random.seed(round_idx); choice(N, P) -> same clients in every experiment
        sel = np.arange(N) if P == N else np.random.RandomState(t).choice(N, P, replace=False)
        n_max = int(sizes[sel].max())                                   # line 7
        b_t = server_bits(cfg.quant, t, T, cfg.bmin, cfg.bmax)          # line 5
        broadcast = fake_quantize_state(global_state, b_t)              # line 6 (+ line 18 at clients)
        down = payload_bits(global_state, b_t) * P
        agg = {k: torch.zeros_like(v) for k, v in global_state.items() if v.is_floating_point()}
        n_tot = 0
        up = 0
        for i in sel:
            model.load_state_dict(broadcast)
            lam = local_train(model, opt, xtr, ytr, parts_t[i], cfg, full_batch, augment, post)  # lines 19-24
            if cfg.lipschitz == "zero":
                lam = 0.0
            scale = 0.0
            if dp_on:                                                        # lines 25-26
                sens = l1_sensitivity(cfg.xi, cfg.lr, E, int(sizes[i]), lam)
                scale = laplace_scale(cfg.epsilon, sens, P, T, N, E)
                add_laplace_noise_([v for v in model.state_dict().values() if v.is_floating_point()], scale)
            nu = client_importance(H[i], int(sizes[i]), n_max, cfg.lambda_h)  # line 27
            b_i = client_bits(cfg.quant, t, T, cfg.bmin, cfg.bmax, nu)        # line 28
            state = model.state_dict()
            local = fake_quantize_state(state, b_i)                          # line 29 + line 11
            up += payload_bits(state, b_i)
            for k in agg:                                                    # line 14
                agg[k].add_(local[k].to(agg[k].dtype), alpha=float(sizes[i]))
            n_tot += int(sizes[i])
            round_log["up_bits"].append(b_i)
            round_log["lam"].append(lam)
            round_log["noise"].append(scale)
            round_log["nu"].append(nu)
        for k in agg:
            global_state[k] = agg[k] / n_tot
        for k, v in model.state_dict().items():   # integer buffers (BN num_batches_tracked)
            if not v.is_floating_point():
                global_state[k] = v.detach().clone()
        total_bits += down + up

        last = t == T - 1
        over_time = cfg.max_minutes > 0 and (time.time() - t0) / 60 > cfg.max_minutes
        # official: test at round_idx % freq == 0 and at the last round; rounds are 0-based
        if t % cfg.eval_every == 0 or last or over_time:
            acc, loss = evaluate(model, global_state, xte, yte,
                                 prep=to_unit if post is None else (lambda x: post(to_unit(x))))
            el = elapsed_before + time.time() - t0
            for key, val in [("round", t), ("test_acc", acc), ("test_loss", loss),
                             ("gib", total_bits / GIB), ("avg_up_bits", float(np.mean(round_log["up_bits"]))),
                             ("down_bits", b_t), ("mean_lambda", float(np.mean(round_log["lam"]))),
                             ("mean_noise_scale", float(np.mean(round_log["noise"]))),
                             ("mean_nu", float(np.mean(round_log["nu"]))), ("sec", el)]:
                hist[key].append(val)
            round_log = {k: [] for k in round_log}
            per_round = (time.time() - t0) / (t + 1 - start_round)
            eta_min = per_round * (T - t - 1) / 60
            print(f"[{name}] round {t:4d}/{T} acc={acc * 100:6.2f}% loss={loss:.3f} "
                  f"GiB={total_bits / GIB:8.3f} bits(down={b_t},up~{hist['avg_up_bits'][-1]:.1f}) "
                  f"lambda~{hist['mean_lambda'][-1]:.3g} noise~{hist['mean_noise_scale'][-1]:.2e} "
                  f"| {per_round:.2f}s/round ETA {eta_min:.1f} min", flush=True)
            _save(res_path, cfg, hist, finished=last, total_bits=total_bits)
            torch.save({"global_state": global_state, "round": t + 1, "total_bits": total_bits,
                        "hist": hist, "elapsed": el, "np_rng": rng.bit_generator.state,
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None},
                       ckpt_path)
            if over_time and not last:
                print(f"[{name}] stopping after {cfg.max_minutes} min; rerun the same command to resume")
                break
    return json.load(open(res_path))


def _save(path, cfg, hist, finished, total_bits):
    accs = hist["test_acc"]
    best = int(np.argmax(accs)) if accs else 0
    summary = {
        "best_acc": accs[best] if accs else None,
        "best_round": hist["round"][best] if accs else None,
        "final_acc": accs[-1] if accs else None,
        "total_gib": total_bits / GIB,
        "rounds_done": hist["round"][-1] if accs else 0,
        "wall_sec": hist["sec"][-1] if accs else 0,
    }
    cfgd = asdict(cfg)
    for k in ("xi", "epsilon"):
        if math.isinf(cfgd[k]):
            cfgd[k] = "inf"
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"name": cfg.name(), "config": cfgd, "summary": summary,
                   "finished": finished, "history": hist}, f, indent=1)
    os.replace(tmp, path)
