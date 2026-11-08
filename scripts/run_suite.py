"""Run a group of experiments (one paper table/figure) in parallel, skipping finished ones.

    python scripts/run_suite.py --suite quant_mnist --preset fast --jobs 2
    python scripts/run_suite.py --suite all_mnist   --preset fast --jobs 2
    python scripts/run_suite.py --suite quant_cifar --preset paper --jobs 2     # on a GPU (Kaggle)
    python scripts/run_suite.py --list

Each run writes results/<name>.json (updated during training) and results/logs/<name>.log.
Interrupted runs resume from results/ckpt/<name>.pt when you start the suite again.
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# preset -> common arguments
PRESETS = {
    # CPU laptop: same algorithm, shorter schedule. ~10-25 min per MNIST run on 4 cores.
    "fast": dict(rounds=150, local_epochs=2, eval_every=5, lip_batches=4),
    # Even shorter, for a first look / slides (~3-8 min per MNIST run).
    "tiny": dict(rounds=60, local_epochs=1, eval_every=5, lip_batches=2),
    # Paper protocol (Sec. IV-B): 1000 rounds, 5 local epochs. Use a GPU.
    "paper": dict(rounds=1000, local_epochs=5, eval_every=10, lip_batches=-1),
}

QUANTS = ["fp32", "int4", "int8", "int16", "cosine", "dynamic"]


def _suite(ds, n=100):
    """Experiment grids; each entry is a dict of run.py arguments."""
    big_eps = 1e6 if (ds == "cifar10" and n >= 1000) else 1e4  # paper: eps=1e6 for CIFAR10 @1000 clients
    return {
        # Table 3: gradient clipping bound, no DP noise
        f"clip_{ds}": [dict(dataset=ds, n_clients=n, quant="fp32", epsilon="inf", xi=x)
                       for x in ["10", "20", "50", "100", "inf"]],
        # Table 5 / Fig 5: privacy budget
        f"dp_{ds}": [dict(dataset=ds, n_clients=n, quant="fp32", epsilon=e, xi=100)
                     for e in ["inf", "5000", "7500", "10000"]],
        # Table 6 / Fig 8: quantization schemes under DP
        f"quant_{ds}": [dict(dataset=ds, n_clients=n, quant=q, epsilon=big_eps, xi=100) for q in QUANTS],
        # Fig 7: lambda_h of the Dynamic scheduler
        f"lambda_{ds}": [dict(dataset=ds, n_clients=n, quant="dynamic", epsilon=big_eps, xi=100, lambda_h=l)
                         for l in [0.25, 0.5, 0.75, 1.0]],
        # Table 4: number of clients (no DP)
        f"clients_{ds}": [dict(dataset=ds, n_clients=c, quant="fp32", epsilon="inf", xi=100)
                          for c in [50, 100, 200]],
        # EXTENSION for your title: degree of non-IID-ness (Dirichlet alpha) x {no DP, DP}
        f"noniid_{ds}": [dict(dataset=ds, n_clients=n, quant=q, epsilon=e, xi=100, partition="dirichlet", alpha=a)
                         for a in [0.1, 0.5, 1.0, 100.0] for q, e in [("fp32", "inf"), ("fp32", "10000"),
                                                                      ("dynamic", "10000")]],
    }


def all_suites():
    s = {"smoke": [dict(dataset="mnist", n_clients=100, quant="dynamic", epsilon=1e4, xi=100,
                        rounds=5, local_epochs=1, eval_every=1)]}
    for ds in ("mnist", "cifar10"):
        s.update(_suite(ds))
    for ds in ("mnist", "cifar10"):
        core = [f"clip_{ds}", f"dp_{ds}", f"quant_{ds}", f"lambda_{ds}"]
        s[f"all_{ds}"] = [r for k in core for r in s[k]]
    return s


def run_name(args: dict, suite: str) -> str:
    keys = ["dataset", "n_clients", "quant", "epsilon", "xi", "lambda_h", "partition", "alpha",
            "rounds", "local_epochs"]
    parts = [suite.replace("all_", "")] + [f"{k[:3]}{args[k]}" for k in keys if k in args]
    return "_".join(str(p) for p in parts).replace(".", "p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="smoke")
    ap.add_argument("--preset", default="fast", choices=list(PRESETS))
    ap.add_argument("--jobs", type=int, default=1, help="runs in parallel")
    ap.add_argument("--threads", type=int, default=0, help="torch threads per run (0 = cores/jobs)")
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--extra", default="", help="extra args passed to run.py, e.g. '--seed 1'")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    suites = all_suites()
    if a.list:
        for k, v in suites.items():
            print(f"{k:16s} {len(v):3d} runs")
        return
    runs = suites[a.suite]
    threads = a.threads or max(1, (os.cpu_count() or 2) // a.jobs)
    os.makedirs(os.path.join(a.out_dir, "logs"), exist_ok=True)

    cmds = []
    for r in runs:
        args = {**PRESETS[a.preset], **r}
        suite_tag = a.suite if a.suite != "smoke" else "smoke"
        # the suite prefix of the grid this run belongs to (for plotting)
        tag = next((k for k, v in suites.items() if r in v and not k.startswith("all_")), suite_tag)
        name = f"{a.preset}_" + run_name(args, tag)
        if a.extra.strip():  # keep e.g. '--mnist_source leaf' runs separate from the others
            name += "_" + "".join(ch for ch in a.extra.replace("--", "_") if ch.isalnum() or ch == "_").strip("_")
        res = os.path.join(a.out_dir, f"{name}.json")
        if os.path.exists(res) and json.load(open(res)).get("finished"):
            print(f"skip (done)  {name}")
            continue
        cmd = [sys.executable, os.path.join(ROOT, "run.py"), "--run_name", name, "--out_dir", a.out_dir,
               "--threads", str(threads)]
        for k, v in args.items():
            cmd += [f"--{k}", str(v)]
        cmd += a.extra.split()
        cmds.append((name, cmd))

    # download datasets once, before parallel jobs could race on the same files
    ex = a.extra.split()
    src = ex[ex.index("--mnist_source") + 1] if "--mnist_source" in ex else "torchvision"
    leaf = ex[ex.index("--leaf_dir") + 1] if "--leaf_dir" in ex else ""
    for ds in sorted({r["dataset"] for r in runs}):
        subprocess.run([sys.executable, "-c",
                        f"from fl.data import load_dataset; load_dataset('{ds}', './data', '{src}', '{leaf}')"],
                       cwd=ROOT, check=True)

    print(f"{len(cmds)} runs to do, {a.jobs} in parallel, {threads} threads each")
    running = []
    t0 = time.time()
    while cmds or running:
        while cmds and len(running) < a.jobs:
            name, cmd = cmds.pop(0)
            log = open(os.path.join(a.out_dir, "logs", f"{name}.log"), "a")
            print(f"start  {name}")
            running.append((name, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT), log))
        for item in list(running):
            name, p, log = item
            if p.poll() is not None:
                log.close()
                running.remove(item)
                status = "ok" if p.returncode == 0 else f"FAILED ({p.returncode}) - see results/logs/{name}.log"
                print(f"finish {name}: {status}  [{(time.time() - t0) / 60:.1f} min elapsed]")
        time.sleep(2)


if __name__ == "__main__":
    main()
