"""Run one federated experiment.

Examples
--------
# 2-minute smoke test
python run.py --dataset mnist --n_clients 100 --rounds 5 --local_epochs 1 --eval_every 1

# one row of Table 6 (CPU-friendly length)
python run.py --dataset mnist --n_clients 100 --quant dynamic --rounds 150 --local_epochs 2

# the paper's full protocol (use a GPU)
python run.py --dataset cifar10 --n_clients 100 --quant cosine --rounds 1000 --local_epochs 5
"""
import argparse
import dataclasses

from fl.federated import Config, run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in dataclasses.fields(Config):
        if f.type in ("bool", bool):
            ap.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"), default=f.default)
        else:
            typ = {"int": int, "float": float, "str": str}.get(f.type if isinstance(f.type, str) else f.type.__name__, str)
            ap.add_argument(f"--{f.name}", type=typ, default=f.default)
    cfg = Config(**vars(ap.parse_args()))
    res = run(cfg)
    s = res["summary"]
    print(f"\nDONE {res['name']}: best acc {s['best_acc'] * 100:.2f}% at round {s['best_round']}, "
          f"total {s['total_gib']:.2f} GiB, {s['wall_sec'] / 60:.1f} min")


if __name__ == "__main__":
    main()
