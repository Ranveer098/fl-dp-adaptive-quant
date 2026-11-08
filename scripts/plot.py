"""Summarize results/*.json into tables (CSV + Markdown) and figures.

    python scripts/plot.py                # reads ./results, writes ./results/figures + summary files
    python scripts/plot.py --results results_kaggle
"""
import argparse
import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Fixed categorical order (never cycled); line style is a second encoding for print / colour-blind readers.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
QUANT_ORDER = ["fp32", "int4", "int8", "int16", "cosine", "dynamic"]
VARY_KEYS = ["quant", "epsilon", "xi", "lambda_h", "n_clients", "alpha", "partition"]


def load(results_dir):
    runs = []
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            r = json.load(open(p))
        except json.JSONDecodeError:
            continue
        if not r.get("history", {}).get("round"):
            continue
        parts = r["name"].split("_")
        r["group"] = "_".join(parts[:3]) if len(parts) >= 3 else r["name"]  # e.g. fast_quant_mnist
        runs.append(r)
    return runs


def sort_key(r):
    c = r["config"]
    q = QUANT_ORDER.index(c["quant"]) if c["quant"] in QUANT_ORDER else 99
    num = lambda v: float("inf") if v == "inf" else float(v)
    return (q, num(c["epsilon"]), num(c["xi"]), c["lambda_h"], c["n_clients"], c["alpha"])


def label(r, keys):
    c = r["config"]
    if not keys:
        return c["quant"]
    return ", ".join(f"{k}={c[k]}" if k != "quant" else str(c[k]) for k in keys)


def varying(group_runs):
    return [k for k in VARY_KEYS if len({str(r["config"][k]) for r in group_runs}) > 1]


def write_tables(runs, out_dir):
    rows = []
    for g in sorted({r["group"] for r in runs}):
        gr = sorted([r for r in runs if r["group"] == g], key=sort_key)
        fp32 = {(r["config"]["epsilon"], r["config"]["n_clients"], r["config"]["alpha"], r["config"]["partition"]):
                r["summary"]["total_gib"] for r in gr if r["config"]["quant"] == "fp32"}
        for r in gr:
            c, s = r["config"], r["summary"]
            base = fp32.get((c["epsilon"], c["n_clients"], c["alpha"], c["partition"]))
            red = (1 - s["total_gib"] / base) * 100 if base else None
            rows.append({
                "group": g, "dataset": c["dataset"], "clients": c["n_clients"], "quant": c["quant"],
                "epsilon": c["epsilon"], "xi": c["xi"], "lambda_h": c["lambda_h"],
                "partition": c["partition"], "alpha": c["alpha"], "rounds": c["rounds"], "E": c["local_epochs"],
                "best_acc_%": round(s["best_acc"] * 100, 2), "best_round": s["best_round"],
                "final_acc_%": round(s["final_acc"] * 100, 2), "total_GiB": round(s["total_gib"], 3),
                "comm_reduction_vs_fp32_%": None if red is None else round(red, 2),
                "minutes": round(s["wall_sec"] / 60, 1), "finished": r["finished"],
            })
    if not rows:
        print("no results yet")
        return
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cols = ["group", "clients", "quant", "epsilon", "xi", "lambda_h", "alpha", "best_acc_%", "best_round",
            "total_GiB", "comm_reduction_vs_fp32_%", "minutes", "finished"]
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for row in rows:
            f.write("| " + " | ".join("" if row[c] is None else str(row[c]) for c in cols) + " |\n")
    print(open(os.path.join(out_dir, "summary.md")).read())


def _style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel("communication round")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e6e6e3", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9a9a94")


def plot_group(g, gr, fig_dir):
    keys = varying(gr)
    gr = sorted(gr, key=sort_key)[:8]  # at most 8 series per chart
    # accuracy curves
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, r in enumerate(gr):
        h = r["history"]
        ax.plot(h["round"], [a * 100 for a in h["test_acc"]], color=PALETTE[i], linestyle=STYLES[i % len(STYLES)],
                linewidth=1.8, label=label(r, keys))
    _style(ax, f"{g}: global test accuracy", "test accuracy (%)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{g}_accuracy.png"), dpi=160)
    plt.close(fig)
    # bit-length per round (only meaningful when quantization varies)
    if "quant" in keys or any(r["config"]["quant"] in ("cosine", "dynamic") for r in gr):
        fig, ax = plt.subplots(figsize=(7, 3.6))
        for i, r in enumerate(gr):
            h = r["history"]
            ax.plot(h["round"], h["avg_up_bits"], color=PALETTE[i], linestyle=STYLES[i % len(STYLES)],
                    linewidth=1.8, label=label(r, keys))
        _style(ax, f"{g}: average client upload bit-length", "bits per weight")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{g}_bits.png"), dpi=160)
        plt.close(fig)
    # accuracy vs total communication (one point per run)
    if "quant" in keys:
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, r in enumerate(gr):
            ax.scatter(r["summary"]["total_gib"], r["summary"]["best_acc"] * 100, s=60, color=PALETTE[i],
                       edgecolor="white", linewidth=2, zorder=3, label=label(r, keys))
        _style(ax, f"{g}: accuracy vs communication", "best test accuracy (%)")
        ax.set_xlabel("total communicated GiB (up + down)")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{g}_tradeoff.png"), dpi=160)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    runs = load(a.results)
    fig_dir = os.path.join(a.results, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    write_tables(runs, a.results)
    for g in sorted({r["group"] for r in runs}):
        plot_group(g, [r for r in runs if r["group"] == g], fig_dir)
    print(f"figures -> {fig_dir}")


if __name__ == "__main__":
    main()
