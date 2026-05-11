#!/usr/bin/env python3
"""
plot_cost_sweep.py — plots latency metrics across cost model coefficient configs.

Usage:
    python3 plot_cost_sweep.py --results-dir ../../results_cost_finetuning
    python3 plot_cost_sweep.py --results-dir ../../results_cost_finetuning --workload zipf
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Config metadata ───────────────────────────────────────────────────────────
CONFIG_META = {
    "default":        {"wq": 0.40, "wm": 0.40, "wn": 0.20},
    "queue_heavy":    {"wq": 0.60, "wm": 0.20, "wn": 0.20},
    "memory_heavy":   {"wq": 0.20, "wm": 0.60, "wn": 0.20},
    "network_heavy":  {"wq": 0.20, "wm": 0.20, "wn": 0.60},
    "balanced":       {"wq": 0.34, "wm": 0.33, "wn": 0.33},
    "queue_memory":   {"wq": 0.45, "wm": 0.45, "wn": 0.10},
    "queue_network":  {"wq": 0.45, "wm": 0.10, "wn": 0.45},
    "memory_network": {"wq": 0.10, "wm": 0.45, "wn": 0.45},
}

WORKLOADS = ["zipf", "uniform", "burst"]
WORKLOAD_COLORS = {"zipf": "#4C72B0", "uniform": "#55A868", "burst": "#DD8452"}


def load_results(results_dir: Path, config: str, workload: str) -> dict | None:
    path = results_dir / config / workload / "results.jsonl"
    if not path.exists():
        return None

    latencies = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") == "summary":
                return {
                    "p50":  r.get("p50_ms"),
                    "p95":  r.get("p95_ms"),
                    "p99":  r.get("p99_ms"),
                    "mean": r.get("mean_ms"),
                    "total": r.get("total"),
                    "successful": r.get("successful"),
                    "forward_rate": r.get("forward_rate"),
                }
            if r.get("status") == 200:
                latencies.append(r["latency_ms"])

    if not latencies:
        return None
    return {
        "p50":  np.percentile(latencies, 50),
        "p95":  np.percentile(latencies, 95),
        "p99":  np.percentile(latencies, 99),
        "mean": np.mean(latencies),
        "total": None,
        "successful": len(latencies),
        "forward_rate": None,
    }


def collect_data(results_dir: Path, workloads: list) -> dict:
    data = defaultdict(dict)
    for config in CONFIG_META:
        for workload in workloads:
            result = load_results(results_dir, config, workload)
            if result:
                data[workload][config] = result
            else:
                print(f"  MISSING: {config}/{workload}")
    return data


def short_label(config: str) -> str:
    m = CONFIG_META[config]
    return f"{config}\nq={m['wq']} m={m['wm']} n={m['wn']}"


# ── Plot 1: grouped bar — p50 / mean / p95 / p99 per config ──────────────────
def plot_grouped_bar(data: dict, workload: str, out_dir: Path):
    configs = [c for c in CONFIG_META if c in data.get(workload, {})]
    if not configs:
        print(f"  No data for workload={workload}")
        return

    metrics = [("p50", "#4C72B0"), ("mean", "#8172B2"),
               ("p95", "#DD8452"), ("p99", "#C44E52")]
    x = np.arange(len(configs))
    width = 0.2

    fig, ax = plt.subplots(figsize=(15, 6))
    for i, (m, color) in enumerate(metrics):
        values = [data[workload][c].get(m) or 0 for c in configs]
        bars = ax.bar(x + i * width, values, width, label=m.upper(),
                      color=color, alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 3,
                    f"{val:.0f}",
                    ha="center", va="bottom", fontsize=6.5, rotation=45)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([short_label(c) for c in configs], fontsize=8)
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Latency Distribution by Cost Config — {workload.capitalize()} Workload",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Metric", fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    plt.tight_layout()

    out_path = out_dir / f"bar_{workload}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  saved: {out_path}")


# ── Plot 2: line chart — all metrics across configs, one subplot per workload ─
def plot_lines_per_workload(data: dict, workloads: list, out_dir: Path):
    configs = list(CONFIG_META.keys())
    metrics = [("p50", "#4C72B0", "o", "-"),
               ("mean", "#8172B2", "s", "--"),
               ("p95", "#DD8452", "^", "-"),
               ("p99", "#C44E52", "x", ":")]

    fig, axes = plt.subplots(1, len(workloads), figsize=(6 * len(workloads), 5),
                             sharey=True)
    if len(workloads) == 1:
        axes = [axes]

    for ax, workload in zip(axes, workloads):
        if workload not in data:
            ax.set_title(f"{workload.capitalize()} (no data)")
            continue

        for metric, color, marker, ls in metrics:
            values, valid = [], []
            for c in configs:
                v = data[workload].get(c, {}).get(metric)
                if v is not None:
                    values.append(v)
                    valid.append(c)
            if not values:
                continue
            ax.plot(range(len(valid)), values, marker=marker, linestyle=ls,
                    color=color, linewidth=2, label=metric.upper(), markersize=6)
            for i, v in enumerate(values):
                ax.annotate(f"{v:.0f}", (i, v),
                            textcoords="offset points", xytext=(0, 7),
                            ha="center", fontsize=6.5, color=color)

        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels([short_label(c) for c in configs],
                           fontsize=7, rotation=10, ha="right")
        ax.set_title(f"{workload.capitalize()}", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3, linestyle="--")
        ax.set_xlabel("Cost config")

    axes[0].set_ylabel("Latency (ms)")
    fig.suptitle("Latency Metrics Across Cost Configs", fontsize=13, fontweight="bold", y=1.02)
    handles = [mpatches.Patch(color=c, label=m.upper())
               for m, c, _, _ in metrics]
    fig.legend(handles=handles, loc="upper right", fontsize=9, title="Metric")
    plt.tight_layout()

    out_path = out_dir / "lines_all_workloads.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out_path}")


# ── Plot 3: heatmap — one panel per metric ────────────────────────────────────
def plot_heatmap_multi(data: dict, workloads: list, out_dir: Path):
    configs = list(CONFIG_META.keys())
    metrics = ["p50", "mean", "p95", "p99"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6))

    for ax, metric in zip(axes, metrics):
        matrix = np.full((len(configs), len(workloads)), np.nan)
        for j, workload in enumerate(workloads):
            for i, config in enumerate(configs):
                v = data.get(workload, {}).get(config, {}).get(metric)
                if v is not None:
                    matrix[i, j] = v

        im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(len(workloads)))
        ax.set_xticklabels([w.capitalize() for w in workloads], fontsize=9)
        ax.set_yticks(range(len(configs)))
        ax.set_yticklabels([short_label(c) for c in configs], fontsize=7)
        ax.set_title(metric.upper(), fontsize=11, fontweight="bold")

        # annotate cells
        vmin = np.nanmin(matrix)
        vmax = np.nanmax(matrix)
        for i in range(len(configs)):
            for j in range(len(workloads)):
                if not np.isnan(matrix[i, j]):
                    # white text on dark cells, black on light
                    norm = (matrix[i, j] - vmin) / (vmax - vmin + 1e-9)
                    text_color = "white" if norm > 0.6 else "black"
                    ax.text(j, i, f"{matrix[i,j]:.0f}",
                            ha="center", va="center", fontsize=8,
                            color=text_color, fontweight="bold")

        plt.colorbar(im, ax=ax, label="ms", shrink=0.8)

    fig.suptitle("Latency Heatmap — Cost Config × Workload",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = out_dir / "heatmap_all_metrics.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out_path}")


# ── Plot 4: mean vs p95 scatter — one point per config, colored by workload ───
def plot_mean_vs_p95(data: dict, workloads: list, out_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 6))

    for workload in workloads:
        if workload not in data:
            continue
        color = WORKLOAD_COLORS[workload]
        for config, d in data[workload].items():
            mean = d.get("mean")
            p95  = d.get("p95")
            if mean is None or p95 is None:
                continue
            ax.scatter(mean, p95, color=color, s=80, zorder=3)
            ax.annotate(config, (mean, p95),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=7, color=color)

    # legend for workloads
    handles = [mpatches.Patch(color=WORKLOAD_COLORS[w], label=w.capitalize())
               for w in workloads if w in data]
    ax.legend(handles=handles, title="Workload", fontsize=9)

    ax.set_xlabel("Mean Latency (ms)", fontsize=11)
    ax.set_ylabel("P95 Latency (ms)", fontsize=11)
    ax.set_title("Mean vs P95 Latency — Cost Config Comparison",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3, linestyle="--")

    # diagonal reference line: p95 = mean (perfect consistency)
    all_vals = [d.get("mean") or d.get("p95")
                for w in workloads for d in data.get(w, {}).values()
                if d.get("mean") and d.get("p95")]
    if all_vals:
        lo, hi = min(all_vals) * 0.98, max(all_vals) * 1.02
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.2, linewidth=1,
                label="mean=p95")

    plt.tight_layout()
    out_path = out_dir / "scatter_mean_vs_p95.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  saved: {out_path}")


# ── Print summary table ───────────────────────────────────────────────────────
def print_summary_table(data: dict, workloads: list):
    configs = list(CONFIG_META.keys())
    col = 16
    header = f"{'config':<20}" + "".join(
        f"  {(w+':p50'):<{col}} {(w+':mean'):<{col}} {(w+':p95'):<{col}}"
        for w in workloads
    )
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for config in configs:
        row = f"{config:<20}"
        for workload in workloads:
            d = data.get(workload, {}).get(config, {})
            for m in ("p50", "mean", "p95"):
                v = d.get(m)
                row += f"  {v:>{col-2}.0f}ms" if v else f"  {'N/A':>{col-2}}"
        print(row)
    print(sep)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results_cost_finetuning")
    parser.add_argument("--workload", default="all",
                        choices=["all"] + WORKLOADS)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    workloads = WORKLOADS if args.workload == "all" else [args.workload]

    print(f"==> Loading results from {results_dir}")
    data = collect_data(results_dir, workloads)

    print_summary_table(data, workloads)

    print(f"\n==> Generating plots → {out_dir}")

    # 1. grouped bar per workload (p50 / mean / p95 / p99)
    for workload in workloads:
        plot_grouped_bar(data, workload, out_dir)

    # 2. line chart — all metrics, one subplot per workload
    plot_lines_per_workload(data, workloads, out_dir)

    # 3. heatmap — one panel per metric
    plot_heatmap_multi(data, workloads, out_dir)

    # 4. scatter — mean vs p95, colored by workload
    plot_mean_vs_p95(data, workloads, out_dir)

    n = len(list(out_dir.glob("*.png")))
    print(f"\n==> Done. {n} plots saved to {out_dir}")


if __name__ == "__main__":
    main()