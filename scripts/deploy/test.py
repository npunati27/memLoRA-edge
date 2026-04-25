import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import requests


DEFAULT_BASE_URL = os.getenv("MEMLORA_BASE_URL", "http://127.0.0.1:5000")
DEFAULT_OUTPUT_ROOT = Path("results") / "s3_cold_start"
DEFAULT_HOME = Path(os.path.expanduser("~"))
DEFAULT_ADAPTER_ROOT = DEFAULT_HOME / "adapters"
TIER_ORDER = ["gpu", "cpu", "disk", "s3"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a cluster-aware S3 cold-start evaluation and produce graphs showing "
            "requests moving from S3-backed cold loads to local GPU/CPU/disk tiers."
        )
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--adapters", default="")
    parser.add_argument("--num-adapters", type=int, default=8)
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--sleep-s", type=float, default=0.2)
    parser.add_argument("--timeout-s", type=float, default=240.0)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--clear-local-cache", action="store_true")
    parser.add_argument("--reset-cluster-cache", action="store_true")
    parser.add_argument("--snapshot-cluster", action="store_true", dest="snapshot_cluster")
    parser.add_argument("--no-snapshot-cluster", action="store_false", dest="snapshot_cluster")
    parser.set_defaults(snapshot_cluster=True)
    parser.add_argument("--adapter-root", default=str(DEFAULT_ADAPTER_ROOT))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def fetch_adapter_names(base_url: str, timeout_s: float) -> list[str]:
    resp = requests.get(f"{base_url}/v1/models", timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    adapters = []
    for entry in data:
        model_id = entry.get("id", "")
        if "/" in model_id:
            adapters.append(model_id.split("/", 1)[1])
    if not adapters:
        raise RuntimeError("No adapter-backed models were returned by /v1/models")
    return sorted(set(adapters))


def choose_adapters(args) -> list[str]:
    if args.adapters.strip():
        adapters = [item.strip() for item in args.adapters.split(",") if item.strip()]
        if not adapters:
            raise RuntimeError("The --adapters list was provided but no valid names were found")
        return adapters

    adapters = fetch_adapter_names(args.base_url, args.timeout_s)
    return adapters[: min(args.num_adapters, len(adapters))]


def clear_local_cache(adapter_root: Path, adapters: list[str]) -> list[str]:
    cleared = []
    for adapter in adapters:
        path = adapter_root / adapter
        if path.exists():
            shutil.rmtree(path)
            cleared.append(adapter)
    return sorted(cleared)


def reset_cluster_cache(base_url: str, adapters: list[str], timeout_s: float) -> dict:
    resp = requests.post(
        f"{base_url}/internal/debug/reset_cache",
        json={"adapters": adapters, "fanout": True},
        timeout=timeout_s,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_cluster_snapshot(base_url: str, timeout_s: float) -> dict:
    resp = requests.get(f"{base_url}/internal/cluster", timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def summarize_cluster_snapshot(snapshot: dict, adapters: list[str], after_request_index: int) -> dict:
    tracked = set(adapters)
    tier_counts = {tier: 0 for tier in TIER_ORDER}
    node_status = {}
    node_tier_counts = {}

    for node_ip, node_data in snapshot.get("nodes", {}).items():
        node_status[node_ip] = node_data.get("status", "unknown")
        local_adapters = node_data.get("local_adapters", {})
        tier_detail = {tier: 0 for tier in TIER_ORDER}

        for tier in TIER_ORDER:
            present = set(local_adapters.get(tier, []))
            count = len(present & tracked)
            tier_counts[tier] += count
            tier_detail[tier] = count

        node_tier_counts[node_ip] = tier_detail

    tier_counts["s3"] = max(
        0,
        len(tracked)
        - tier_counts["gpu"]
        - tier_counts["cpu"]
        - tier_counts["disk"],
    )
    return {
        "after_request_index": after_request_index,
        "ts_unix": snapshot.get("ts", time.time()),
        "source_node": snapshot.get("source_node", "unknown"),
        "tier_counts": tier_counts,
        "node_status": node_status,
        "node_tier_counts": node_tier_counts,
    }


def build_schedule(adapters: list[str], passes: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    schedule = []
    first_pass = list(adapters)
    schedule.extend(first_pass)
    for _ in range(max(0, passes - 1)):
        shuffled = list(adapters)
        rng.shuffle(shuffled)
        schedule.extend(shuffled)
    return schedule


def send_request(
    base_url: str,
    adapter_name: str,
    timeout_s: float,
    max_tokens: int,
    temperature: float,
    request_id: str,
):
    body = {
        "model": f"qwen-base/{adapter_name}",
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Give a short two-sentence summary for adapter '{adapter_name}' "
                    f"to exercise LoRA loading."
                ),
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "request_id": request_id,
    }

    start = time.perf_counter()
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json=body,
        timeout=timeout_s,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    payload = {}
    try:
        payload = resp.json()
    except Exception:
        payload = {"raw_text": resp.text}
    return resp.status_code, payload, latency_ms


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def flatten_snapshots(snapshots: list[dict]) -> list[dict]:
    rows = []
    for snap in snapshots:
        row = {
            "after_request_index": snap["after_request_index"],
            "ts_unix": snap["ts_unix"],
        }
        for tier in TIER_ORDER:
            row[f"{tier}_count"] = snap["tier_counts"].get(tier, 0)
        rows.append(row)
    return rows


def plot_results(output_dir: Path, rows: list[dict], snapshots: list[dict], adapters: list[str]):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Plot skipped: matplotlib is unavailable ({exc})")
        return None

    fig, axes = plt.subplots(4, 1, figsize=(14, 15), sharex=False)

    indices = [row["request_index"] for row in rows]
    latencies = [row["latency_ms"] for row in rows]
    colors = [
        "#d62728" if row["adapter_source"] == "s3" else "#1f77b4"
        for row in rows
    ]

    axes[0].plot(indices, latencies, color="#7f7f7f", alpha=0.45, linewidth=1.3)
    axes[0].scatter(indices, latencies, c=colors, s=62, edgecolors="black", linewidths=0.3)
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("S3 Cold Start Across Cluster")
    axes[0].grid(alpha=0.25)
    axes[0].scatter([], [], c="#d62728", label="Served after S3 fetch")
    axes[0].scatter([], [], c="#1f77b4", label="Served from local node cache")
    axes[0].legend(loc="upper right")

    rolling_s3_share = []
    rolling_latency = []
    window = max(3, min(7, len(rows)))
    for idx, _ in enumerate(rows):
        win_rows = rows[max(0, idx - window + 1): idx + 1]
        s3_in_window = sum(1 for item in win_rows if item["adapter_source"] == "s3")
        rolling_s3_share.append(s3_in_window / len(win_rows))
        rolling_latency.append(sum(item["latency_ms"] for item in win_rows) / len(win_rows))

    axes[1].plot(indices, rolling_s3_share, color="#d62728", linewidth=2.0, label="Rolling S3 share")
    axes[1].fill_between(indices, rolling_s3_share, color="#ff9896", alpha=0.35)
    axes[1].set_ylabel("S3 Share")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(alpha=0.25)
    twin = axes[1].twinx()
    twin.plot(indices, rolling_latency, color="#2ca02c", linewidth=1.8, linestyle="--", label="Rolling avg latency")
    twin.set_ylabel("Rolling Latency (ms)")

    snap_x = [snap["after_request_index"] for snap in snapshots]
    for tier, color in [
        ("s3", "#d62728"),
        ("disk", "#ff7f0e"),
        ("cpu", "#9467bd"),
        ("gpu", "#2ca02c"),
    ]:
        axes[2].plot(
            snap_x,
            [snap["tier_counts"].get(tier, 0) for snap in snapshots],
            linewidth=2.2,
            color=color,
            label=f"{tier.upper()} placements",
        )
    axes[2].set_ylabel("Adapter-Node Placements")
    axes[2].set_xlabel("After Request")
    axes[2].grid(alpha=0.25)
    axes[2].legend(loc="upper right")

    cumulative_source = defaultdict(int)
    cumulative_s3 = []
    cumulative_local = []
    for row in rows:
        cumulative_source[row["adapter_source"]] += 1
        cumulative_s3.append(cumulative_source.get("s3", 0))
        cumulative_local.append(cumulative_source.get("local", 0))

    axes[3].plot(indices, cumulative_s3, color="#d62728", linewidth=2.0, label="Cumulative S3-served requests")
    axes[3].plot(indices, cumulative_local, color="#1f77b4", linewidth=2.0, label="Cumulative local-served requests")
    axes[3].set_xlabel("Request Index")
    axes[3].set_ylabel("Requests")
    axes[3].grid(alpha=0.25)
    axes[3].legend(loc="upper left")

    fig.text(
        0.01,
        0.01,
        (
            f"Adapters: {', '.join(adapters)}\n"
            "Snapshot lines show where adapter replicas live across the cluster after each request."
        ),
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    output_path = output_dir / "cold_start_timeline.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main():
    args = parse_args()
    adapter_root = Path(args.adapter_root)
    adapters = choose_adapters(args)
    schedule = build_schedule(adapters, args.passes, args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    local_cleared = []
    if args.clear_local_cache:
        local_cleared = clear_local_cache(adapter_root, adapters)

    cluster_reset_result = {}
    if args.reset_cluster_cache:
        cluster_reset_result = reset_cluster_cache(args.base_url, adapters, args.timeout_s)

    print(f"Base URL: {args.base_url}")
    print(f"Adapters under test ({len(adapters)}): {', '.join(adapters)}")
    print(f"Passes: {args.passes}")
    print(f"Output dir: {output_dir}")
    if args.clear_local_cache:
        print(f"Locally cleared {len(local_cleared)} adapters from {adapter_root}")
    if args.reset_cluster_cache:
        print("Cluster cache reset requested across peers")

    rows = []
    snapshots = []
    request_failures = 0
    source_counts = defaultdict(int)

    if args.snapshot_cluster:
        snapshots.append(
            summarize_cluster_snapshot(
                fetch_cluster_snapshot(args.base_url, args.timeout_s),
                adapters,
                after_request_index=0,
            )
        )

    experiment_start = time.time()
    for request_index, adapter_name in enumerate(schedule, start=1):
        request_id = f"cold-start-{timestamp}-{request_index:04d}"
        status_code, payload, latency_ms = send_request(
            base_url=args.base_url,
            adapter_name=adapter_name,
            timeout_s=args.timeout_s,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            request_id=request_id,
        )

        adapter_source = payload.get("adapter_source", "unknown")
        adapter_load_ms = payload.get("adapter_load_ms", 0.0)
        served_by = payload.get("served_by", "unknown")
        tier_before = payload.get("tier_before")
        source_counts[adapter_source] += 1

        snapshot_summary = None
        if args.snapshot_cluster:
            snapshot_summary = summarize_cluster_snapshot(
                fetch_cluster_snapshot(args.base_url, args.timeout_s),
                adapters,
                after_request_index=request_index,
            )
            snapshots.append(snapshot_summary)

        row = {
            "request_index": request_index,
            "request_id": request_id,
            "adapter_name": adapter_name,
            "status_code": status_code,
            "latency_ms": round(latency_ms, 3),
            "adapter_source": adapter_source,
            "adapter_load_ms": round(float(adapter_load_ms), 3),
            "served_by": served_by,
            "tier_before": tier_before,
            "ts_unix": time.time(),
            "response_error": payload.get("error", ""),
        }

        if snapshot_summary is not None:
            for tier in TIER_ORDER:
                row[f"cluster_{tier}_count"] = snapshot_summary["tier_counts"].get(tier, 0)

        rows.append(row)

        if status_code != 200:
            request_failures += 1

        cluster_suffix = ""
        if snapshot_summary is not None:
            cluster_suffix = (
                f" cluster[gpu={snapshot_summary['tier_counts']['gpu']},"
                f" cpu={snapshot_summary['tier_counts']['cpu']},"
                f" disk={snapshot_summary['tier_counts']['disk']},"
                f" s3={snapshot_summary['tier_counts']['s3']}]"
            )

        print(
            f"[{request_index:03d}/{len(schedule):03d}] "
            f"adapter={adapter_name} status={status_code} "
            f"source={adapter_source} load_ms={row['adapter_load_ms']:.1f} "
            f"e2e_ms={row['latency_ms']:.1f} served_by={served_by}{cluster_suffix}"
        )

        if args.sleep_s > 0:
            time.sleep(args.sleep_s)

    total_runtime_s = time.time() - experiment_start
    plot_path = plot_results(output_dir, rows, snapshots, adapters)

    summary = {
        "base_url": args.base_url,
        "adapters": adapters,
        "passes": args.passes,
        "request_count": len(rows),
        "failures": request_failures,
        "sleep_s": args.sleep_s,
        "timeout_s": args.timeout_s,
        "clear_local_cache": args.clear_local_cache,
        "reset_cluster_cache": args.reset_cluster_cache,
        "locally_cleared_adapters": local_cleared,
        "cluster_reset_result": cluster_reset_result,
        "source_counts": dict(source_counts),
        "total_runtime_s": round(total_runtime_s, 3),
        "avg_latency_ms": round(sum(row["latency_ms"] for row in rows) / len(rows), 3),
        "avg_adapter_load_ms": round(sum(row["adapter_load_ms"] for row in rows) / len(rows), 3),
        "plot_path": str(plot_path) if plot_path else "",
        "initial_cluster_tiers": snapshots[0]["tier_counts"] if snapshots else {},
        "final_cluster_tiers": snapshots[-1]["tier_counts"] if snapshots else {},
    }

    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "results.json", rows)
    write_json(output_dir / "cluster_snapshots.json", snapshots)
    write_csv(output_dir / "results.csv", rows)
    write_csv(output_dir / "cluster_snapshots.csv", flatten_snapshots(snapshots))

    print("\nSummary")
    print(json.dumps(summary, indent=2))
    if plot_path:
        print(f"\nGraph saved to: {plot_path}")
    print(f"Raw results saved to: {output_dir / 'results.csv'}")

    if request_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
