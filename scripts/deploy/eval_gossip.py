import argparse
import csv
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import requests


DEFAULT_BASE_URL = os.getenv("MEMLORA_BASE_URL", "http://127.0.0.1:5000")
DEFAULT_OUTPUT_ROOT = Path("results") / "gossip_eval"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate gossip freshness (staleness), queue estimate accuracy, and "
            "network overhead using /internal/cluster snapshots."
        )
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--sample-interval-s", type=float, default=0.2)
    parser.add_argument("--gossip-interval-s", type=float, default=0.15)
    parser.add_argument("--timeout-s", type=float, default=3.0)
    parser.add_argument("--serve-port", type=int, default=5000)
    parser.add_argument(
        "--http-overhead-bytes",
        type=int,
        default=700,
        help=(
            "Approximate per-message HTTP overhead for rough traffic estimate "
            "(headers + framing + TCP/IP metadata)."
        ),
    )
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if pct <= 0:
        return float(min(values))
    if pct >= 100:
        return float(max(values))

    sorted_vals = sorted(values)
    rank = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = rank - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def fetch_cluster_snapshot(base_url: str, timeout_s: float) -> dict:
    resp = requests.get(f"{base_url}/internal/cluster", timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


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


def estimate_payload_bytes(local_queue: int, sample_ts: float) -> int:
    msg = {
        "type": "queue_length",
        "node": "0.0.0.0",
        "queue_len": int(local_queue),
        "ts": float(sample_ts),
    }
    return len(json.dumps(msg, separators=(",", ":")).encode("utf-8"))


def sample_once(
    snapshot: dict,
    wall_ts: float,
    gossip_interval_s: float,
    http_overhead_bytes: int,
):
    nodes = snapshot.get("nodes", {})
    node_ips = sorted(nodes.keys())
    node_count = len(node_ips)

    staleness_rows = []
    error_rows = []
    payload_sum = 0
    active_senders = 0
    reachable_nodes = 0

    # Estimate per-target clock offset using freshest observer view for each target.
    # This lets us report skew-normalized staleness for cross-node comparisons.
    target_clock_offset_s: dict[str, float] = {}
    for observer in nodes.values():
        if observer.get("status") != "ok":
            continue
        peer_ts = observer.get("peer_timestamps", {}) or {}
        for target_ip, ts in peer_ts.items():
            if not isinstance(ts, (int, float)):
                continue
            offset = wall_ts - float(ts)
            best = target_clock_offset_s.get(target_ip)
            if best is None or offset < best:
                target_clock_offset_s[target_ip] = offset

    for observer_ip, observer in nodes.items():
        if observer.get("status") != "ok":
            continue
        reachable_nodes += 1

        local_queue = observer.get("local_queue")
        if isinstance(local_queue, int):
            payload_sum += estimate_payload_bytes(local_queue, wall_ts)
            active_senders += 1

        peer_ts = observer.get("peer_timestamps", {}) or {}
        peer_q = observer.get("peer_queues", {}) or {}

        for target_ip, ts in peer_ts.items():
            if not isinstance(ts, (int, float)):
                continue
            raw_age_s = max(0.0, wall_ts - float(ts))
            offset = target_clock_offset_s.get(target_ip, 0.0)
            normalized_age_s = max(0.0, raw_age_s - max(0.0, offset))
            staleness_rows.append(
                {
                    "sample_ts": wall_ts,
                    "observer_node": observer_ip,
                    "target_node": target_ip,
                    "staleness_s": round(raw_age_s, 6),
                    "staleness_normalized_s": round(normalized_age_s, 6),
                    "estimated_clock_offset_s": round(offset, 6),
                }
            )

            target = nodes.get(target_ip, {})
            true_q = target.get("local_queue")
            est_q = peer_q.get(target_ip)
            if isinstance(true_q, int) and isinstance(est_q, int):
                error_rows.append(
                    {
                        "sample_ts": wall_ts,
                        "observer_node": observer_ip,
                        "target_node": target_ip,
                        "estimated_queue": est_q,
                        "true_queue": true_q,
                        "abs_error": abs(est_q - true_q),
                        "staleness_s": round(raw_age_s, 6),
                        "staleness_normalized_s": round(normalized_age_s, 6),
                    }
                )

    edges = max(0, node_count - 1)
    msgs_per_second = (
        (active_senders * edges) / gossip_interval_s if gossip_interval_s > 0 else 0.0
    )
    payload_bytes_per_second = (
        (payload_sum * edges) / gossip_interval_s if gossip_interval_s > 0 else 0.0
    )
    total_bytes_per_second = (
        ((payload_sum + (http_overhead_bytes * active_senders)) * edges) / gossip_interval_s
        if gossip_interval_s > 0
        else 0.0
    )

    overhead_row = {
        "sample_ts": wall_ts,
        "node_count": node_count,
        "reachable_nodes": reachable_nodes,
        "active_senders": active_senders,
        "estimated_msgs_per_second": round(msgs_per_second, 3),
        "estimated_payload_bytes_per_second": round(payload_bytes_per_second, 3),
        "estimated_total_bytes_per_second": round(total_bytes_per_second, 3),
        "assumed_http_overhead_bytes_per_msg": http_overhead_bytes,
    }

    return staleness_rows, error_rows, overhead_row


def build_summary(
    started_at: float,
    ended_at: float,
    samples: int,
    staleness_rows: list[dict],
    error_rows: list[dict],
    overhead_rows: list[dict],
    args,
):
    staleness_vals = [float(r["staleness_s"]) for r in staleness_rows]
    staleness_norm_vals = [float(r["staleness_normalized_s"]) for r in staleness_rows]
    clock_offset_vals = [float(r["estimated_clock_offset_s"]) for r in staleness_rows]
    error_vals = [float(r["abs_error"]) for r in error_rows]
    msg_rate_vals = [float(r["estimated_msgs_per_second"]) for r in overhead_rows]
    payload_bps_vals = [float(r["estimated_payload_bytes_per_second"]) for r in overhead_rows]
    total_bps_vals = [float(r["estimated_total_bytes_per_second"]) for r in overhead_rows]
    reachable_vals = [float(r["reachable_nodes"]) for r in overhead_rows]
    cluster_node_vals = [float(r["node_count"]) for r in overhead_rows]

    expected_obs_per_sample = 0.0
    observed_obs_per_sample = 0.0
    coverage_ratio = 0.0
    if cluster_node_vals and samples > 0:
        mean_nodes = sum(cluster_node_vals) / len(cluster_node_vals)
        expected_obs_per_sample = max(0.0, mean_nodes * max(0.0, mean_nodes - 1.0))
        observed_obs_per_sample = len(staleness_rows) / samples
        coverage_ratio = (
            observed_obs_per_sample / expected_obs_per_sample
            if expected_obs_per_sample > 0
            else 0.0
        )

    warnings = []
    if staleness_vals and percentile(staleness_vals, 95) > 5.0:
        warnings.append(
            "Raw staleness appears very high; likely cross-node clock skew. "
            "Use staleness_normalized_s and sync node clocks (NTP/chrony)."
        )
    if coverage_ratio > 0 and coverage_ratio < 0.9:
        warnings.append(
            "Staleness coverage is below 90% of expected node-to-node observations. "
            "One or more nodes may be unreachable or missing peer timestamp entries."
        )

    summary = {
        "base_url": args.base_url,
        "duration_s_requested": args.duration_s,
        "sample_interval_s": args.sample_interval_s,
        "gossip_interval_s": args.gossip_interval_s,
        "timeout_s": args.timeout_s,
        "http_overhead_bytes_assumed": args.http_overhead_bytes,
        "serve_port": args.serve_port,
        "started_unix_s": started_at,
        "ended_unix_s": ended_at,
        "runtime_s": round(ended_at - started_at, 3),
        "sample_count": samples,
        "staleness_observation_count": len(staleness_rows),
        "queue_error_observation_count": len(error_rows),
        "reachability": {
            "cluster_nodes_mean": round(sum(cluster_node_vals) / len(cluster_node_vals), 3)
            if cluster_node_vals
            else 0.0,
            "reachable_nodes_mean": round(sum(reachable_vals) / len(reachable_vals), 3)
            if reachable_vals
            else 0.0,
            "expected_staleness_obs_per_sample": round(expected_obs_per_sample, 3),
            "observed_staleness_obs_per_sample": round(observed_obs_per_sample, 3),
            "staleness_observation_coverage_ratio": round(coverage_ratio, 6),
        },
        "staleness_s": {
            "p50": round(percentile(staleness_vals, 50), 6),
            "p95": round(percentile(staleness_vals, 95), 6),
            "p99": round(percentile(staleness_vals, 99), 6),
            "max": round(percentile(staleness_vals, 100), 6),
            "mean": round(sum(staleness_vals) / len(staleness_vals), 6)
            if staleness_vals
            else 0.0,
        },
        "staleness_normalized_s": {
            "p50": round(percentile(staleness_norm_vals, 50), 6),
            "p95": round(percentile(staleness_norm_vals, 95), 6),
            "p99": round(percentile(staleness_norm_vals, 99), 6),
            "max": round(percentile(staleness_norm_vals, 100), 6),
            "mean": round(sum(staleness_norm_vals) / len(staleness_norm_vals), 6)
            if staleness_norm_vals
            else 0.0,
        },
        "estimated_clock_offset_s": {
            "p50": round(percentile(clock_offset_vals, 50), 6),
            "p95": round(percentile(clock_offset_vals, 95), 6),
            "p99": round(percentile(clock_offset_vals, 99), 6),
            "max": round(percentile(clock_offset_vals, 100), 6),
            "mean": round(sum(clock_offset_vals) / len(clock_offset_vals), 6)
            if clock_offset_vals
            else 0.0,
        },
        "queue_abs_error": {
            "p50": round(percentile(error_vals, 50), 6),
            "p95": round(percentile(error_vals, 95), 6),
            "p99": round(percentile(error_vals, 99), 6),
            "max": round(percentile(error_vals, 100), 6),
            "mean": round(sum(error_vals) / len(error_vals), 6) if error_vals else 0.0,
        },
        "estimated_gossip_rate": {
            "msgs_per_sec_mean": round(sum(msg_rate_vals) / len(msg_rate_vals), 3)
            if msg_rate_vals
            else 0.0,
            "payload_bytes_per_sec_mean": round(
                sum(payload_bps_vals) / len(payload_bps_vals), 3
            )
            if payload_bps_vals
            else 0.0,
            "total_bytes_per_sec_mean": round(sum(total_bps_vals) / len(total_bps_vals), 3)
            if total_bps_vals
            else 0.0,
            "total_mib_per_min_mean": round(
                (
                    (sum(total_bps_vals) / len(total_bps_vals)) * 60.0 / (1024.0 * 1024.0)
                    if total_bps_vals
                    else 0.0
                ),
                6,
            ),
        },
    }
    if warnings:
        summary["warnings"] = warnings
    return summary


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    next_sample = started_at

    staleness_rows = []
    error_rows = []
    overhead_rows = []
    sample_errors = []
    samples = 0

    print(f"Base URL: {args.base_url}")
    print(f"Duration: {args.duration_s}s")
    print(f"Sample interval: {args.sample_interval_s}s")
    print(f"Gossip interval (assumed): {args.gossip_interval_s}s")
    print(f"Output dir: {output_dir}")

    while time.time() - started_at < args.duration_s:
        now = time.time()
        if now < next_sample:
            time.sleep(min(0.05, next_sample - now))
            continue

        sample_ts = time.time()
        try:
            snapshot = fetch_cluster_snapshot(args.base_url, args.timeout_s)
            sample_stale, sample_error, sample_overhead = sample_once(
                snapshot=snapshot,
                wall_ts=sample_ts,
                gossip_interval_s=args.gossip_interval_s,
                http_overhead_bytes=args.http_overhead_bytes,
            )
            staleness_rows.extend(sample_stale)
            error_rows.extend(sample_error)
            overhead_rows.append(sample_overhead)
            samples += 1

            print(
                f"[sample {samples:04d}] "
                f"staleness_obs={len(sample_stale)} "
                f"error_obs={len(sample_error)} "
                f"msgs_s={sample_overhead['estimated_msgs_per_second']:.1f} "
                f"total_Bps={sample_overhead['estimated_total_bytes_per_second']:.1f}"
            )
        except Exception as exc:
            sample_errors.append({"sample_ts": sample_ts, "error": str(exc)})
            print(f"[sample ERROR] {exc}")

        next_sample += args.sample_interval_s

    ended_at = time.time()
    summary = build_summary(
        started_at=started_at,
        ended_at=ended_at,
        samples=samples,
        staleness_rows=staleness_rows,
        error_rows=error_rows,
        overhead_rows=overhead_rows,
        args=args,
    )
    summary["sample_errors"] = sample_errors

    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "staleness.json", staleness_rows)
    write_json(output_dir / "queue_error.json", error_rows)
    write_json(output_dir / "overhead.json", overhead_rows)
    write_csv(output_dir / "staleness.csv", staleness_rows)
    write_csv(output_dir / "queue_error.csv", error_rows)
    write_csv(output_dir / "overhead.csv", overhead_rows)

    print("\nSummary")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
