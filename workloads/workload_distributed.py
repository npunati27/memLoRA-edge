#!/usr/bin/env python3
"""
workload_distributed.py — distributed workload generator for memLoRA

Usage:
    python3 workload_distributed.py --mode zipf --rps 4 --duration 120
    python3 workload_distributed.py --mode zipf --num-nodes 4 --rps 4 --duration 120
    python3 workload_distributed.py --mode all --num-nodes 20 --rps 4 --duration 120
"""

import argparse, asyncio, aiohttp, time, json, random, uuid
import numpy as np
from collections import defaultdict

# ── All 20 nodes ──────────────────────────────────────────────────────────────
ALL_NODES = [
    "http://sp26-cs525-0701.cs.illinois.edu:5000",
    "http://sp26-cs525-0702.cs.illinois.edu:5000",
    "http://sp26-cs525-0703.cs.illinois.edu:5000",
    "http://sp26-cs525-0704.cs.illinois.edu:5000",
    "http://sp26-cs525-0705.cs.illinois.edu:5000",
    "http://sp26-cs525-0706.cs.illinois.edu:5000",
    "http://sp26-cs525-0707.cs.illinois.edu:5000",
    "http://sp26-cs525-0708.cs.illinois.edu:5000",
    "http://sp26-cs525-0710.cs.illinois.edu:5000",
    "http://sp26-cs525-0711.cs.illinois.edu:5000",
    "http://sp26-cs525-0712.cs.illinois.edu:5000",
    "http://sp26-cs525-0713.cs.illinois.edu:5000",
    "http://sp26-cs525-0714.cs.illinois.edu:5000",
    "http://sp26-cs525-0715.cs.illinois.edu:5000",
    "http://sp26-cs525-0716.cs.illinois.edu:5000",
    "http://sp26-cs525-0717.cs.illinois.edu:5000",
    "http://sp26-cs525-0718.cs.illinois.edu:5000",
    "http://sp26-cs525-0719.cs.illinois.edu:5000",
    "http://sp26-cs525-0720.cs.illinois.edu:5000",
]

DEFAULT_NUM_NODES = 8

# ── Adapters ──────────────────────────────────────────────────────────────────
ALL_ADAPTERS = [
    "crop_corn_disease","crop_wheat_disease","crop_soy_disease",
    "crop_tomato_disease","crop_cotton_disease","crop_general_health",
    "crop_rice_disease","crop_barley_disease","crop_oat_disease",
    "crop_potato_disease","crop_berry_disease","crop_grape_disease",
    "crop_citrus_disease","crop_apple_disease","crop_peach_disease",
    "crop_canola_disease","crop_sunflower_disease","crop_alfalfa_health",
    "crop_pasture_health","crop_yield_pred_north","crop_yield_pred_south",
    "crop_stress_heat","crop_stress_drought","crop_ndvi_zones",
    "crop_growth_stage","crop_harvest_window",
    "pest_aphid","pest_rootworm","pest_spider_mite","pest_caterpillar","pest_general",
    "pest_grasshopper","pest_weevil","pest_thrips","pest_whitefly","pest_cutworm",
    "pest_borer","pest_leafhopper","pest_slug","pest_snail","pest_ant",
    "pest_bee_health","pest_beneficial_count",
    "soil_nitrogen","soil_phosphorus","soil_moisture","soil_ph",
    "soil_potassium","soil_organic_matter","soil_compaction",
    "soil_salinity","soil_erosion_risk","soil_temp_root",
    "soil_n_source","soil_microbiome","soil_carbon_estimate",
    "irrigation_zone_a","irrigation_zone_b","irrigation_zone_c","irrigation_zone_d",
    "irrigation_zone_e","irrigation_zone_f","irrigation_sched_block1",
    "irrigation_sched_block2","irrigation_drip_health","irrigation_sprinkler_uniform",
    "irrigation_water_quality","irrigation_pressure","irrigation_flow_meter",
    "irrigation_leak_detect",
    "weather_forecast","weather_frost_alert","weather_humidity",
    "weather_wind_alert","weather_hail_risk","weather_precipitation",
    "weather_heat_index","weather_dew_point","weather_soil_temp",
    "weather_evapotranspiration",
    "equip_tractor","equip_drone","equip_sensor",
    "equip_harvester","equip_planter","equip_sprayer",
    "equip_baler","equip_spreader","equip_gps_guidance",
    "equip_yield_monitor","equip_fuel_telemetry",
    "livestock_cattle_health","livestock_poultry","livestock_pasture_rotation",
    "dairy_milk_quality","dairy_feed_ration","grain_storage_temp",
    "grain_moisture_bin","carbon_footprint_field","nutrient_runoff_risk",
]

HOT_ADAPTERS  = ALL_ADAPTERS[:3]
WARM_ADAPTERS = ALL_ADAPTERS[3:9]
COLD_ADAPTERS = ALL_ADAPTERS[9:]

PROMPTS = {
    "crop_corn_disease":    "What are the symptoms of northern corn leaf blight?",
    "crop_wheat_disease":   "Describe wheat rust disease progression.",
    "crop_soy_disease":     "What causes soybean sudden death syndrome?",
    "crop_tomato_disease":  "How does early blight affect tomato plants?",
    "crop_cotton_disease":  "Describe cotton root rot symptoms.",
    "crop_general_health":  "What indicates a healthy crop stand?",
    "pest_aphid":           "How do aphids damage crops?",
    "pest_rootworm":        "Describe corn rootworm lifecycle.",
    "pest_spider_mite":     "What crops are most affected by spider mites?",
    "pest_caterpillar":     "How do caterpillars damage soybean leaves?",
    "pest_general":         "What are common signs of pest infestation?",
    "soil_nitrogen":        "What are symptoms of nitrogen deficiency?",
    "soil_phosphorus":      "How does phosphorus deficiency affect plant growth?",
    "soil_moisture":        "What is the optimal soil moisture for corn?",
    "soil_ph":              "How does soil pH affect nutrient availability?",
    "irrigation_zone_a":    "What is the irrigation schedule for zone A?",
    "irrigation_zone_b":    "Describe zone B water requirements.",
    "irrigation_zone_c":    "When should zone C be irrigated?",
    "irrigation_zone_d":    "What triggers zone D irrigation?",
    "weather_forecast":     "What weather conditions favor disease spread?",
    "weather_frost_alert":  "At what temperature should frost protection begin?",
    "weather_humidity":     "How does humidity affect fungal disease risk?",
    "equip_tractor":        "What is the maintenance schedule for a tractor?",
    "equip_drone":          "How should agricultural drones be calibrated?",
    "equip_sensor":         "What do soil sensor readings indicate?",
}

# ── Distributions ─────────────────────────────────────────────────────────────

def zipf_distribution(adapters: list, s: float = 1.5) -> list[float]:
    ranks = np.arange(1, len(adapters) + 1)
    weights = 1.0 / (ranks ** s)
    return (weights / weights.sum()).tolist()

def uniform_distribution(adapters: list) -> list[float]:
    n = len(adapters)
    return [1.0 / n] * n

# ── Node selection ────────────────────────────────────────────────────────────

def pick_node(nodes: list, strategy: str = "random") -> str:
    if strategy == "random":
        return random.choice(nodes)
    return random.choice(nodes)

# ── Request sender ────────────────────────────────────────────────────────────

async def send_request(
    session: aiohttp.ClientSession,
    nodes: list,
    node_strategy: str,
    adapter_name: str,
    results: list,
    semaphore: asyncio.Semaphore,
):
    async with semaphore:
        target_node = pick_node(nodes, node_strategy)
        model = f"qwen-base/{adapter_name}" if adapter_name else "qwen-base"
        prompt = PROMPTS.get(adapter_name, "Describe this agricultural topic.")
        body = {
            "model":       model,
            "messages":    [{"role": "user", "content": prompt}],
            "max_tokens":  32,
            "temperature": 0.0,
            "request_id":  str(uuid.uuid4()),
        }

        start = time.perf_counter()
        status = None
        served_by = None
        sent_to = target_node
        error = None

        try:
            async with session.post(
                f"{target_node}/v1/chat/completions",
                json=body,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                status = resp.status
                data = await resp.json()
                served_by = data.get("served_by")
                if status != 200:
                    error = data.get("error")
        except Exception as e:
            error = str(e)
            status = 0

        latency_ms = (time.perf_counter() - start) * 1000

        results.append({
            "ts":        time.time(),
            "adapter":   adapter_name,
            "latency_ms": latency_ms,
            "status":    status,
            "sent_to":   sent_to,
            "served_by": served_by,
            "forwarded": sent_to != served_by and served_by is not None,
            "error":     error,
        })

        if error:
            print(f"  ERR  {adapter_name:<30} {latency_ms:6.0f}ms  sent_to={sent_to}  {error}")
        else:
            forwarded = sent_to != served_by and served_by is not None
            fwd_marker = "→" if forwarded else " "
            print(
                f"  OK {fwd_marker} {adapter_name:<30} {latency_ms:6.0f}ms  "
                f"sent={sent_to.split('//')[1].split(':')[0].split('.')[0]}  "
                f"served={str(served_by).split('.')[0] if served_by else '?'}"
            )

# ── Workload runners ──────────────────────────────────────────────────────────

async def run_uniform(session, nodes, node_strategy, rps, duration, results, semaphore):
    print(f"\n[uniform] {rps} rps for {duration}s across {len(nodes)} nodes — equal adapter probability")
    weights = uniform_distribution(ALL_ADAPTERS)
    interval = 1.0 / rps
    end = time.time() + duration
    tasks = []
    while time.time() < end:
        adapter = random.choices(ALL_ADAPTERS, weights=weights)[0]
        tasks.append(asyncio.create_task(
            send_request(session, nodes, node_strategy, adapter, results, semaphore)
        ))
        await asyncio.sleep(interval)
    await asyncio.gather(*tasks, return_exceptions=True)


async def run_zipf(session, nodes, node_strategy, rps, duration, results, semaphore):
    print(f"\n[zipf] {rps} rps for {duration}s across {len(nodes)} nodes — Zipf s=1.5")
    weights = zipf_distribution(ALL_ADAPTERS, s=1.5)
    print(f"  Top 5 weights: {[f'{ALL_ADAPTERS[i]}={weights[i]:.3f}' for i in range(5)]}")
    interval = 1.0 / rps
    end = time.time() + duration
    tasks = []
    while time.time() < end:
        adapter = random.choices(ALL_ADAPTERS, weights=weights)[0]
        tasks.append(asyncio.create_task(
            send_request(session, nodes, node_strategy, adapter, results, semaphore)
        ))
        await asyncio.sleep(interval)
    await asyncio.gather(*tasks, return_exceptions=True)


async def run_burst(session, nodes, node_strategy, rps, duration, results, semaphore):
    print(f"\n[burst] {rps} rps for {duration}s across {len(nodes)} nodes — 20-request bursts")
    interval = 1.0 / rps
    end = time.time() + duration
    tasks = []
    burst_count = 0
    current_adapter = random.choice(ALL_ADAPTERS)
    while time.time() < end:
        if burst_count >= 20:
            current_adapter = random.choice(ALL_ADAPTERS)
            burst_count = 0
            print(f"  [burst] switching to adapter: {current_adapter}")
        tasks.append(asyncio.create_task(
            send_request(session, nodes, node_strategy, current_adapter, results, semaphore)
        ))
        burst_count += 1
        await asyncio.sleep(interval)
    await asyncio.gather(*tasks, return_exceptions=True)

# ── Results analysis ──────────────────────────────────────────────────────────

def analyze(results: list, mode: str, nodes: list, routing_mode: str):
    if not results:
        print("No results.")
        return

    successful = [r for r in results if r["status"] == 200]
    failed     = [r for r in results if r["status"] != 200]
    latencies  = [r["latency_ms"] for r in successful]
    forwarded  = [r for r in successful if r.get("forwarded")]

    print(f"\n{'='*60}")
    print(f"Results: mode={mode} routing={routing_mode} nodes={len(nodes)}")
    print(f"{'='*60}")
    print(f"  Total requests:  {len(results)}")
    print(f"  Successful:      {len(successful)}")
    print(f"  Failed:          {len(failed)}")
    print(f"  Forwarded:       {len(forwarded)} ({100*len(forwarded)/max(len(successful),1):.1f}%)")

    if latencies:
        print(f"  Latency p50:     {np.percentile(latencies, 50):.0f}ms")
        print(f"  Latency p95:     {np.percentile(latencies, 95):.0f}ms")
        print(f"  Latency p99:     {np.percentile(latencies, 99):.0f}ms")
        print(f"  Latency mean:    {np.mean(latencies):.0f}ms")

    adapter_latencies = defaultdict(list)
    for r in successful:
        adapter_latencies[r["adapter"]].append(r["latency_ms"])

    print(f"\n  Per-adapter mean latency (top 10 by request count):")
    sorted_adapters = sorted(adapter_latencies.items(), key=lambda x: -len(x[1]))
    for adapter, lats in sorted_adapters[:10]:
        tier = "GPU" if adapter in HOT_ADAPTERS else ("CPU" if adapter in WARM_ADAPTERS else "DISK")
        print(f"    {adapter:<35} n={len(lats):4d}  p50={np.percentile(lats,50):6.0f}ms  [{tier}]")

    print(f"\n  Requests sent to each node:")
    sent_counts = defaultdict(int)
    for r in results:
        node_short = r["sent_to"].split("//")[1].split(":")[0].split(".")[0]
        sent_counts[node_short] += 1
    for node, count in sorted(sent_counts.items()):
        print(f"    {node}: {count} ({100*count/len(results):.1f}%)")

    print(f"\n  Requests served by each node:")
    served_counts = defaultdict(int)
    for r in successful:
        served_counts[r.get("served_by", "unknown")] += 1
    for node, count in sorted(served_counts.items()):
        print(f"    {node}: {count} ({100*count/len(successful):.1f}%)")

    return {
        "mode":         mode,
        "routing":      routing_mode,
        "num_nodes":    len(nodes),
        "total":        len(results),
        "successful":   len(successful),
        "failed":       len(failed),
        "forwarded":    len(forwarded),
        "forward_rate": len(forwarded) / max(len(successful), 1),
        "p50_ms":       np.percentile(latencies, 50) if latencies else None,
        "p95_ms":       np.percentile(latencies, 95) if latencies else None,
        "p99_ms":       np.percentile(latencies, 99) if latencies else None,
        "mean_ms":      np.mean(latencies) if latencies else None,
    }

# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",     default="zipf",
                        choices=["uniform", "zipf", "burst", "all"])
    parser.add_argument("--num-nodes", type=int, default=DEFAULT_NUM_NODES,
                        help=f"Number of nodes to use from the list (default: {DEFAULT_NUM_NODES}, max: {len(ALL_NODES)})")
    parser.add_argument("--nodes",    nargs="+", default=None,
                        help="Explicit list of node URLs (overrides --num-nodes)")
    parser.add_argument("--node-strategy", default="random",
                        choices=["random"],
                        help="How to pick which node gets each request")
    parser.add_argument("--rps",      type=float, default=2.0)
    parser.add_argument("--duration", type=int,   default=120)
    parser.add_argument("--routing",  default="cost",
                        help="Label for output (baseline, memory, cost)")
    parser.add_argument("--out",      default="results.jsonl")
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()

    # resolve node list
    if args.nodes:
        nodes = args.nodes
    else:
        if args.num_nodes < 1 or args.num_nodes > len(ALL_NODES):
            print(f"Error: --num-nodes must be between 1 and {len(ALL_NODES)}")
            return
        nodes = ALL_NODES[:args.num_nodes]

    print(f"==> Distributed workload generator")
    print(f"    nodes:    {len(nodes)}")
    for n in nodes:
        print(f"      {n}")
    print(f"    mode:     {args.mode}")
    print(f"    rps:      {args.rps}")
    print(f"    duration: {args.duration}s")
    print(f"    routing:  {args.routing}")
    print(f"    strategy: {args.node_strategy}")

    semaphore = asyncio.Semaphore(args.concurrency)
    results   = []

    async with aiohttp.ClientSession() as session:
        if args.mode == "all":
            for mode in ["uniform", "zipf", "burst"]:
                mode_results = []
                if mode == "uniform":
                    await run_uniform(session, nodes, args.node_strategy, args.rps, args.duration, mode_results, semaphore)
                elif mode == "zipf":
                    await run_zipf(session, nodes, args.node_strategy, args.rps, args.duration, mode_results, semaphore)
                elif mode == "burst":
                    await run_burst(session, nodes, args.node_strategy, args.rps, args.duration, mode_results, semaphore)
                results.extend(mode_results)
                analyze(mode_results, mode, nodes, args.routing)
                print(f"\n  Cooling down 10s between modes...")
                await asyncio.sleep(10)
        elif args.mode == "uniform":
            await run_uniform(session, nodes, args.node_strategy, args.rps, args.duration, results, semaphore)
        elif args.mode == "zipf":
            await run_zipf(session, nodes, args.node_strategy, args.rps, args.duration, results, semaphore)
        elif args.mode == "burst":
            await run_burst(session, nodes, args.node_strategy, args.rps, args.duration, results, semaphore)

    summary = analyze(results, args.mode, nodes, args.routing)

    with open(args.out, "a") as f:
        for r in results:
            r["workload_mode"] = args.mode
            r["routing_mode"]  = args.routing
            f.write(json.dumps(r) + "\n")
        if summary:
            f.write(json.dumps({"type": "summary", **summary}) + "\n")

    print(f"\nResults saved to {args.out}")

if __name__ == "__main__":
    asyncio.run(main())