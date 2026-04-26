#!/usr/bin/env python3
"""
workload.py — tiered memory pressure workload generator for memLoRA

Usage:
    python3 workload.py --mode zipf --node http://10.10.1.1:5000 --rps 4 --duration 120
    python3 workload.py --mode uniform --node http://10.10.1.1:5000 --rps 4 --duration 120
    python3 workload.py --mode burst --node http://10.10.1.1:5000 --rps 4 --duration 120
"""

import argparse, asyncio, aiohttp, time, json, random, uuid, os
import numpy as np
from collections import defaultdict

# all 25 adapters
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

# designed to exceed memory: 3 GPU + 6 CPU = 9 warm slots across 25 adapters
# so at steady state ~16 adapters are always cold on disk
HOT_ADAPTERS  = ALL_ADAPTERS[:3]   # fits in GPU
WARM_ADAPTERS = ALL_ADAPTERS[3:9]  # fits in CPU
COLD_ADAPTERS = ALL_ADAPTERS[9:]   # always on disk

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

# ── workload distributions ────────────────────────────────────────────────────

def zipf_distribution(adapters: list, s: float = 1.5) -> list[float]:
    """Zipf distribution — a few adapters get most traffic."""
    ranks = np.arange(1, len(adapters) + 1)
    weights = 1.0 / (ranks ** s)
    return (weights / weights.sum()).tolist()

def uniform_distribution(adapters: list) -> list[float]:
    n = len(adapters)
    return [1.0 / n] * n

def burst_sequence(burst_size: int = 20) -> list[str]:
    """Returns infinite sequence of adapters in bursts."""
    while True:
        adapter = random.choice(ALL_ADAPTERS)
        for _ in range(burst_size):
            yield adapter

# ── request sender ────────────────────────────────────────────────────────────
async def check_nodes(session, node_urls: list[str]):
    print("\n=== Preflight: node health check ===")
    alive = []

    for url in node_urls:
        try:
            async with session.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                text = await resp.text()
                ok = resp.status == 200
                print(f"{url:<25} health={resp.status} ok={ok} body={text[:120]}")
                if ok:
                    alive.append(url)
        except Exception as e:
            print(f"{url:<25} health=FAIL error={e}")

    print(f"Alive nodes: {len(alive)}/{len(node_urls)}")
    print("=" * 40)

    if len(alive) != len(node_urls):
        print("WARNING: Not all nodes are reachable before workload starts.")

    return alive

async def send_request(
    session: aiohttp.ClientSession,
    node_url: str,
    adapter_name: str,
    results: list,
    semaphore: asyncio.Semaphore,
):
    async with semaphore:
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
        error = None

        try:
            async with session.post(
                f"{node_url}/v1/chat/completions",
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
            "ts":         time.time(),
            "adapter":    adapter_name,
            "latency_ms": latency_ms,
            "status":     status,
            "served_by":  served_by,
            "error":      error,
        })

        if error:
            print(f"  ERR {adapter_name} {latency_ms:.0f}ms {error}")
        else:
            print(f"  OK  {adapter_name:<30} {latency_ms:6.0f}ms  served_by={served_by}")

# ── workload runners ──────────────────────────────────────────────────────────

async def run_uniform(session, node_url, rps, duration, results, semaphore):
    print(f"\n[uniform] {rps} rps for {duration}s — equal probability across all 25 adapters")
    weights = uniform_distribution(ALL_ADAPTERS)
    interval = 1.0 / rps
    end = time.time() + duration
    tasks = []

    while time.time() < end:
        adapter = random.choices(ALL_ADAPTERS, weights=weights)[0]
        task = asyncio.create_task(
            send_request(session, node_url, adapter, results, semaphore)
        )
        tasks.append(task)
        await asyncio.sleep(interval)

    await asyncio.gather(*tasks, return_exceptions=True)

async def run_zipf(session, node_url, rps, duration, results, semaphore):
    print(f"\n[zipf] {rps} rps for {duration}s — Zipf s=1.5, top 3 adapters get ~60% traffic")
    weights = zipf_distribution(ALL_ADAPTERS, s=1.5)
    print(f"  Top 5 weights: {[f'{ALL_ADAPTERS[i]}={weights[i]:.3f}' for i in range(5)]}")
    interval = 1.0 / rps
    end = time.time() + duration
    tasks = []

    while time.time() < end:
        adapter = random.choices(ALL_ADAPTERS, weights=weights)[0]
        task = asyncio.create_task(
            send_request(session, node_url, adapter, results, semaphore)
        )
        tasks.append(task)
        await asyncio.sleep(interval)

    await asyncio.gather(*tasks, return_exceptions=True)

async def run_burst(session, node_url, rps, duration, results, semaphore):
    print(f"\n[burst] {rps} rps for {duration}s — 20-request bursts per adapter, then switch")
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

        task = asyncio.create_task(
            send_request(session, node_url, current_adapter, results, semaphore)
        )
        tasks.append(task)
        burst_count += 1
        await asyncio.sleep(interval)

    await asyncio.gather(*tasks, return_exceptions=True)

# ── results analysis ──────────────────────────────────────────────────────────

def analyze(results: list, mode: str, node_url: str, routing_mode: str):
    if not results:
        print("No results.")
        return

    successful = [r for r in results if r["status"] == 200]
    failed     = [r for r in results if r["status"] != 200]
    latencies  = [r["latency_ms"] for r in successful]

    print(f"\n{'='*60}")
    print(f"Results: mode={mode} routing={routing_mode} node={node_url}")
    print(f"{'='*60}")
    print(f"  Total requests:  {len(results)}")
    print(f"  Successful:      {len(successful)}")
    print(f"  Failed:          {len(failed)}")

    if latencies:
        print(f"  Latency p50:     {np.percentile(latencies, 50):.0f}ms")
        print(f"  Latency p95:     {np.percentile(latencies, 95):.0f}ms")
        print(f"  Latency p99:     {np.percentile(latencies, 99):.0f}ms")
        print(f"  Latency mean:    {np.mean(latencies):.0f}ms")

    # per-adapter breakdown
    adapter_latencies = defaultdict(list)
    for r in successful:
        adapter_latencies[r["adapter"]].append(r["latency_ms"])

    print(f"\n  Per-adapter mean latency (top 10 by request count):")
    sorted_adapters = sorted(adapter_latencies.items(), key=lambda x: -len(x[1]))
    for adapter, lats in sorted_adapters[:10]:
        tier = "GPU" if adapter in HOT_ADAPTERS else ("CPU" if adapter in WARM_ADAPTERS else "DISK")
        print(f"    {adapter:<35} n={len(lats):4d}  p50={np.percentile(lats,50):6.0f}ms  [{tier}]")

    # served_by distribution
    served_counts = defaultdict(int)
    for r in successful:
        served_counts[r.get("served_by", "unknown")] += 1
    print(f"\n  Served by node:")
    for node, count in sorted(served_counts.items()):
        print(f"    {node}: {count} ({100*count/len(successful):.1f}%)")

    return {
        "mode":         mode,
        "routing":      routing_mode,
        "total":        len(results),
        "successful":   len(successful),
        "failed":       len(failed),
        "p50_ms":       np.percentile(latencies, 50) if latencies else None,
        "p95_ms":       np.percentile(latencies, 95) if latencies else None,
        "p99_ms":       np.percentile(latencies, 99) if latencies else None,
        "mean_ms":      np.mean(latencies) if latencies else None,
    }

# ── main ──────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",     default="zipf",
                        choices=["uniform", "zipf", "burst", "all"])
    parser.add_argument("--node",     default="http://10.10.1.1:5000")
    parser.add_argument("--rps",      type=float, default=2.0)
    parser.add_argument("--duration", type=int,   default=120)
    parser.add_argument("--routing",  default="baseline",
                        help="Label for output (baseline or memory)")
    parser.add_argument("--out",      default="results.jsonl")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument(
        "--nodes",
        default="http://10.10.1.1:5000,http://10.10.1.2:5000,http://10.10.1.3:5000,http://10.10.1.4:5000",
        help="Comma-separated list of all node URLs for preflight health check",
    )
    args = parser.parse_args()

    semaphore = asyncio.Semaphore(args.concurrency)
    results   = []

    async with aiohttp.ClientSession() as session:
        node_urls = [u.strip() for u in args.nodes.split(",") if u.strip()]
        await check_nodes(session, node_urls)
        if args.mode == "all":
            for mode in ["uniform", "zipf", "burst"]:
                mode_results = []
                if mode == "uniform":
                    await run_uniform(session, args.node, args.rps, args.duration, mode_results, semaphore)
                elif mode == "zipf":
                    await run_zipf(session, args.node, args.rps, args.duration, mode_results, semaphore)
                elif mode == "burst":
                    await run_burst(session, args.node, args.rps, args.duration, mode_results, semaphore)
                results.extend(mode_results)
                analyze(mode_results, mode, args.node, args.routing)
                print(f"\n  Cooling down 10s between modes...")
                await asyncio.sleep(10)
        elif args.mode == "uniform":
            await run_uniform(session, args.node, args.rps, args.duration, results, semaphore)
        elif args.mode == "zipf":
            await run_zipf(session, args.node, args.rps, args.duration, results, semaphore)
        elif args.mode == "burst":
            await run_burst(session, args.node, args.rps, args.duration, results, semaphore)

    summary = analyze(results, args.mode, args.node, args.routing)

    # save raw results
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
