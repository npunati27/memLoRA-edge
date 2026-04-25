#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 [options] NODE_IDX IP0 [IP1 ...]" >&2
    echo "  NODE_IDX — 0-based index of this machine in the IP list" >&2
    echo "  IPn      — LAN IPs of every node, same order on all hosts" >&2
    echo " --enable-tc — apply per-peer HTB+netem limits (off by default)" >&2
    echo "  --no-tc    — do not run tc (default)" >&2
    echo " --no-sudo — run apt and tc without sudo (overrides MEMLORA_USE_SUDO)" >&2
    echo "  --sudo    — use sudo for apt and tc (default)" >&2
    echo "Environment:" >&2
    echo "  MEMLORA_USE_SUDO=yes|no (default yes). 0/no/false/off also disable." >&2
    echo "  MEMLORA_ENABLE_TC=yes|no (default no). 1/yes/true/on enables tc." >&2
    echo "Example (3 nodes, this is node 1): $0 1 10.0.0.1 10.0.0.2 10.0.0.3" >&2
    echo "Example with shaping: $0 --enable-tc 1 10.0.0.1 10.0.0.2 10.0.0.3" >&2
    exit 1
}

SHOULD_SUDO=true
case "${MEMLORA_USE_SUDO:-yes}" in 0|no|false|off) SHOULD_SUDO=false ;;
    1|yes|true|on)  SHOULD_SUDO=true ;;
esac

ENABLE_TC=false
case "${MEMLORA_ENABLE_TC:-no}" in 1|yes|true|on) ENABLE_TC=true ;;
 0|no|false|off) ENABLE_TC=false ;;
esac

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-sudo) SHOULD_SUDO=false; shift ;;
        --sudo)    SHOULD_SUDO=true; shift ;;
        --enable-tc|--tc) ENABLE_TC=true; shift ;;
        --no-tc)   ENABLE_TC=false; shift ;;
        -h|--help) usage ;;
        *) break ;;
    esac
done

if [[ "$SHOULD_SUDO" == true ]]; then SUDO=(sudo)
else
    SUDO=()
fi

[[ $# -lt 2 ]] && usage

NODE_IDX=$1
shift
NODE_IPS=("$@")
NUM_NODES=${#NODE_IPS[@]}

if ! [[ "$NODE_IDX" =~ ^[0-9]+$ ]] || (( NODE_IDX < 0 || NODE_IDX >= NUM_NODES )); then
    echo "Error: NODE_IDX must be an integer in [0, $((NUM_NODES - 1))] (got '$NODE_IDX')" >&2
    exit 1
fi

MY_IP="${NODE_IPS[$NODE_IDX]}"

echo "==> Setting up node$NODE_IDX / $NUM_NODES nodes (LAN IP: $MY_IP)"
if [[ ${#SUDO[@]} -gt 0 ]]; then
    echo "==> Using sudo for package install"
else
    echo "==> Running without sudo (MEMLORA_USE_SUDO / --no-sudo)"
fi
if [[ "$ENABLE_TC" == true ]]; then
    echo "==> Traffic shaping (tc) will be configured (--enable-tc / MEMLORA_ENABLE_TC)"
else
    echo "==> Traffic shaping (tc) skipped (use --enable-tc or MEMLORA_ENABLE_TC=yes)"
fi

"${SUDO[@]}" apt-get update -qq
"${SUDO[@]}" apt-get install -y python3-pip python3-venv iproute2 -qq

if [[ ! -d ~/venv ]]; then
    python3 -m venv ~/venv
fi
source ~/venv/bin/activate

grep -qxF 'source ~/venv/bin/activate' ~/.bashrc \
    || echo 'source ~/venv/bin/activate' >> ~/.bashrc

pip install -q \
    "torch==2.5.1" \
    --index-url https://download.pytorch.org/whl/cu121 \
    --no-cache-dir

pip install -q \
    "vllm==0.8.5" \
    "transformers==4.51.3" \
    "huggingface_hub>=0.23.0" \
    "click==8.1.7" \
    "boto3" \
    "peft" \
    "aiohttp" \
    "sortedcontainers" \
    --no-cache-dir

mkdir -p ~/model_cache ~/adapters

python3 - <<'EOF'
from huggingface_hub import snapshot_download
import os
dest = os.path.expanduser("~/model_cache/Qwen2.5-0.5B-Instruct")
if not os.path.exists(dest):
    snapshot_download(
        repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        local_dir=dest,
        ignore_patterns=["*.msgpack","*.h5","flax*","tf_*"]
    )
    print("Model downloaded.")
else:
    print("Model exists.")
EOF

python3 - <<'EOF'
import os, torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

ALL = [
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

model_dir = os.path.expanduser("~/model_cache/Qwen2.5-0.5B-Instruct")
adapter_dir = os.path.expanduser("~/adapters")

missing = [n for n in ALL if not os.path.exists(os.path.join(adapter_dir, n))]
if not missing:
    print("All adapters already exist, skipping.")
else:
    print(f"Creating {len(missing)} missing adapters...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float16, device_map="cpu"
    )
    cfg = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj","v_proj"],
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
    )
    for name in missing:
        path = os.path.join(adapter_dir, name)
        get_peft_model(model, cfg).save_pretrained(path)
        print(f"  created: {name}")
EOF

if [[ "$ENABLE_TC" == true ]]; then
    LAN_IFACE="enp65s0np0"
    LAN_IP=$(ip -4 addr show "$LAN_IFACE" | grep -oP '(?<=inet\s)\d+(\.\d+){3}')

    "${SUDO[@]}" tc qdisc del dev "$LAN_IFACE" root 2>/dev/null || true

    "${SUDO[@]}" tc qdisc add dev "$LAN_IFACE" root handle 1: htb default 1
    "${SUDO[@]}" tc class add dev "$LAN_IFACE" parent 1: classid 1:1 htb rate 1gbit

    # Per-link shaping: deterministic from (src_idx, dst_idx) so any N works.
    _link_latency() {
        local i=$1 j=$2
        local d=$(( i > j ? i - j : j - i ))
        local ms=$(( 10 + d * 30 + (i + j) * 5 ))
        (( ms > 120 )) && ms=120
        echo "${ms}ms"
    }

    _link_bw() {
        local i=$1 j=$2
        local d=$(( i > j ? i - j : j - i ))
        local mb=$(( 100 - d * 25 ))
        (( mb < 10 )) && mb=10
        echo "${mb}mbit"
    }

    _link_loss() {
        local i=$1 j=$2
        local p=$(( (i * 3 + j * 5 + i * j) % 9 ))
        echo "${p}%"
    }

    CLASS_ID=10

    for (( dst = 0; dst < NUM_NODES; dst++ )); do
        (( dst == NODE_IDX )) && continue

        DST_IP="${NODE_IPS[$dst]}"
        CLASS="1:$CLASS_ID"
        LAT=$(_link_latency "$NODE_IDX" "$dst")
        BW=$(_link_bw "$NODE_IDX" "$dst")
        LOSS=$(_link_loss "$NODE_IDX" "$dst")

        "${SUDO[@]}" tc class add dev "$LAN_IFACE" parent 1:1 classid "$CLASS" \
            htb rate "$BW"

        "${SUDO[@]}" tc qdisc add dev "$LAN_IFACE" parent "$CLASS" handle "${CLASS_ID}0:" \
            netem delay "$LAT" loss "$LOSS"

        "${SUDO[@]}" tc filter add dev "$LAN_IFACE" protocol ip parent 1: prio 1 \
            u32 match ip dst "$DST_IP/32" flowid "$CLASS"

        CLASS_ID=$((CLASS_ID + 1))
    done
fi

# ray stop --force 2>/dev/null || true
# sleep 2

# # every node is its own head 
# ray start --head \
#     --node-ip-address="$LAN_IP" \
#     --port=6379 \
#     --dashboard-host=0.0.0.0 \
#     --num-gpus=1 \
#     --num-cpus=32

echo "==> node$NODE_IDX ready"


PEERS_JSON=$(printf '"%s",' "${NODE_IPS[@]}")
PEERS_JSON="[${PEERS_JSON%,}]"

cat > ~/peers.json << PEERS
{
    "my_ip":    "$MY_IP",
    "node_idx": $NODE_IDX,
    "peers":    $PEERS_JSON
}
PEERS

echo "==> node$NODE_IDX ready — peers.json written"
echo "    Run: cd ~/memLoRA-edge/scripts && python3 -m deploy"