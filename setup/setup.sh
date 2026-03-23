#!/usr/bin/env bash
# =============================================================================
# setup.sh - Run on EVERY node
# 25 LoRA adapters simulating a smart farm edge deployment
# Each node only has a subset — forces cold fetches and makes routing hard
#
# Usage:
#   node0: bash setup.sh 0 <node0_ip> <node1_ip> <node2_ip> <node3_ip>
#   node1: bash setup.sh 1 <node0_ip> <node1_ip> <node2_ip> <node3_ip>
#   node2: bash setup.sh 2 <node0_ip> <node1_ip> <node2_ip> <node3_ip>
#   node3: bash setup.sh 3 <node0_ip> <node1_ip> <node2_ip> <node3_ip>
# =============================================================================
set -euo pipefail

NODE_IDX=$1
NODE0_IP=$2
NODE1_IP=$3
NODE2_IP=$4
NODE3_IP=$5

case $NODE_IDX in
    0) MY_IP=$NODE0_IP ;;
    1) MY_IP=$NODE1_IP ;;
    2) MY_IP=$NODE2_IP ;;
    3) MY_IP=$NODE3_IP ;;
    *) echo "Invalid node index"; exit 1 ;;
esac

echo "==> Setting up node$NODE_IDX (IP: $MY_IP)"

# ── 1. Install deps ────────────────────────────────────────────────────────────
echo "==> Installing dependencies..."
sudo apt-get update -qq
sudo apt-get install -y python3-pip python3-venv iproute2 -qq

python3 -m venv ~/venv
source ~/venv/bin/activate
echo "source ~/venv/bin/activate" >> ~/.bashrc

pip install -q "ray[serve,llm]" vllm torch transformers peft huggingface_hub aiohttp

# ── 2. Download base model ─────────────────────────────────────────────────────
echo "==> Downloading base model..."
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
    print("Model already exists, skipping.")
EOF

# ── 3. Create 25 dummy LoRA adapters ──────────────────────────────────────────
# Grouped by domain — simulates a real smart farm adapter library:
#
# Crop monitoring (6):   different crops, different disease models
# Pest detection (5):    different pest types
# Soil analysis (4):     different soil properties
# Irrigation (4):        different field zones
# Weather/env (3):       weather, frost, humidity
# Equipment (3):         tractor, drone, sensor diagnostics
echo "==> Creating 25 LoRA adapters..."

python3 - <<'EOF'
import os, torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

ALL_ADAPTERS = [
    # Crop monitoring
    "crop_corn_disease",
    "crop_wheat_disease",
    "crop_soy_disease",
    "crop_tomato_disease",
    "crop_cotton_disease",
    "crop_general_health",
    # Pest detection
    "pest_aphid",
    "pest_rootworm",
    "pest_spider_mite",
    "pest_caterpillar",
    "pest_general",
    # Soil analysis
    "soil_nitrogen",
    "soil_phosphorus",
    "soil_moisture",
    "soil_ph",
    # Irrigation
    "irrigation_zone_a",
    "irrigation_zone_b",
    "irrigation_zone_c",
    "irrigation_zone_d",
    # Weather/environment
    "weather_forecast",
    "weather_frost_alert",
    "weather_humidity",
    # Equipment diagnostics
    "equip_tractor",
    "equip_drone",
    "equip_sensor",
]

model_dir = os.path.expanduser("~/model_cache/Qwen2.5-0.5B-Instruct")
adapter_dir = os.path.expanduser("~/adapters")

print(f"Loading base model for adapter generation...")
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    device_map="cpu"
)

cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
)

created = 0
for name in ALL_ADAPTERS:
    path = os.path.join(adapter_dir, name)
    if not os.path.exists(path):
        get_peft_model(model, cfg).save_pretrained(path)
        created += 1
        print(f"  [{created}/{len(ALL_ADAPTERS)}] Created {name}")
    else:
        print(f"  Skipping {name} (exists)")

print(f"Done. {created} adapters created, {len(ALL_ADAPTERS)-created} already existed.")
EOF

# ── 4. Keep only this node's adapter subset ────────────────────────────────────
# Distribution strategy:
#   - Popular adapters (crop_general_health, pest_general, weather_forecast)
#     exist on multiple nodes — always have a warm replica somewhere
#   - Specialist adapters exist on 1-2 nodes only — cold on others
#   - No node has more than 10/25 adapters — forces constant cross-node fetches
#   - Intentionally uneven — some nodes are "richer" than others
#
# This distribution means:
#   ~60% of requests hit a cold adapter on the assigned node
#   baseline router doesn't know this → high TTFT
#   tier-aware router avoids cold nodes → lower TTFT
echo "==> Setting per-node adapter subset for node$NODE_IDX..."

python3 - <<EOF
import shutil, os

ALL_ADAPTERS = [
    "crop_corn_disease","crop_wheat_disease","crop_soy_disease",
    "crop_tomato_disease","crop_cotton_disease","crop_general_health",
    "pest_aphid","pest_rootworm","pest_spider_mite","pest_caterpillar","pest_general",
    "soil_nitrogen","soil_phosphorus","soil_moisture","soil_ph",
    "irrigation_zone_a","irrigation_zone_b","irrigation_zone_c","irrigation_zone_d",
    "weather_forecast","weather_frost_alert","weather_humidity",
    "equip_tractor","equip_drone","equip_sensor",
]

# Each node keeps ~8-10 adapters out of 25
# Carefully designed so popular adapters are on multiple nodes
# but specialist ones are rare
NODE_SUBSETS = {
    0: [
        # Ground robot — crop monitoring + general
        "crop_corn_disease",
        "crop_wheat_disease",
        "crop_general_health",    # popular, on multiple nodes
        "pest_aphid",
        "pest_general",           # popular, on multiple nodes
        "soil_nitrogen",
        "soil_moisture",
        "irrigation_zone_a",
        "weather_forecast",       # popular, on multiple nodes
        "equip_tractor",
    ],
    1: [
        # Drone — aerial monitoring + pest detection
        "crop_soy_disease",
        "crop_cotton_disease",
        "crop_general_health",    # popular
        "pest_rootworm",
        "pest_spider_mite",
        "pest_general",           # popular
        "soil_ph",
        "irrigation_zone_b",
        "weather_forecast",       # popular
        "equip_drone",
    ],
    2: [
        # Sensor hub — soil + weather + irrigation
        "crop_tomato_disease",
        "crop_general_health",    # popular
        "pest_caterpillar",
        "soil_phosphorus",
        "soil_moisture",
        "irrigation_zone_c",
        "irrigation_zone_d",
        "weather_frost_alert",
        "weather_humidity",
        "equip_sensor",
    ],
    3: [
        # Home hub — general purpose, some overlap
        "crop_wheat_disease",
        "crop_general_health",    # popular
        "pest_general",           # popular
        "soil_nitrogen",
        "soil_ph",
        "irrigation_zone_a",
        "irrigation_zone_b",
        "weather_forecast",       # popular
        "weather_frost_alert",
        "equip_tractor",
    ],
}

keep = NODE_SUBSETS[$NODE_IDX]
adapter_dir = os.path.expanduser("~/adapters")

print(f"node$NODE_IDX keeping {len(keep)}/25 adapters:")
for name in ALL_ADAPTERS:
    path = os.path.join(adapter_dir, name)
    if name in keep:
        print(f"  KEEP   {name}")
    else:
        if os.path.exists(path):
            shutil.rmtree(path)
        print(f"  REMOVE {name}")

print(f"Done. node$NODE_IDX has {len(keep)} adapters on local disk.")
EOF

# ── 5. Print adapter summary ──────────────────────────────────────────────────
echo ""
echo "==> node$NODE_IDX adapter summary:"
echo "    Local adapters ($(ls ~/adapters | wc -l)/25):"
ls ~/adapters | sed 's/^/      /'

# ── 6. Apply tc netem network emulation ───────────────────────────────────────
echo "==> Applying network emulation..."

# Find the experimental LAN interface
# On CloudLab d7525 this is typically enp65s0f0
IFACE=$(ip route | grep "10\." | awk '{print $3}' | head -1)
[[ -z "$IFACE" ]] && IFACE="enp65s0f0"
echo "    Using interface: $IFACE"

# Clear existing rules
sudo tc qdisc del dev "$IFACE" root 2>/dev/null || true

# Apply netem per destination IP
# Simulates sparse ad-hoc wireless mesh
NODE_IPS=("$NODE0_IP" "$NODE1_IP" "$NODE2_IP" "$NODE3_IP")

# Link profiles: latency, bandwidth, loss
declare -A LAT=( ["0_1"]="10ms" ["0_2"]="50ms" ["0_3"]="80ms"
                 ["1_0"]="10ms" ["1_2"]="40ms" ["1_3"]="30ms"
                 ["2_0"]="50ms" ["2_1"]="40ms" ["2_3"]="60ms"
                 ["3_0"]="80ms" ["3_1"]="30ms" ["3_2"]="60ms" )
declare -A BW=(  ["0_1"]="100mbit" ["0_2"]="20mbit" ["0_3"]="10mbit"
                 ["1_0"]="100mbit" ["1_2"]="25mbit" ["1_3"]="50mbit"
                 ["2_0"]="20mbit"  ["2_1"]="25mbit" ["2_3"]="15mbit"
                 ["3_0"]="10mbit"  ["3_1"]="50mbit" ["3_2"]="15mbit" )
declare -A LOSS=( ["0_1"]="0%" ["0_2"]="3%" ["0_3"]="8%"
                  ["1_0"]="0%" ["1_2"]="2%" ["1_3"]="1%"
                  ["2_0"]="3%" ["2_1"]="2%" ["2_3"]="5%"
                  ["3_0"]="8%" ["3_1"]="1%" ["3_2"]="5%" )

sudo tc qdisc add dev "$IFACE" root handle 1: prio bands 5

for dst in 0 1 2 3; do
    [[ "$dst" == "$NODE_IDX" ]] && continue
    DST_IP="${NODE_IPS[$dst]}"
    BAND=$((dst + 2))
    KEY="${NODE_IDX}_${dst}"
    sudo tc qdisc add dev "$IFACE" parent "1:$BAND" handle "$((BAND*10)):" \
        netem delay "${LAT[$KEY]}" loss "${LOSS[$KEY]}" rate "${BW[$KEY]}"
    sudo tc filter add dev "$IFACE" protocol ip parent 1:0 prio "$BAND" \
        u32 match ip dst "$DST_IP/32" flowid "1:$BAND"
    echo "    node$NODE_IDX -> node$dst: ${LAT[$KEY]} ${BW[$KEY]} ${LOSS[$KEY]}"
done

# ── 7. Start Ray ───────────────────────────────────────────────────────────────
echo "==> Starting Ray..."
ray stop --force 2>/dev/null || true
sleep 2

if [[ "$NODE_IDX" == "0" ]]; then
    ray start --head \
        --node-ip-address="$MY_IP" \
        --port=6379 \
        --dashboard-host=0.0.0.0 \
        --num-gpus=1 \
        --num-cpus=32
    echo ""
    echo "==> Ray head started on node0."
    echo "    Dashboard: http://$MY_IP:8265"
    echo "    Now run setup.sh on node1, node2, node3."
else
    ray start \
        --address="$NODE0_IP:6379" \
        --node-ip-address="$MY_IP" \
        --num-gpus=1 \
        --num-cpus=32
    echo "==> node$NODE_IDX joined cluster at $NODE0_IP."
fi

echo ""
echo "==> node$NODE_IDX setup complete."