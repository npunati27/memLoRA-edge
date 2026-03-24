#!/usr/bin/env bash
set -euo pipefail

NODE_IDX=$1
NODE0_IP=$2
NODE1_IP=$3
NODE2_IP=$4
NODE3_IP=$5

NODE_IPS=("$NODE0_IP" "$NODE1_IP" "$NODE2_IP" "$NODE3_IP")

case $NODE_IDX in
    0) MY_IP=$NODE0_IP ;;
    1) MY_IP=$NODE1_IP ;;
    2) MY_IP=$NODE2_IP ;;
    3) MY_IP=$NODE3_IP ;;
    *) echo "Invalid node index"; exit 1 ;;
esac

echo "==> Setting up node$NODE_IDX (LAN IP: $MY_IP)"

# ── 1. Install deps ───────────────────────────────────────────────────────────
sudo apt-get update -qq
sudo apt-get install -y python3-pip python3-venv iproute2 -qq

python3 -m venv ~/venv
source ~/venv/bin/activate
echo "source ~/venv/bin/activate" >> ~/.bashrc

pip install -q "ray[serve,llm]" vllm torch transformers peft huggingface_hub aiohttp

# ── 2. Download model ─────────────────────────────────────────────────────────
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

# ── 3. Create LoRA adapters ───────────────────────────────────────────────────
python3 - <<'EOF'
import os, torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

ALL = [
 "crop_corn_disease","crop_wheat_disease","crop_soy_disease",
 "crop_tomato_disease","crop_cotton_disease","crop_general_health",
 "pest_aphid","pest_rootworm","pest_spider_mite","pest_caterpillar","pest_general",
 "soil_nitrogen","soil_phosphorus","soil_moisture","soil_ph",
 "irrigation_zone_a","irrigation_zone_b","irrigation_zone_c","irrigation_zone_d",
 "weather_forecast","weather_frost_alert","weather_humidity",
 "equip_tractor","equip_drone","equip_sensor",
]

model_dir = os.path.expanduser("~/model_cache/Qwen2.5-0.5B-Instruct")
adapter_dir = os.path.expanduser("~/adapters")

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    device_map="cpu"
)

cfg = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
)

for name in ALL:
    path = os.path.join(adapter_dir, name)
    if not os.path.exists(path):
        get_peft_model(model, cfg).save_pretrained(path)
EOF

# ── 4. Subset adapters ────────────────────────────────────────────────────────
python3 - <<EOF
import os, shutil

ALL = [
 "crop_corn_disease","crop_wheat_disease","crop_soy_disease",
 "crop_tomato_disease","crop_cotton_disease","crop_general_health",
 "pest_aphid","pest_rootworm","pest_spider_mite","pest_caterpillar","pest_general",
 "soil_nitrogen","soil_phosphorus","soil_moisture","soil_ph",
 "irrigation_zone_a","irrigation_zone_b","irrigation_zone_c","irrigation_zone_d",
 "weather_forecast","weather_frost_alert","weather_humidity",
 "equip_tractor","equip_drone","equip_sensor",
]

SUB = {
0:["crop_corn_disease","crop_wheat_disease","crop_general_health","pest_aphid","pest_general","soil_nitrogen","soil_moisture","irrigation_zone_a","weather_forecast","equip_tractor"],
1:["crop_soy_disease","crop_cotton_disease","crop_general_health","pest_rootworm","pest_spider_mite","pest_general","soil_ph","irrigation_zone_b","weather_forecast","equip_drone"],
2:["crop_tomato_disease","crop_general_health","pest_caterpillar","soil_phosphorus","soil_moisture","irrigation_zone_c","irrigation_zone_d","weather_frost_alert","weather_humidity","equip_sensor"],
3:["crop_wheat_disease","crop_general_health","pest_general","soil_nitrogen","soil_ph","irrigation_zone_a","irrigation_zone_b","weather_forecast","weather_frost_alert","equip_tractor"]
}

keep = SUB[$NODE_IDX]
adapter_dir = os.path.expanduser("~/adapters")

for name in ALL:
    path = os.path.join(adapter_dir, name)
    if name not in keep and os.path.exists(path):
        shutil.rmtree(path)
EOF

# ── 5. TC NETWORK SHAPING (LAN ONLY) ───────────────────────────────────────────
LAN_IFACE="enp65s0np0"
LAN_IP=$(ip -4 addr show "$LAN_IFACE" | grep -oP '(?<=inet\s)\d+(\.\d+){3}')

sudo tc qdisc del dev "$LAN_IFACE" root 2>/dev/null || true

sudo tc qdisc add dev "$LAN_IFACE" root handle 1: htb default 1
sudo tc class add dev "$LAN_IFACE" parent 1: classid 1:1 htb rate 1gbit

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

CLASS_ID=10

for dst in 0 1 2 3; do
    [[ "$dst" == "$NODE_IDX" ]] && continue

    DST_IP="${NODE_IPS[$dst]}"
    KEY="${NODE_IDX}_${dst}"
    CLASS="1:$CLASS_ID"

    sudo tc class add dev "$LAN_IFACE" parent 1:1 classid "$CLASS" \
        htb rate "${BW[$KEY]}"

    sudo tc qdisc add dev "$LAN_IFACE" parent "$CLASS" handle "${CLASS_ID}0:" \
        netem delay "${LAT[$KEY]}" loss "${LOSS[$KEY]}"

    sudo tc filter add dev "$LAN_IFACE" protocol ip parent 1: prio 1 \
        u32 match ip dst "$DST_IP/32" flowid "$CLASS"

    CLASS_ID=$((CLASS_ID + 1))
done

# ── 6. START RAY (LAN ONLY) ───────────────────────────────────────────────────
ray stop --force 2>/dev/null || true
sleep 2

if [[ "$NODE_IDX" == "0" ]]; then
    ray start --head \
        --node-ip-address="$LAN_IP" \
        --port=6379 \
        --dashboard-host=0.0.0.0 \
        --num-gpus=1 \
        --num-cpus=32

    echo "Dashboard (via SSH tunnel): http://localhost:8265"
else
    ray start \
        --address="$NODE0_IP:6379" \
        --node-ip-address="$LAN_IP" \
        --num-gpus=1 \
        --num-cpus=32
fi

echo "==> node$NODE_IDX ready"
