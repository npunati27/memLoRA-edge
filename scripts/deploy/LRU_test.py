import json
import time
import requests

BASE_URL = "http://128.105.146.42:5000"
CHAT_ENDPOINT = f"{BASE_URL}/internal/chat/completions"
STATE_ENDPOINT = f"{BASE_URL}/internal/debug/state"

ADAPTERS = [
    "qwen-base/equip_drone",
    "qwen-base/crop_corn_disease",
    "qwen-base/crop_wheat_disease",
    "qwen-base/crop_soy_disease",
]

def send_request(model_name: str):
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": f"Hello who are you?"}
        ],
        "max_tokens": 32,
    }

    resp = requests.post(CHAT_ENDPOINT, json=payload, timeout=180)
    print(f"\n=== REQUEST model={model_name} ===")
    print(f"status={resp.status_code}")
    try:
        print(json.dumps(resp.json(), indent=2)[:1000])
    except Exception:
        print(resp.text[:1000])

    resp.raise_for_status()
    return resp.json()

def get_state():
    resp = requests.get(STATE_ENDPOINT, timeout=30)
    resp.raise_for_status()
    return resp.json()

def print_state(label: str, state: dict):
    print(f"\n--- {label} ---")
    print(f"node: {state['node']}")
    print(f"GPU LRU: {state['local_adapters']['gpu']}")
    print(f"CPU LRU: {state['local_adapters']['cpu']}")
    print("Peer adapter state:")
    print(json.dumps(state["adapter_state"], indent=2))

def assert_state(expected_gpu=None, expected_cpu=None):
    state = get_state()
    gpu = state['local_adapters']['gpu']
    cpu = state['local_adapters']['cpu']

    if expected_gpu is not None:
        assert gpu == expected_gpu, f"GPU mismatch: expected {expected_gpu}, got {gpu}"
    if expected_cpu is not None:
        assert cpu == expected_cpu, f"CPU mismatch: expected {expected_cpu}, got {cpu}"

    return state

def main():
    print("Initial state:")
    state = get_state()
    print_state("INITIAL", state)

    # Request 1: disk -> gpu
    send_request(ADAPTERS[0])
    time.sleep(1)
    state = assert_state(
        expected_gpu=["equip_drone"],
        expected_cpu=[],
    )
    print_state("AFTER REQUEST 1", state)

    # Request 2: disk -> gpu (2)
    send_request(ADAPTERS[1])
    time.sleep(1)
    state = assert_state(
        expected_gpu=[
            "equip_drone",
            "crop_corn_disease",
        ],
        expected_cpu=[],
    )
    print_state("AFTER REQUEST 2", state)

    # Request 3: disk -> gpu (3)
    send_request(ADAPTERS[2])
    time.sleep(1)
    state = assert_state(
        expected_gpu=[
            "equip_drone",
            "crop_corn_disease",
            "crop_wheat_disease",
        ],
        expected_cpu=[],
    )
    print_state("AFTER REQUEST 3", state)

    # Request 4 -> disk -> gpu (4), oldest gpu should move to cpu
    send_request(ADAPTERS[3])
    time.sleep(1)
    state = assert_state(
        expected_gpu=[
            "crop_corn_disease",
            "crop_wheat_disease",
            "crop_soy_disease",
        ],
        expected_cpu=[
            "equip_drone",
        ],
    )
    print_state("AFTER REQUEST 4", state)

    # Request 5: touch GPU-resident adapter -> should become MRU
    send_request(ADAPTERS[1])  # crop_corn_disease
    time.sleep(1)
    state = assert_state(
        expected_gpu=[
            "crop_wheat_disease",
            "crop_soy_disease",
            "crop_corn_disease",
        ],
        expected_cpu=[
            "equip_drone",
        ],
    )
    print_state("AFTER REQUEST 5 (touch corn)", state)

    # Request 6: add new adapter -> oldest GPU (wheat) should move to CPU
    send_request("qwen-base/crop_tomato_disease")
    time.sleep(1)
    state = assert_state(
        expected_gpu=[
            "crop_soy_disease",
            "crop_corn_disease",
            "crop_tomato_disease",
        ],
        expected_cpu=[
            "equip_drone",
            "crop_wheat_disease",
        ],
    )
    print_state("AFTER REQUEST 6 (crop_tomato_disease)", state)

    print("\nPASS: LRU system flow behaved as expected.")

if __name__ == "__main__":
    main()