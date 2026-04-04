import time
import json
import requests
from concurrent.futures import ThreadPoolExecutor

NODE_URLS = [
    "http://128.105.146.30:5000",
    "http://128.105.146.28:5000",
    "http://128.105.146.35:5000",
    "http://128.105.146.42:5000",
]

ENTRY_NODE = NODE_URLS[0]
ADAPTER = "qwen-base/equip_drone"
TIMEOUT = 60


def get_json(url, timeout=10):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def post_json(url, payload, timeout=TIMEOUT):
    r = requests.post(url, json=payload, timeout=timeout)
    try:
        body = r.json()
    except Exception:
        body = r.text
    return r.status_code, body


def get_state(node_url):
    return get_json(f"{node_url}/internal/debug/state")


def print_cluster_states(label):
    print(f"\n===== {label} =====")
    for node in NODE_URLS:
        try:
            state = get_state(node)
            print(f"\nnode: {state.get('node')}")
            print(f"  local_queue: {state.get('local_queue')}")
            print(f"  peer_queues: {json.dumps(state.get('peer_queues', {}), indent=2)}")
            print(f"  local GPU: {state.get('local_adapters', {}).get('gpu', [])}")
            print(f"  local CPU: {state.get('local_adapters', {}).get('cpu', [])}") 
            print(f"  gossip_running: {state.get('gossip_running')}")
            print(f"  gossip_task_active: {state.get('gossip_task_active')}")
        except Exception as e:
            print(f"\nnode: {node} ERROR: {e}")


def assert_gossip_running():
    print("\nChecking gossip health...")
    for node in NODE_URLS:
        state = get_state(node)
        assert state.get("gossip_running") is True, f"{node} gossip_running is False"
        assert state.get("gossip_task_active") is True, f"{node} gossip_task_active is False"
    print("All nodes report gossip loop active.")


def test_queue_gossip():
    print("\n=== TEST 1: Queue gossip ===")

    payload = {
        "model": ADAPTER,
        "messages": [{"role": "user", "content": "Explain what a drone is in one sentence."}],
        "max_tokens": 256,
        "temperature": 0.0,
    }

    def send_req():
        return post_json(f"{ENTRY_NODE}/v1/chat/completions", payload, timeout=120)

    # fire a few concurrent requests so at least one node has nonzero queue
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(send_req) for _ in range(4)]
        time.sleep(1.0)  # give requests time to start + gossip loop time to broadcast

        print_cluster_states("DURING QUEUE TEST")

        observed_nonzero = False
        for node in NODE_URLS:
            state = get_state(node)
            peer_queues = state.get("peer_queues", {})
            if any(v > 0 for v in peer_queues.values()):
                observed_nonzero = True
                break

        assert observed_nonzero, "No node observed a nonzero peer queue via gossip."

        for f in futures:
            status, body = f.result()
            assert status == 200, f"Request failed: {status}, {body}"

    time.sleep(0.5)
    print("Queue gossip test passed.")


def test_adapter_gossip():
    print("\n=== TEST 2: Adapter-state gossip ===")

    payload = {
        "model": ADAPTER,
        "messages": [{"role": "user", "content": "Hello, who are you?"}],
        "max_tokens": 32,
        "temperature": 0.0,
    }

    status, body = post_json(f"{ENTRY_NODE}/v1/chat/completions", payload, timeout=120)
    assert status == 200, f"Inference failed: {status}, {body}"

    served_by = body.get("served_by")
    assert served_by, f"No served_by field in response: {body}"

    print(f"Request served by: {served_by}")

    time.sleep(1.0)  # allow adapter_state gossip to propagate

    adapter_name = ADAPTER.split("/", 1)[1]

    for node in NODE_URLS:
        state = get_state(node)
        adapter_state = state.get("adapter_state", {})
        assert adapter_name in adapter_state, f"{adapter_name} missing from adapter_state on {node}"

        tiers = adapter_state[adapter_name]
        gpu_nodes = set(tiers.get("gpu", []))
        cpu_nodes = set(tiers.get("cpu", []))
        disk_nodes = set(tiers.get("disk", []))

        # the serving node should now be known as gpu or cpu depending on state after tracking
        assert served_by in (gpu_nodes | cpu_nodes | disk_nodes), (
            f"{served_by} not present in any tier for {adapter_name} on {node}"
        )

    print("Adapter gossip test passed.")


def test_lru_plus_gossip():
    print("\n=== TEST 3: LRU changes reflected in gossip ===")

    adapters = [
        "qwen-base/equip_drone",
        "qwen-base/irrigation_zone_a",
        "qwen-base/pest_aphid",
        "qwen-base/soil_nitrogen",
    ]

    for model_name in adapters:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": f"Say hi from {model_name}"}],
            "max_tokens": 16,
            "temperature": 0.0,
        }
        status, body = post_json(f"{ENTRY_NODE}/v1/chat/completions", payload, timeout=120)
        assert status == 200, f"Failed for {model_name}: {status}, {body}"
        time.sleep(0.3)

    time.sleep(1.5)

    print_cluster_states("AFTER LRU LOADS")

    # We just verify adapter_state exists cluster-wide and tiers are not duplicated per node.
    for node in NODE_URLS:
        state = get_state(node)
        adapter_state = state.get("adapter_state", {})
        for adapter, tiers in adapter_state.items():
            locations = {}
            for tier_name, nodes in tiers.items():
                for n in nodes:
                    locations.setdefault(n, []).append(tier_name)
            for n, tier_list in locations.items():
                assert len(tier_list) == 1, (
                    f"Node {n} appears in multiple tiers {tier_list} for adapter {adapter} on observer {node}"
                )

    print("LRU + gossip consistency test passed.")


def main():
    print_cluster_states("INITIAL")
    assert_gossip_running()
    test_queue_gossip()
    test_adapter_gossip()
    # only run this if those adapter names actually exist on disk
    test_lru_plus_gossip()
    print_cluster_states("FINAL")
    print("\nAll gossip tests passed.")


if __name__ == "__main__":
    main()