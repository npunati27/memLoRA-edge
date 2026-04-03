import json
import time
import requests

NODE_URLS = [
    "http://128.105.146.30:5000",
    "http://128.105.146.28:5000",
    "http://128.105.146.35:5000",
    "http://128.105.146.42:5000",
]

ENTRY_NODE = NODE_URLS[0]
ADAPTER = "qwen-base/equip_drone"


def post_json(url, payload, timeout=60):
    resp = requests.post(url, json=payload, timeout=timeout)
    try:
        body = resp.json()
    except Exception:
        body = resp.text
    return resp, body


def get_json(url, timeout=30):
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def reset_all_nodes():
    print("\n== RESET ALL NODES ==")
    for node in NODE_URLS:
        resp, body = post_json(f"{node}/debug/reset", {})
        print(node, resp.status_code, body)


def print_adapter_views(adapter_name):
    print(f"\n== ADAPTER VIEW FOR {adapter_name} ==")
    for node in NODE_URLS:
        data = get_json(f"{node}/debug/adapter/{adapter_name}")
        print(f"\nNODE {node}")
        print(json.dumps(data, indent=2))


def print_route_views(adapter_name, mode="memory"):
    print(f"\n== ROUTE VIEW FOR {adapter_name}, mode={mode} ==")
    for node in NODE_URLS:
        data = get_json(f"{node}/debug/route?adapter_name={adapter_name}&mode={mode}")
        print(f"\nCHOOSER {node}")
        print(json.dumps(data, indent=2))


def send_inference(entry_node, model_name):
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": "Hello who are you?"}
        ],
        "max_tokens": 16,
    }
    resp = requests.post(f"{entry_node}/v1/chat/completions", json=payload, timeout=180)
    print(f"\n=== INFERENCE model={model_name} via {entry_node} ===")
    print("status:", resp.status_code)
    try:
        print(json.dumps(resp.json(), indent=2)[:1000])
    except Exception:
        print(resp.text[:1000])
    resp.raise_for_status()
    return resp.json()


def assert_gossip_propagated(adapter_name, expected_gpu_node):
    print("\n== ASSERT GOSSIP PROPAGATED ==")
    for node in NODE_URLS:
        data = get_json(f"{node}/debug/adapter/{adapter_name}")
        gpu_nodes = data["cluster_view"]["gpu"]

        assert expected_gpu_node.replace("http://", "").replace(":5000", "") in gpu_nodes or \
               expected_gpu_node in gpu_nodes, \
               f"Node {node} does not think {expected_gpu_node} has {adapter_name} in GPU. Saw {gpu_nodes}"

    print("PASS: all nodes learned the adapter GPU location.")


def assert_route_prefers_gpu(adapter_name, expected_node, chooser_node=None):
    print("\n== ASSERT ROUTING PREFERS GPU ==")
    nodes = [chooser_node] if chooser_node else NODE_URLS
    for node in nodes:
        data = get_json(f"{node}/debug/route?adapter_name={adapter_name}&mode=memory")
        chosen = data["chosen_node"]
        assert chosen == expected_node, \
            f"Chooser {node} picked {chosen}, expected {expected_node}"
    print("PASS: routing prefers expected GPU node.")


def main():
    reset_all_nodes()
    time.sleep(1)

    print_adapter_views(ADAPTER)

    # send one request through entry node
    send_inference(ENTRY_NODE, ADAPTER)

    # Wait for gossip to propagate
    time.sleep(2)

    print_adapter_views(ADAPTER)
    print_route_views(ADAPTER, mode="memory")

    gpu_holders = []
    for node in NODE_URLS:
        data = get_json(f"{node}/debug/adapter/{ADAPTER}")
        if data["local_tier"] == "gpu":
            gpu_holders.append(data["node"])

    assert len(gpu_holders) == 1, f"Expected exactly one GPU holder, got {gpu_holders}"
    expected_gpu_node = gpu_holders[0]

    assert_gossip_propagated(ADAPTER, expected_gpu_node)

    assert_route_prefers_gpu(ADAPTER, expected_gpu_node)

    print("\nPASS: gossip and routing system test succeeded.")


if __name__ == "__main__":
    main()