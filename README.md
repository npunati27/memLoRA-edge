# memLoRA-edge

Tiered memory-aware load balancing for multi-LoRA LLM serving on the edge.

## Prerequisites (GPU path)

- Linux host with an **NVIDIA GPU** and a recent driver (setup pins **CUDA 12.1** wheels).
- Python 3 venv (the setup script creates `~/venv`).
- Same **LAN IPs** for every node so `peers.json` matches across the cluster.

## One-time node setup (GPU)

From the repo, run `scripts/setup.sh` with this node’s **0-based index** and the **full ordered list of node IPs** (same order on every machine):

```bash
bash scripts/setup.sh <NODE_IDX> <IP0> <IP1> ...
```

Example for the second of three nodes:

```bash
bash scripts/setup.sh 1 10.0.0.1 10.0.0.2 10.0.0.3
```

The script will:

- Install system packages (`python3-venv`, `iproute2`, …) and a venv at `~/venv`.
- Install **PyTorch (cu121)** and **vLLM** plus API dependencies.
- Download **Qwen2.5-0.5B-Instruct** into `~/model_cache/`.
- Create placeholder LoRA adapter dirs under `~/adapters/`.
- Write **`~/peers.json`** with `my_ip`, `node_idx`, and `peers` for gossip and routing.

Optional: traffic shaping with `tc` (see script `--help` and `MEMLORA_ENABLE_TC`).

## Run the API

Activate the venv (`source ~/venv/bin/activate`), go to the repo, and start the server:

```bash
# From repository root (recommended)
python -m scripts.deploy
```

If your `PYTHONPATH` is the `scripts/` directory only:

```bash
cd scripts && python3 -m deploy
```

The HTTP server listens on **`SERVE_PORT`** (default **5000**). Endpoints include `/v1/chat/completions`, `/health`, `/internal/cluster` (dashboard aggregation), and gossip internals under `/internal/`.

### Routing

```bash
export ROUTING_MODE=baseline   # default, or: memory
```

## CPU-only mock (no GPU / no vLLM)

Same process and **same routes** as the GPU server; only local inference is faked (sleep + fixed text). Use when you want gossip, routing, and the cluster view without loading a model.

```bash
export MEMLORA_MOCK=1
python -m scripts.deploy
```

Shorthand (sets `MEMLORA_MOCK` if unset):

```bash
python -m scripts.deploy.mock_main
```

Tuning (environment variables; defaults live in `scripts/deploy/mock_settings.py`):

| Variable | Role |
|----------|------|
| `MEMLORA_MOCK_DELAY_MS` | Simulated inference delay (ms) |
| `MEMLORA_MOCK_JITTER_MS` | Extra random delay 0..N ms |
| `MEMLORA_MOCK_RESPONSE` | Assistant reply text |
| `MEMLORA_MOCK_MODEL_ID` | Base model id for `/v1/models` and parsing (default `qwen-base`) |
| `MEMLORA_MOCK_LORA_NAMES` | Comma-separated adapter names if `~/adapters` is empty |
| `MEMLORA_MOCK_SKIP_ADAPTER_CHECK` | `1` to skip on-disk adapter paths (default on) |

You still need **`~/peers.json`** (same shape as the GPU setup) so peers can gossip and `/internal/cluster` can fan out.
