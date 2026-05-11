#!/usr/bin/env bash
# Coordinator script: sets up and starts memLoRA-edge on all nodes.
# Usage: bash run_cluster.sh [--nodes N] [--routing-mode cost|memory|baseline] [--mock 0|1] [--s3 0|1]

set -euo pipefail

# ── All 20 nodes ──────────────────────────────────────────────────────────────
ALL_HOSTS=(
    sp26-cs525-0701.cs.illinois.edu
    sp26-cs525-0702.cs.illinois.edu
    sp26-cs525-0703.cs.illinois.edu
    sp26-cs525-0704.cs.illinois.edu
    sp26-cs525-0705.cs.illinois.edu
    sp26-cs525-0706.cs.illinois.edu
    sp26-cs525-0707.cs.illinois.edu
    sp26-cs525-0708.cs.illinois.edu
    sp26-cs525-0710.cs.illinois.edu
    sp26-cs525-0711.cs.illinois.edu
    sp26-cs525-0712.cs.illinois.edu
    sp26-cs525-0713.cs.illinois.edu
    sp26-cs525-0714.cs.illinois.edu
    sp26-cs525-0715.cs.illinois.edu
    sp26-cs525-0716.cs.illinois.edu
    sp26-cs525-0717.cs.illinois.edu
    sp26-cs525-0718.cs.illinois.edu
    sp26-cs525-0719.cs.illinois.edu
    sp26-cs525-0720.cs.illinois.edu
)

SSH_USER="${SSH_USER:-npunati2}"
SSH_PORT="${SSH_PORT:-22}"
REPO_URL="https://github.com/npunati27/memLoRA-edge.git"
REPO_NAME="memLoRA-edge"
SERVE_PORT="${SERVE_PORT:-5000}"

# ── Defaults ──────────────────────────────────────────────────────────────────
ROUTING_MODE="cost"
MEMLORA_MOCK="1"
USE_S3_ADAPTERS="1"
NUM_NODES=8   # default to 8, override with --nodes
BRANCH="main"
COST_W_QUEUE="0.4"
COST_W_MEMORY="0.4"
COST_W_NETWORK="0.2"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nodes)          NUM_NODES="$2";      shift 2 ;;
        --branch)         BRANCH="$2";         shift 2 ;;
        --routing-mode)   ROUTING_MODE="$2";   shift 2 ;;
        --mock)           MEMLORA_MOCK="$2";   shift 2 ;;
        --s3)             USE_S3_ADAPTERS="$2"; shift 2 ;;
        --user)           SSH_USER="$2";       shift 2 ;;
        --port)           SSH_PORT="$2";       shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--nodes N] [--branch BRANCH] [--routing-mode cost|memory|baseline] [--mock 0|1] [--s3 0|1] [--cost-w-queue W] [--cost-w-memory W] [--cost-w-network W] [--user USER] [--port PORT]"
            echo "  --nodes N           Use first N nodes from the list (default: 8, max: 20)"
            echo "  --branch NAME       Git branch to clone/pull on each node (default: main)"
            exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# validate NUM_NODES
if (( NUM_NODES < 1 || NUM_NODES > ${#ALL_HOSTS[@]} )); then
    echo "Error: --nodes must be between 1 and ${#ALL_HOSTS[@]} (got $NUM_NODES)" >&2
    exit 1
fi

# slice to first NUM_NODES
HOSTS=("${ALL_HOSTS[@]:0:$NUM_NODES}")
HOSTS_STR="${HOSTS[*]}"

echo "==> memLoRA-edge cluster setup"
echo "    nodes:        $NUM_NODES / ${#ALL_HOSTS[@]}"
echo "    user:         $SSH_USER"
echo "    routing_mode: $ROUTING_MODE"
echo "    mock:         $MEMLORA_MOCK"
echo "    s3:           $USE_S3_ADAPTERS"
echo "    serve_port:   $SERVE_PORT"
echo "    active nodes:"
for idx in "${!HOSTS[@]}"; do
    echo "      node$idx: ${HOSTS[$idx]}"
done
echo ""

# ── Step 1: SSH key setup ─────────────────────────────────────────────────────
setup_ssh_keys() {
    echo "==> Step 1: Setting up SSH keys (you will be prompted for passwords)"
    echo ""

    if [[ ! -f ~/.ssh/id_ed25519 ]]; then
        echo "    Generating SSH key..."
        ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" -C "memlora-cluster"
    else
        echo "    SSH key already exists at ~/.ssh/id_ed25519"
    fi

    for host in "${HOSTS[@]}"; do
        echo "    Copying key to $host (enter password when prompted)..."
        ssh-copy-id -i ~/.ssh/id_ed25519.pub \
            -p "$SSH_PORT" \
            -o StrictHostKeyChecking=no \
            "$SSH_USER@$host" \
            && echo "    ✓ $host" \
            || echo "    ✗ $host — failed, will need password during later steps"
    done
    echo ""
}

# ── Step 2: Setup each node ───────────────────────────────────────────────────
setup_node() {
    local idx=$1
    local host="${HOSTS[$idx]}"

    echo "[node$idx] Setting up $host..."

    ssh -p "$SSH_PORT" \
        -o StrictHostKeyChecking=no \
        -o BatchMode=yes \
        "$SSH_USER@$host" bash << EOF

if [[ ! -d ~/memLoRA-edge ]]; then
    echo "[node$idx] Cloning repository (branch: $BRANCH)..."
    git clone -b $BRANCH $REPO_URL ~/memLoRA-edge
else
    echo "[node$idx] Repo already exists, ensuring $BRANCH branch..."
    cd ~/memLoRA-edge
    if git rev-parse --verify $BRANCH >/dev/null 2>&1; then
        git checkout $BRANCH
    else
        git checkout -b $BRANCH
    fi
    git pull origin $BRANCH
fi

cd ~/memLoRA-edge

echo "[node$idx] Running setup-mock-vm.sh..."
bash scripts/setup-mock-vm.sh --no-sudo $idx $HOSTS_STR

echo "[node$idx] Opening firewall port $SERVE_PORT (best effort)..."
sudo ufw allow $SERVE_PORT/tcp 2>/dev/null || true
sudo firewall-cmd --permanent --add-port=$SERVE_PORT/tcp 2>/dev/null || true
sudo firewall-cmd --reload 2>/dev/null || true

echo "[node$idx] Setup complete."
EOF

    echo "[node$idx] ✓ $host setup done"
}

# ── Step 3: Start server on each node ────────────────────────────────────────
start_node() {
    local idx=$1
    local host="${HOSTS[$idx]}"

    echo "[node$idx] Starting server on $host..."

    ssh -p "$SSH_PORT" \
        -o StrictHostKeyChecking=no \
        -o BatchMode=yes \
        "$SSH_USER@$host" bash << EOF
set -euo pipefail

pkill -f "python -m scripts.deploy" 2>/dev/null && echo "[node$idx] Stopped existing server" || true
sleep 1

mkdir -p ~/logs

source ~/venv/bin/activate
cd ~/$REPO_NAME

export MEMLORA_MOCK=$MEMLORA_MOCK
export ROUTING_MODE=$ROUTING_MODE
export USE_S3_ADAPTERS=$USE_S3_ADAPTERS
export SERVE_PORT=$SERVE_PORT

nohup python -m scripts.deploy > ~/logs/deploy.log 2>&1 &
echo "[node$idx] Started pid \$!"
EOF

    echo "[node$idx] ✓ $host server started"
}

# ── Step 4: Health check ──────────────────────────────────────────────────────
health_check() {
    echo ""
    echo "==> Health checks (waiting 5s for servers to come up...)"
    sleep 5
    echo ""

    for idx in "${!HOSTS[@]}"; do
        host="${HOSTS[$idx]}"
        echo -n "    node$idx ($host): "
        curl -s --max-time 5 "http://$host:$SERVE_PORT/health" \
            | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print('ok | queue:', d.get('ongoing', '?'), '| node:', d.get('node', '?'))
except:
    print('ERROR: invalid response')
" 2>/dev/null || echo "UNREACHABLE"
    done
}

# ── Main ──────────────────────────────────────────────────────────────────────

setup_ssh_keys

echo "==> Step 2: Setting up all nodes (parallel)..."
echo ""
pids=()
for idx in "${!HOSTS[@]}"; do
    setup_node "$idx" &
    pids+=($!)
done

failed=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "✗ node$i setup FAILED"
        failed=$((failed + 1))
    fi
done

[[ $failed -gt 0 ]] && echo "WARNING: $failed node(s) failed setup. Continuing..."
echo ""

echo "==> Step 3: Starting servers on all nodes (parallel)..."
echo ""
pids=()
for idx in "${!HOSTS[@]}"; do
    start_node "$idx" &
    pids+=($!)
done

for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "✗ node$i start FAILED"
    fi
done

health_check

echo ""
echo "==> Done. To check logs on any node:"
echo "    ssh $SSH_USER@${HOSTS[0]} 'tail -f ~/logs/deploy.log'"
echo ""
echo "==> To send a test request:"
echo "    curl -X POST http://${HOSTS[0]}:$SERVE_PORT/v1/chat/completions \\"
echo "        -H 'Content-Type: application/json' \\"
echo "        -d '{\"model\":\"qwen-base/crop_corn_disease\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}]}'"