#!/usr/bin/env bash
# Coordinator script: sets up and starts memLoRA-edge on all 8 nodes.
# Usage: bash run_cluster.sh [--routing-mode cost|memory|baseline] [--mock 0|1] [--s3 0|1]
# Run from your local machine or any of the VMs.

# default: mock=1, routing=cost, s3=1
#bash run_cluster.sh

# custom routing mode
#bash run_cluster.sh --routing-mode baseline

# different user
#bash run_cluster.sh --user yournetid

# no mock (real vllm)
#bash run_cluster.sh --mock 0

set -euo pipefail

# ── Node configuration ────────────────────────────────────────────────────────
ALL_HOSTS=(
    sp26-cs525-0701.cs.illinois.edu   # node 0
    sp26-cs525-0702.cs.illinois.edu   # node 1
    sp26-cs525-0703.cs.illinois.edu   # node 2
    sp26-cs525-0704.cs.illinois.edu   # node 3
    sp26-cs525-0705.cs.illinois.edu   # node 4
    sp26-cs525-0706.cs.illinois.edu   # node 5
    sp26-cs525-0707.cs.illinois.edu   # node 6
    sp26-cs525-0708.cs.illinois.edu   # node 7
)

SSH_USER="${SSH_USER:-npunati2}"
SSH_PORT="${SSH_PORT:-22}"
REPO_URL="https://github.com/npunati27/memLoRA-edge.git"
REPO_NAME="memLoRA-edge"
SERVE_PORT="${SERVE_PORT:-5000}"

# ── Defaults (can be overridden via args) ─────────────────────────────────────
ROUTING_MODE="cost"
MEMLORA_MOCK="1"
USE_S3_ADAPTERS="1"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --routing-mode) ROUTING_MODE="$2"; shift 2 ;;
        --mock)         MEMLORA_MOCK="$2"; shift 2 ;;
        --s3)           USE_S3_ADAPTERS="$2"; shift 2 ;;
        --user)         SSH_USER="$2"; shift 2 ;;
        --port)         SSH_PORT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--routing-mode cost|memory|baseline] [--mock 0|1] [--s3 0|1] [--user USER] [--port PORT]"
            exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

ALL_HOSTS_STR="${ALL_HOSTS[*]}"
NUM_NODES=${#ALL_HOSTS[@]}

echo "==> memLoRA-edge cluster setup"
echo "    nodes:        $NUM_NODES"
echo "    user:         $SSH_USER"
echo "    routing_mode: $ROUTING_MODE"
echo "    mock:         $MEMLORA_MOCK"
echo "    s3:           $USE_S3_ADAPTERS"
echo "    serve_port:   $SERVE_PORT"
echo ""

# ── Step 1: SSH key setup ─────────────────────────────────────────────────────
setup_ssh_keys() {
    echo "==> Step 1: Setting up SSH keys (you will be prompted for passwords)"
    echo ""

    # generate key if it doesn't exist
    if [[ ! -f ~/.ssh/id_ed25519 ]]; then
        echo "    Generating SSH key..."
        ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" -C "memlora-cluster"
    else
        echo "    SSH key already exists at ~/.ssh/id_ed25519"
    fi

    # copy key to each node — will prompt for password once per node
    for host in "${ALL_HOSTS[@]}"; do
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
    local host="${ALL_HOSTS[$idx]}"

    echo "[node$idx] Setting up $host..."

    ssh -p "$SSH_PORT" \
        -o StrictHostKeyChecking=no \
        -o BatchMode=yes \
        "$SSH_USER@$host" bash << EOF
# no set -euo pipefail here — we want to continue past sudo failures

# clone repo if not already there, explicitly using the bloom branch
if [[ ! -d ~/memLoRA-edge ]]; then
    echo "[node$idx] Cloning repository (branch: bloom)..."
    git clone -b bloom $REPO_URL ~/memLoRA-edge
else
    echo "[node$idx] Repo already exists, ensuring bloom branch is active..."
    cd ~/memLoRA-edge
    
    # Check if bloom branch exists locally
    if git rev-parse --verify bloom >/dev/null 2>&1; then
        git checkout bloom
    else
        echo "[node$idx] Creating local bloom branch..."
        git checkout -b bloom
    fi
    
    git pull origin bloom
fi

cd ~/memLoRA-edge

# run setup — no sudo so dnf is skipped
echo "[node$idx] Running setup-mock-vm.sh..."
bash scripts/setup-mock-vm.sh --no-sudo $idx $ALL_HOSTS_STR

# open firewall — best effort, don't fail if sudo not available
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
    local host="${ALL_HOSTS[$idx]}"

    echo "[node$idx] Starting server on $host..."

    ssh -p "$SSH_PORT" \
        -o StrictHostKeyChecking=no \
        -o BatchMode=yes \
        "$SSH_USER@$host" bash << EOF
set -euo pipefail

# kill any existing instance
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

# ── Step 4: Health check all nodes ───────────────────────────────────────────
health_check() {
    echo ""
    echo "==> Health checks (waiting 5s for servers to come up...)"
    sleep 5
    echo ""

    for idx in "${!ALL_HOSTS[@]}"; do
        host="${ALL_HOSTS[$idx]}"
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

# Step 1: SSH keys
setup_ssh_keys

# Step 2: Setup all nodes in parallel
echo "==> Step 2: Setting up all nodes (parallel)..."
echo ""
pids=()
for idx in "${!ALL_HOSTS[@]}"; do
    setup_node "$idx" &
    pids+=($!)
done

# wait for all setup jobs and report failures
failed=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "✗ node$i setup FAILED"
        failed=$((failed + 1))
    fi
done

if [[ $failed -gt 0 ]]; then
    echo ""
    echo "WARNING: $failed node(s) failed setup. Continuing with the rest..."
fi

echo ""

# Step 3: Start all nodes in parallel
echo "==> Step 3: Starting servers on all nodes (parallel)..."
echo ""
pids=()
for idx in "${!ALL_HOSTS[@]}"; do
    start_node "$idx" &
    pids+=($!)
done

for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "✗ node$i start FAILED"
    fi
done

# Step 4: Health checks
health_check

echo ""
echo "==> Done. To check logs on any node:"
echo "    ssh $SSH_USER@sp26-cs525-0701.cs.illinois.edu 'tail -f ~/logs/deploy.log'"
echo ""
echo "==> To send a test request:"
echo "    curl -X POST http://sp26-cs525-0701.cs.illinois.edu:$SERVE_PORT/v1/chat/completions \\"
echo "        -H 'Content-Type: application/json' \\"
echo "        -d '{\"model\":\"qwen-base/crop_corn_disease\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}]}'"