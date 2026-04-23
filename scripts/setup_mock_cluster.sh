#!/usr/bin/env bash
# Run this directly on each node.
# Usage: bash setup_mock_cluster.sh <node_idx>
# Example on node 2: bash setup_mock_cluster.sh 2
set -euo pipefail

# ── Configure your IPs here — must be identical on every node ────────────────
ALL_IPS=(
    sp26-cs525-0701.cs.illinois.edu   # node 0
    sp26-cs525-0702.cs.illinois.edu   # node 1
    sp26-cs525-0703.cs.illinois.edu   # node 2
    sp26-cs525-0704.cs.illinois.edu   # node 3
    sp26-cs525-0705.cs.illinois.edu   # node 4
    sp26-cs525-0706.cs.illinois.edu   # node 5
)

# Partial mesh topology:
#   node0 -- node1 -- node2
#     |                 |
#   node3 -- node4 -- node5
declare -A PEERS
PEERS[0]="sp26-cs525-0702.cs.illinois.edu,sp26-cs525-0704.cs.illinois.edu"
PEERS[1]="sp26-cs525-0701.cs.illinois.edu,sp26-cs525-0703.cs.illinois.edu"
PEERS[2]="sp26-cs525-0702.cs.illinois.edu,sp26-cs525-0706.cs.illinois.edu"
PEERS[3]="sp26-cs525-0701.cs.illinois.edu,sp26-cs525-0705.cs.illinois.edu"
PEERS[4]="sp26-cs525-0704.cs.illinois.edu,sp26-cs525-0706.cs.illinois.edu"
PEERS[5]="sp26-cs525-0703.cs.illinois.edu,sp26-cs525-0705.cs.illinois.edu"

REPO_PATH="${REPO_PATH:-$(cd "$(dirname "$0")/.." && pwd)}"
# ─────────────────────────────────────────────────────────────────────────────

usage() {
    echo "Usage: $0 <node_idx>" >&2
    echo "  node_idx — which node is this machine (0-$((${#ALL_IPS[@]} - 1)))" >&2
    echo "Example: $0 2" >&2
    exit 1
}

[[ $# -lt 1 ]] && usage

NODE_IDX=$1
NUM_NODES=${#ALL_IPS[@]}

if ! [[ "$NODE_IDX" =~ ^[0-9]+$ ]] || (( NODE_IDX < 0 || NODE_IDX >= NUM_NODES )); then
    echo "Error: node_idx must be in [0, $((NUM_NODES - 1))] (got '$NODE_IDX')" >&2
    exit 1
fi

MY_IP="${ALL_IPS[$NODE_IDX]}"
MY_PEERS="${PEERS[$NODE_IDX]}"
ALL_IPS_STR="${ALL_IPS[*]}"

echo "==> This is node$NODE_IDX (IP: $MY_IP)"
echo "    Peers: $MY_PEERS"
echo "    Repo:  $REPO_PATH"
echo ""

bash "$REPO_PATH/scripts/setup-mock-vm.sh" \
    --no-sudo \
    --peers "$MY_PEERS" \
    "$NODE_IDX" \
    $ALL_IPS_STR

echo ""
echo "==> Setup complete. To start this node:"
echo "    source ~/venv/bin/activate"
echo "    export MEMLORA_MOCK=1"
echo "    export ROUTING_MODE=cost"
echo "    cd $REPO_PATH && python -m scripts.deploy"
echo ""
echo "    Or use the start script:"
echo "    bash $REPO_PATH/scripts/start_mock_node.sh $NODE_IDX"