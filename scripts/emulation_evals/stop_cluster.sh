#!/usr/bin/env bash
# Usage: bash stop_cluster.sh [--nodes N]
set -euo pipefail

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
NUM_NODES=8

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nodes) NUM_NODES="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if (( NUM_NODES < 1 || NUM_NODES > ${#ALL_HOSTS[@]} )); then
    echo "Error: --nodes must be between 1 and ${#ALL_HOSTS[@]} (got $NUM_NODES)" >&2
    exit 1
fi

HOSTS=("${ALL_HOSTS[@]:0:$NUM_NODES}")

echo "==> Stopping deploy servers and clearing logs on $NUM_NODES nodes..."

stop_node() {
    local idx=$1
    local host="${HOSTS[$idx]}"
    echo -n "    node$idx ($host): "
    ssh -p "$SSH_PORT" \
        -o StrictHostKeyChecking=no \
        -o BatchMode=yes \
        "$SSH_USER@$host" \
        "pkill -f 'python -m scripts.deploy' 2>/dev/null && echo -n 'stopped ' || echo -n 'not running '; rm -f ~/logs/* && echo '| logs cleared'" \
        2>/dev/null || echo "unreachable"
}

for idx in "${!HOSTS[@]}"; do
    stop_node "$idx" &
done
wait

echo ""
echo "==> All nodes stopped."