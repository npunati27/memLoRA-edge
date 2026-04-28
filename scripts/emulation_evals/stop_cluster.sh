#!/usr/bin/env bash
# Stops the deploy server on all 8 nodes.
# Usage: bash stop_cluster.sh

ALL_HOSTS=(
    sp26-cs525-0701.cs.illinois.edu
    sp26-cs525-0702.cs.illinois.edu
    sp26-cs525-0703.cs.illinois.edu
    sp26-cs525-0704.cs.illinois.edu
    sp26-cs525-0705.cs.illinois.edu
    sp26-cs525-0706.cs.illinois.edu
    sp26-cs525-0707.cs.illinois.edu
    sp26-cs525-0708.cs.illinois.edu
)

SSH_USER="${SSH_USER:-npunati2}"
SSH_PORT="${SSH_PORT:-22}"

echo "==> Stopping deploy servers on all nodes..."

stop_node() {
    local idx=$1
    local host="${ALL_HOSTS[$idx]}"
    echo -n "    node$idx ($host): "
    ssh -p "$SSH_PORT" \
        -o StrictHostKeyChecking=no \
        -o BatchMode=yes \
        "$SSH_USER@$host" \
        "pkill -f 'python -m scripts.deploy' 2>/dev/null && echo stopped || echo 'not running'; rm -f ~/logs/* && echo '| logs cleared'" \
        2>/dev/null || echo "unreachable"
}

for idx in "${!ALL_HOSTS[@]}"; do
    stop_node "$idx" &
done
wait

echo ""
echo "==> All nodes stopped."