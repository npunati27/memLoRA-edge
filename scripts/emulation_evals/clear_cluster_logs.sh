#!/usr/bin/env bash
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
)

SSH_USER="${SSH_USER:-npunati2}"

echo "==> Clearing logs on all nodes..."

for host in "${ALL_HOSTS[@]}"; do
    echo -n "    $host: "
    ssh "$SSH_USER@$host" "rm -f ~/logs/* && echo 'cleared'" 2>/dev/null || echo "unreachable"
done

echo ""
echo "==> Done."