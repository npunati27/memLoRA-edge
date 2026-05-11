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

echo "==> Clearing logs on $NUM_NODES nodes..."

for host in "${HOSTS[@]}"; do
    echo -n "    $host: "
    ssh "$SSH_USER@$host" "rm -f ~/logs/* && echo 'cleared'" 2>/dev/null || echo "unreachable"
done

echo ""
echo "==> Done."