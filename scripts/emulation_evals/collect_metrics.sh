#!/usr/bin/env bash
# Collects metrics files from all nodes and commits them to a branch sequentially.
# Usage: bash collect_metrics.sh [--nodes N] [--branch BRANCH] <results_dir>
# Example: bash collect_metrics.sh results/run1_cost_zipf
#          bash collect_metrics.sh --nodes 4 --branch bloom results/run1_cost_zipf
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
REPO_PATH="~/memLoRA-edge"
NUM_NODES=8
BRANCH="main"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nodes)  NUM_NODES="$2"; shift 2 ;;
        --branch) BRANCH="$2";    shift 2 ;;
        -*) echo "Unknown arg: $1" >&2; exit 1 ;;
        *) break ;;
    esac
done

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 [--nodes N] [--branch BRANCH] <results_dir>" >&2
    echo "Example: $0 results/run1_cost_zipf" >&2
    echo "         $0 --nodes 4 --branch bloom results/run1_cost_zipf" >&2
    exit 1
fi

if (( NUM_NODES < 1 || NUM_NODES > ${#ALL_HOSTS[@]} )); then
    echo "Error: --nodes must be between 1 and ${#ALL_HOSTS[@]} (got $NUM_NODES)" >&2
    exit 1
fi

RESULTS_DIR="$1"
HOSTS=("${ALL_HOSTS[@]:0:$NUM_NODES}")

echo "==> Collecting metrics into $RESULTS_DIR (branch: $BRANCH)"
echo "    Nodes: $NUM_NODES"
echo "    Processing sequentially to avoid git conflicts..."
echo ""

for idx in "${!HOSTS[@]}"; do
    host="${HOSTS[$idx]}"
    metrics_file="metrics_${host}.jsonl"

    echo "── node$idx ($host)"

    exists=$(ssh "$SSH_USER@$host" "test -f ~/logs/$metrics_file && echo yes || echo no" 2>/dev/null)
    if [[ "$exists" != "yes" ]]; then
        echo "   SKIP: ~/logs/$metrics_file not found on $host"
        echo ""
        continue
    fi

    ssh "$SSH_USER@$host" bash << EOF
set -euo pipefail

cd $REPO_PATH

# Check out or create the local branch
if git rev-parse --verify $BRANCH > /dev/null 2>&1; then
    git checkout $BRANCH
else
    echo "   Local branch $BRANCH not found — creating it."
    git checkout -b $BRANCH
fi

# Pull only if the remote branch already exists
if git ls-remote --exit-code --heads origin $BRANCH > /dev/null 2>&1; then
    git pull origin $BRANCH
else
    echo "   Remote branch $BRANCH does not exist yet — will create on push."
fi

mkdir -p "$RESULTS_DIR"

cp ~/logs/$metrics_file "$RESULTS_DIR/$metrics_file"

git add "$RESULTS_DIR/$metrics_file"
git diff --cached --quiet && echo "   already committed, skipping" || git commit -m "metrics: add $metrics_file to $RESULTS_DIR"
git push --set-upstream origin $BRANCH

echo "   pushed $metrics_file"
EOF

    echo "   ✓ node$idx done"
    echo ""
done

echo "==> All metrics collected into $RESULTS_DIR on $BRANCH."
echo "    Pull locally to see results:"
echo "    git pull origin $BRANCH"
