#!/usr/bin/env bash
# Collects metrics files from all nodes and commits them to bloom sequentially.
# Usage: bash collect_metrics.sh <results_dir>
# Example: bash collect_metrics.sh results/run1_cost_zipf
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
REPO_PATH="~/memLoRA-edge"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <results_dir>" >&2
    echo "Example: $0 results/run1_cost_zipf" >&2
    exit 1
fi

RESULTS_DIR="$1"

echo "==> Collecting metrics into $RESULTS_DIR"
echo "    Processing nodes sequentially to avoid git conflicts..."
echo ""

for idx in "${!ALL_HOSTS[@]}"; do
    host="${ALL_HOSTS[$idx]}"
    metrics_file="metrics_${host}.jsonl"

    echo "── node$idx ($host)"

    # check if metrics file exists on this node
    exists=$(ssh "$SSH_USER@$host" "test -f ~/logs/$metrics_file && echo yes || echo no" 2>/dev/null)
    if [[ "$exists" != "yes" ]]; then
        echo "   SKIP: ~/logs/$metrics_file not found on $host"
        echo ""
        continue
    fi

    # on the node: pull latest, copy metrics file into results dir, commit and push
    ssh "$SSH_USER@$host" bash << EOF
set -euo pipefail

cd $REPO_PATH

# make sure we're on bloom and up to date
git checkout bloom
git pull origin bloom

# create results directory if needed
mkdir -p "$RESULTS_DIR"

# copy metrics file into repo
cp ~/logs/$metrics_file "$RESULTS_DIR/$metrics_file"

# commit and push
git add "$RESULTS_DIR/$metrics_file"
git diff --cached --quiet && echo "   already committed, skipping" || git commit -m "metrics: add $metrics_file to $RESULTS_DIR"
git push origin bloom

echo "   pushed $metrics_file"
EOF

    echo "   ✓ node$idx done"
    echo ""
done

echo "==> All metrics collected into $RESULTS_DIR on bloom."
echo "    Pull locally to see results:"
echo "    git pull origin bloom"