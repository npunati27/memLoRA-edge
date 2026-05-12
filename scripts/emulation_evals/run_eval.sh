#!/usr/bin/env bash
# Full evaluation pipeline for memLoRA-edge
#
# Steps:
#   1. Start cluster     (run_mock_cluster.sh)
#   2. Run workload      (workloads/workload_distributed.py)
#      └─ saves results.jsonl + summary.txt locally, then git-pushes them
#   3. Stop cluster      (stop_cluster.sh)
#   4. Collect metrics   (collect_metrics.sh)  — each node pushes its metrics to git
#   5. Clear logs        (clear_cluster_logs.sh)
#
# All output lands in --results-dir, both locally and in git on --branch.
#
# Usage:
#   bash run_eval.sh --results-dir <path> [options]
#
# Example:
#   bash run_eval.sh \
#       --results-dir results_bloom/cost_zipf \
#       --branch      bloom \
#       --routing-mode cost \
#       --workload-mode zipf \
#       --rps 2 --duration 120 --nodes 8

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKLOADS_DIR="$REPO_ROOT/workloads"

# ── Defaults ──────────────────────────────────────────────────────────────────
RESULTS_DIR=""
BRANCH="main"
NUM_NODES=8
ROUTING_MODE="cost"
WORKLOAD_MODE="zipf"
RPS=2
DURATION=120
MOCK="1"
S3="1"
CONCURRENCY=16
NODE_STRATEGY="random"
SERVE_PORT="${SERVE_PORT:-5000}"
COST_W_QUEUE="0.4"
COST_W_MEMORY="0.4"
COST_W_NETWORK="0.2"
SSH_USER="${SSH_USER:-npunati2}"
SSH_PORT="${SSH_PORT:-22}"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --results-dir)    RESULTS_DIR="$2";    shift 2 ;;
        --branch)         BRANCH="$2";         shift 2 ;;
        --nodes)          NUM_NODES="$2";      shift 2 ;;
        --routing-mode)   ROUTING_MODE="$2";   shift 2 ;;
        --workload-mode)  WORKLOAD_MODE="$2";  shift 2 ;;
        --rps)            RPS="$2";            shift 2 ;;
        --duration)       DURATION="$2";       shift 2 ;;
        --mock)           MOCK="$2";           shift 2 ;;
        --s3)             S3="$2";             shift 2 ;;
        --concurrency)    CONCURRENCY="$2";    shift 2 ;;
        --node-strategy)  NODE_STRATEGY="$2";  shift 2 ;;
        --serve-port)     SERVE_PORT="$2";     shift 2 ;;
        --cost-w-queue)   COST_W_QUEUE="$2";   shift 2 ;;
        --cost-w-memory)  COST_W_MEMORY="$2";  shift 2 ;;
        --cost-w-network) COST_W_NETWORK="$2"; shift 2 ;;
        --user)           SSH_USER="$2";       shift 2 ;;
        --port)           SSH_PORT="$2";       shift 2 ;;
        -h|--help)
            cat <<EOF
Usage: $0 --results-dir <path> [options]

Required:
  --results-dir PATH      Path relative to repo root (e.g. results_bloom/cost_zipf).
                          results.jsonl, summary.txt, and node metrics all land here.

Git options:
  --branch BRANCH         Branch to commit and push results to (default: main)

Cluster options (run_mock_cluster.sh / stop_cluster.sh):
  --nodes N               Number of nodes, 1–20 (default: 8)
  --routing-mode MODE     cost|memory|baseline (default: cost)
  --mock 0|1              Use mock engine (default: 1)
  --s3 0|1                Use S3 adapters (default: 1)
  --serve-port PORT       Port each node listens on (default: 5000)
  --user USER             SSH username (default: \$SSH_USER or npunati2)
  --port PORT             SSH port (default: 22)

Workload options (workload_distributed.py):
  --workload-mode MODE    uniform|zipf|burst|all (default: zipf)
  --rps N                 Requests per second (default: 2)
  --duration N            Duration in seconds (default: 120)
  --concurrency N         Max concurrent in-flight requests (default: 16)
  --node-strategy STR     How to pick target node: random (default: random)
  --cost-w-queue W        Cost weight for queue length (default: 0.4)
  --cost-w-memory W       Cost weight for memory tier (default: 0.4)
  --cost-w-network W      Cost weight for network RTT (default: 0.2)
EOF
            exit 0 ;;
        *) echo "Error: unknown argument '$1'" >&2; exit 1 ;;
    esac
done

# ── Validation ────────────────────────────────────────────────────────────────
if [[ -z "$RESULTS_DIR" ]]; then
    echo "Error: --results-dir is required." >&2
    echo "Run '$0 --help' for usage." >&2
    exit 1
fi

if (( NUM_NODES < 1 || NUM_NODES > 20 )); then
    echo "Error: --nodes must be between 1 and 20 (got $NUM_NODES)" >&2
    exit 1
fi

# ── Derived values ────────────────────────────────────────────────────────────
LOCAL_RESULTS_DIR="$REPO_ROOT/$RESULTS_DIR"
RESULTS_JSONL="$LOCAL_RESULTS_DIR/results.jsonl"
SUMMARY_FILE="$LOCAL_RESULTS_DIR/summary.txt"

# Build node URL list for the workload script from the node count + port
WORKLOAD_NODES=()
for i in $(seq 1 "$NUM_NODES"); do
    WORKLOAD_NODES+=("http://sp26-cs525-07$(printf '%02d' "$i").cs.illinois.edu:$SERVE_PORT")
done

mkdir -p "$LOCAL_RESULTS_DIR"

# ── Header ────────────────────────────────────────────────────────────────────
TS_START="$(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
echo "  memLoRA-edge Evaluation Pipeline"
echo "========================================================"
echo "  started:        $TS_START"
echo "  branch:         $BRANCH"
echo "  results dir:    $LOCAL_RESULTS_DIR"
echo "  nodes:          $NUM_NODES"
echo "  routing mode:   $ROUTING_MODE"
echo "  workload mode:  $WORKLOAD_MODE"
echo "  rps:            $RPS"
echo "  duration:       ${DURATION}s"
echo "  concurrency:    $CONCURRENCY"
echo "  node strategy:  $NODE_STRATEGY"
echo "  mock:           $MOCK"
echo "  s3:             $S3"
echo "  serve port:     $SERVE_PORT"
echo "  ssh user:       $SSH_USER"
echo "========================================================"
echo ""

# ── Step 1: Start cluster ─────────────────────────────────────────────────────
echo "==> [1/5] Starting cluster ($NUM_NODES nodes)..."
SSH_USER="$SSH_USER" SSH_PORT="$SSH_PORT" SERVE_PORT="$SERVE_PORT" \
    bash "$SCRIPT_DIR/run_mock_cluster.sh" \
        --nodes           "$NUM_NODES" \
        --branch          "$BRANCH" \
        --routing-mode    "$ROUTING_MODE" \
        --mock            "$MOCK" \
        --s3              "$S3" \
        --cost-w-queue    "$COST_W_QUEUE" \
        --cost-w-memory   "$COST_W_MEMORY" \
        --cost-w-network  "$COST_W_NETWORK" \
        --user            "$SSH_USER" \
        --port            "$SSH_PORT"
echo ""

# ── Step 2: Run workload ──────────────────────────────────────────────────────
echo "==> [2/5] Running workload..."
echo "    output → $SUMMARY_FILE"
echo "    jsonl  → $RESULTS_JSONL"
echo ""

WORKLOAD_EXIT=0
set +e
python3 "$WORKLOADS_DIR/workload_distributed.py" \
    --mode          "$WORKLOAD_MODE" \
    --nodes         "${WORKLOAD_NODES[@]}" \
    --node-strategy "$NODE_STRATEGY" \
    --rps           "$RPS" \
    --duration      "$DURATION" \
    --routing       "$ROUTING_MODE" \
    --out           "$RESULTS_JSONL" \
    --concurrency   "$CONCURRENCY" \
    2>&1 | tee "$SUMMARY_FILE"
WORKLOAD_EXIT="${PIPESTATUS[0]}"
set -e

echo ""
if [[ $WORKLOAD_EXIT -ne 0 ]]; then
    echo "WARNING: workload exited with code $WORKLOAD_EXIT"
fi

# Push results.jsonl + summary.txt to git
echo "    Committing and pushing workload results to $BRANCH..."
cd "$REPO_ROOT"

# Check out or create the local branch
if git rev-parse --verify "$BRANCH" > /dev/null 2>&1; then
    git checkout "$BRANCH"
else
    echo "    Local branch $BRANCH not found — creating it."
    git checkout -b "$BRANCH"
fi

# Pull only if the remote branch already exists
if git ls-remote --exit-code --heads origin "$BRANCH" > /dev/null 2>&1; then
    git pull origin "$BRANCH"
else
    echo "    Remote branch $BRANCH does not exist yet — will create on push."
fi

git add "$RESULTS_DIR/results.jsonl" "$RESULTS_DIR/summary.txt"
if git diff --cached --quiet; then
    echo "    Nothing new to commit (files unchanged)."
else
    git commit -m "eval: add workload results for $RESULTS_DIR"
    git push --set-upstream origin "$BRANCH"
    echo "    Pushed results.jsonl and summary.txt to $BRANCH."
fi
echo ""

# ── Step 3: Stop cluster ──────────────────────────────────────────────────────
echo "==> [3/5] Stopping cluster..."
SSH_USER="$SSH_USER" SSH_PORT="$SSH_PORT" \
    bash "$SCRIPT_DIR/stop_cluster.sh" \
        --nodes "$NUM_NODES"
echo ""

# ── Step 4: Collect metrics ───────────────────────────────────────────────────
echo "==> [4/5] Collecting node metrics into $RESULTS_DIR (branch: $BRANCH)..."
SSH_USER="$SSH_USER" \
    bash "$SCRIPT_DIR/collect_metrics.sh" \
        --nodes  "$NUM_NODES" \
        --branch "$BRANCH" \
        "$RESULTS_DIR"
echo ""

# ── Step 5: Clear cluster logs ────────────────────────────────────────────────
echo "==> [5/5] Clearing cluster logs..."
SSH_USER="$SSH_USER" \
    bash "$SCRIPT_DIR/clear_cluster_logs.sh" \
        --nodes "$NUM_NODES"
echo ""

# ── Done ──────────────────────────────────────────────────────────────────────
TS_END="$(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
echo "  Pipeline complete — $TS_END"
echo "  Branch: $BRANCH"
echo "  Local results dir: $LOCAL_RESULTS_DIR"
echo "    results.jsonl  — workload request log"
echo "    summary.txt    — printed workload summary"
echo "    metrics_*.jsonl — per-node server metrics (after git pull)"
echo ""
echo "  Sync all results locally:"
echo "    git pull origin $BRANCH"
echo "========================================================"

exit "$WORKLOAD_EXIT"
