#!/usr/bin/env bash
# Sweeps cluster sizes (4, 8, 12, 16) across workload types without bloom filtering.
# Calls run_eval.sh for each (node count, workload) combination.
# All results land on branch: no_bloom
#
# Usage:
#   bash run_no_bloom_evals.sh [options]
#
# Example:
#   bash run_no_bloom_evals.sh --rps 2 --duration 120
#   bash run_no_bloom_evals.sh --dry-run    # preview without running

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Fixed for this sweep ──────────────────────────────────────────────────────
BRANCH="old-main"
BASE_RESULTS_DIR="results_no_bloom"
ROUTING_MODE="cost"

# ── Tuneable defaults ─────────────────────────────────────────────────────────
RPS=2
DURATION=120
MOCK="1"
S3="1"
SSH_USER="${SSH_USER:-npunati2}"
SSH_PORT="${SSH_PORT:-22}"
COOLDOWN=20   # seconds between runs for cluster to settle
DRY_RUN=0

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rps)          RPS="$2";          shift 2 ;;
        --duration)     DURATION="$2";     shift 2 ;;
        --routing-mode) ROUTING_MODE="$2"; shift 2 ;;
        --mock)         MOCK="$2";         shift 2 ;;
        --s3)           S3="$2";           shift 2 ;;
        --cooldown)     COOLDOWN="$2";     shift 2 ;;
        --user)         SSH_USER="$2";     shift 2 ;;
        --port)         SSH_PORT="$2";     shift 2 ;;
        --dry-run)      DRY_RUN=1;         shift   ;;
        -h|--help)
            cat <<EOF
Usage: $0 [options]

Options:
  --rps N          Requests per second for each workload run (default: 2)
  --duration N     Duration in seconds for each workload run (default: 120)
  --routing-mode   cost|memory|baseline (default: cost)
  --mock 0|1       Use mock engine (default: 1)
  --s3 0|1         Use S3 adapters (default: 1)
  --cooldown N     Seconds to wait between runs (default: 20)
  --user USER      SSH username (default: \$SSH_USER or npunati2)
  --port PORT      SSH port (default: 22)
  --dry-run        Print planned runs without executing them
EOF
            exit 0 ;;
        *) echo "Error: unknown argument '$1'" >&2; exit 1 ;;
    esac
done

# ── Sweep dimensions ──────────────────────────────────────────────────────────
NODE_SIZES=(19)
WORKLOADS=(zipf uniform burst)

TOTAL=$(( ${#NODE_SIZES[@]} * ${#WORKLOADS[@]} ))

# ── Header ────────────────────────────────────────────────────────────────────
echo "========================================================"
echo "  no_bloom Cluster Size Sweep"
echo "========================================================"
echo "  branch:       $BRANCH"
echo "  base dir:     $BASE_RESULTS_DIR"
echo "  node sizes:   ${NODE_SIZES[*]}"
echo "  workloads:    ${WORKLOADS[*]}"
echo "  total runs:   $TOTAL"
echo "  rps:          $RPS  |  duration: ${DURATION}s"
echo "  routing mode: $ROUTING_MODE"
[[ $DRY_RUN -eq 1 ]] && echo "  *** DRY RUN — no commands will execute ***"
echo "========================================================"
echo ""

# ── Sweep ─────────────────────────────────────────────────────────────────────
run_num=0
for N in "${NODE_SIZES[@]}"; do
    for workload in "${WORKLOADS[@]}"; do
        run_num=$(( run_num + 1 ))

        results_dir="${BASE_RESULTS_DIR}/nodes_${N}/${workload}"

        echo "────────────────────────────────────────────────────────"
        echo "  Run $run_num / $TOTAL"
        echo "  nodes:     $N"
        echo "  workload:  $workload"
        echo "  results:   $results_dir"
        echo "────────────────────────────────────────────────────────"

        if [[ $DRY_RUN -eq 1 ]]; then
            echo "  [DRY RUN] bash run_eval.sh \\"
            echo "      --results-dir   $results_dir \\"
            echo "      --branch        $BRANCH \\"
            echo "      --nodes         $N \\"
            echo "      --routing-mode  $ROUTING_MODE \\"
            echo "      --workload-mode $workload \\"
            echo "      --rps           $RPS \\"
            echo "      --duration      $DURATION \\"
            echo "      --mock          $MOCK \\"
            echo "      --s3            $S3"
            echo ""
            continue
        fi

        bash "$SCRIPT_DIR/run_eval.sh" \
            --results-dir   "$results_dir" \
            --branch        "$BRANCH" \
            --nodes         "$N" \
            --routing-mode  "$ROUTING_MODE" \
            --workload-mode "$workload" \
            --rps           "$RPS" \
            --duration      "$DURATION" \
            --mock          "$MOCK" \
            --s3            "$S3" \
            --user          "$SSH_USER" \
            --port          "$SSH_PORT"

        if [[ $run_num -lt $TOTAL ]]; then
            echo ""
            echo "  Run $run_num / $TOTAL done. Cooling down ${COOLDOWN}s..."
            sleep "$COOLDOWN"
        fi
        echo ""
    done
done

# ── Done ──────────────────────────────────────────────────────────────────────
echo "========================================================"
if [[ $DRY_RUN -eq 1 ]]; then
    echo "  Dry run complete — $TOTAL runs previewed."
else
    echo "  Sweep complete — $TOTAL runs finished."
    echo ""
    echo "  Pull all results:"
    echo "    git pull origin $BRANCH"
    echo ""
    echo "  Results tree:"
    echo "    $BASE_RESULTS_DIR/"
    for N in "${NODE_SIZES[@]}"; do
        echo "      nodes_$N/"
        for workload in "${WORKLOADS[@]}"; do
            echo "        $workload/  results.jsonl  summary.txt  metrics_*.jsonl"
        done
    done
fi
echo "========================================================"
