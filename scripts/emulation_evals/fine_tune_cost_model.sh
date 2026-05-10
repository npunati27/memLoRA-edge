#!/usr/bin/env bash
# Sweeps cost model coefficients (queue/memory/network) across workload types.
# Calls run_eval.sh for each (config, workload) combination.
# All results land on branch: cost_finetuning
#
# Usage:
#   bash fine_tune_cost_model.sh [options]
#
# Example:
#   bash fine_tune_cost_model.sh --rps 2 --duration 120 --nodes 8
#   bash fine_tune_cost_model.sh --dry-run          # preview without running

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Fixed for this sweep ──────────────────────────────────────────────────────
BRANCH="cost_finetuning"
BASE_RESULTS_DIR="results_cost_finetuning"
ROUTING_MODE="cost"

# ── Tuneable defaults ─────────────────────────────────────────────────────────
RPS=2
DURATION=120
NODES=8
MOCK="1"
S3="1"
SSH_USER="${SSH_USER:-npunati2}"
SSH_PORT="${SSH_PORT:-22}"
COOLDOWN=20   # seconds between runs for cluster to settle
DRY_RUN=0

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rps)      RPS="$2";      shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        --nodes)    NODES="$2";    shift 2 ;;
        --mock)     MOCK="$2";     shift 2 ;;
        --s3)       S3="$2";       shift 2 ;;
        --cooldown) COOLDOWN="$2"; shift 2 ;;
        --user)     SSH_USER="$2"; shift 2 ;;
        --port)     SSH_PORT="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=1;     shift   ;;
        -h|--help)
            cat <<EOF
Usage: $0 [options]

Options:
  --rps N        Requests per second for each workload run (default: 2)
  --duration N   Duration in seconds for each workload run (default: 120)
  --nodes N      Number of cluster nodes (default: 8)
  --mock 0|1     Use mock engine (default: 1)
  --s3 0|1       Use S3 adapters (default: 1)
  --cooldown N   Seconds to wait between runs (default: 20)
  --user USER    SSH username (default: \$SSH_USER or npunati2)
  --port PORT    SSH port (default: 22)
  --dry-run      Print planned runs without executing them
EOF
            exit 0 ;;
        *) echo "Error: unknown argument '$1'" >&2; exit 1 ;;
    esac
done

# ── Cost configs: "label  w_queue  w_memory  w_network" ──────────────────────
# Weights do not need to sum to exactly 1.0 — the cost model uses them as-is.
# Each row tests a different priority hypothesis.
CONFIGS=(
    "default        0.4   0.4   0.2"   # current baseline
    "queue_heavy    0.6   0.2   0.2"   # prioritise load balancing
    "memory_heavy   0.2   0.6   0.2"   # prioritise adapter locality
    "network_heavy  0.2   0.2   0.6"   # prioritise low-RTT nodes
    "balanced       0.34  0.33  0.33"  # equal weight
    "queue_memory   0.45  0.45  0.1"   # queue + memory, ignore network
    "queue_network  0.45  0.1   0.45"  # queue + network, ignore memory
    "memory_network 0.1   0.45  0.45"  # memory + network, ignore queue
)

WORKLOADS=(zipf uniform burst)

TOTAL=$(( ${#CONFIGS[@]} * ${#WORKLOADS[@]} ))

# ── Header ────────────────────────────────────────────────────────────────────
echo "========================================================"
echo "  Cost Model Fine-Tuning Sweep"
echo "========================================================"
echo "  branch:     $BRANCH"
echo "  base dir:   $BASE_RESULTS_DIR"
echo "  configs:    ${#CONFIGS[@]}"
echo "  workloads:  ${WORKLOADS[*]}"
echo "  total runs: $TOTAL"
echo "  rps:        $RPS  |  duration: ${DURATION}s  |  nodes: $NODES"
[[ $DRY_RUN -eq 1 ]] && echo "  *** DRY RUN — no commands will execute ***"
echo "========================================================"
echo ""

# ── Sweep ─────────────────────────────────────────────────────────────────────
run_num=0
for config in "${CONFIGS[@]}"; do
    read -r label wq wm wn <<< "$config"

    for workload in "${WORKLOADS[@]}"; do
        run_num=$(( run_num + 1 ))

        # Directory name encodes both the config label and the workload
        results_dir="${BASE_RESULTS_DIR}/${label}/${workload}"

        echo "────────────────────────────────────────────────────────"
        echo "  Run $run_num / $TOTAL"
        echo "  config:    $label  (queue=$wq  memory=$wm  network=$wn)"
        echo "  workload:  $workload"
        echo "  results:   $results_dir"
        echo "────────────────────────────────────────────────────────"

        if [[ $DRY_RUN -eq 1 ]]; then
            echo "  [DRY RUN] bash run_eval.sh \\"
            echo "      --results-dir    $results_dir \\"
            echo "      --branch         $BRANCH \\"
            echo "      --routing-mode   $ROUTING_MODE \\"
            echo "      --workload-mode  $workload \\"
            echo "      --rps            $RPS \\"
            echo "      --duration       $DURATION \\"
            echo "      --nodes          $NODES \\"
            echo "      --cost-w-queue   $wq \\"
            echo "      --cost-w-memory  $wm \\"
            echo "      --cost-w-network $wn \\"
            echo "      --mock           $MOCK \\"
            echo "      --s3             $S3"
            echo ""
            continue
        fi

        bash "$SCRIPT_DIR/run_eval.sh" \
            --results-dir    "$results_dir" \
            --branch         "$BRANCH" \
            --routing-mode   "$ROUTING_MODE" \
            --workload-mode  "$workload" \
            --rps            "$RPS" \
            --duration       "$DURATION" \
            --nodes          "$NODES" \
            --cost-w-queue   "$wq" \
            --cost-w-memory  "$wm" \
            --cost-w-network "$wn" \
            --mock           "$MOCK" \
            --s3             "$S3" \
            --user           "$SSH_USER" \
            --port           "$SSH_PORT"

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
    for config in "${CONFIGS[@]}"; do
        read -r label wq wm wn <<< "$config"
        echo "      $label/  (q=$wq m=$wm n=$wn)"
        for workload in "${WORKLOADS[@]}"; do
            echo "        $workload/  results.jsonl  summary.txt  metrics_*.jsonl"
        done
    done
fi
echo "========================================================"
