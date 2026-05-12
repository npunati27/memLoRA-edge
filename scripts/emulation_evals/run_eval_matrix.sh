#!/usr/bin/env bash
# Runs run_eval.sh over a matrix of routing modes × workload modes.
# Default: both non-baseline routing modes (cost, memory) × zipf, uniform, burst.
#
# Usage:
#   bash run_eval_matrix.sh [options]
#
# Examples:
#   bash run_eval_matrix.sh --branch bloom --base-results-dir results_bloom
#   bash run_eval_matrix.sh --routing-modes cost,baseline --workloads zipf,uniform,burst
#   bash run_eval_matrix.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────────
BRANCH="main"
BASE_RESULTS_DIR="results_eval_matrix"
NUM_NODES=8
ROUTING_MODES_STR="cost,memory"
WORKLOADS_STR="zipf,uniform,burst"
RPS=2
DURATION=120
MOCK="1"
S3="1"
CONCURRENCY=16
NODE_STRATEGY="random"
SSH_USER="${SSH_USER:-ronita2}"
SSH_PORT="${SSH_PORT:-22}"
COOLDOWN=20
DRY_RUN=0

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --branch)            BRANCH="$2";            shift 2 ;;
        --base-results-dir)  BASE_RESULTS_DIR="$2";  shift 2 ;;
        --nodes)             NUM_NODES="$2";         shift 2 ;;
        --routing-modes)     ROUTING_MODES_STR="$2"; shift 2 ;;
        --workloads)         WORKLOADS_STR="$2";     shift 2 ;;
        --rps)               RPS="$2";               shift 2 ;;
        --duration)          DURATION="$2";          shift 2 ;;
        --mock)              MOCK="$2";              shift 2 ;;
        --s3)                S3="$2";                shift 2 ;;
        --concurrency)       CONCURRENCY="$2";       shift 2 ;;
        --node-strategy)     NODE_STRATEGY="$2";     shift 2 ;;
        --cooldown)          COOLDOWN="$2";          shift 2 ;;
        --user)              SSH_USER="$2";          shift 2 ;;
        --port)              SSH_PORT="$2";          shift 2 ;;
        --dry-run)           DRY_RUN=1;              shift   ;;
        -h|--help)
            cat <<EOF
Usage: $0 [options]

Runs scripts/emulation_evals/run_eval.sh once per (routing-mode × workload-mode).

Defaults match a full sweep of routing cost vs memory over zipf, uniform, and burst.
Override --routing-modes if you meant e.g. cost vs baseline (cost,baseline).

Options:
  --branch BRANCH           Git branch for results (default: main)
  --base-results-dir PATH   Under repo root; each run uses
                            <path>/<routing_mode>/<workload_mode> (default: results_eval_matrix)
  --nodes N                 Cluster size (default: 8)
  --routing-modes LIST      Comma-separated: cost,memory,baseline (default: cost,memory)
  --workloads LIST          Comma-separated: zipf,uniform,burst,all (default: zipf,uniform,burst)
  --rps N                   Requests per second (default: 2)
  --duration N              Workload duration in seconds (default: 120)
  --mock 0|1                Mock engine (default: 1)
  --s3 0|1                  S3 adapters (default: 1)
  --concurrency N           Passed through to run_eval.sh (default: 16)
  --node-strategy STR       Passed through to run_eval.sh (default: random)
  --cooldown N              Seconds between runs (default: 20)
  --user USER               SSH username (default: \$SSH_USER or npunati2)
  --port PORT               SSH port (default: 22)
  --dry-run                 Print planned runs only
EOF
            exit 0 ;;
        *) echo "Error: unknown argument '$1'" >&2; exit 1 ;;
    esac
done

valid_routing_mode() {
    case "$1" in
        cost|memory|baseline) return 0 ;;
        *) return 1 ;;
    esac
}

valid_workload_mode() {
    case "$1" in
        uniform|zipf|burst|all) return 0 ;;
        *) return 1 ;;
    esac
}

IFS=',' read -ra _ROUTING_TMP <<< "$ROUTING_MODES_STR"
ROUTING_MODES=()
for m in "${_ROUTING_TMP[@]}"; do
    m="${m//[[:space:]]/}"
    [[ -z "$m" ]] && continue
    if ! valid_routing_mode "$m"; then
        echo "Error: invalid routing mode '$m' (expected cost, memory, or baseline)" >&2
        exit 1
    fi
    ROUTING_MODES+=("$m")
done
if [[ ${#ROUTING_MODES[@]} -eq 0 ]]; then
    echo "Error: --routing-modes produced an empty list" >&2
    exit 1
fi

IFS=',' read -ra _WORKLOAD_TMP <<< "$WORKLOADS_STR"
WORKLOADS=()
for w in "${_WORKLOAD_TMP[@]}"; do
    w="${w//[[:space:]]/}"
    [[ -z "$w" ]] && continue
    if ! valid_workload_mode "$w"; then
        echo "Error: invalid workload mode '$w' (expected uniform, zipf, burst, or all)" >&2
        exit 1
    fi
    WORKLOADS+=("$w")
done
if [[ ${#WORKLOADS[@]} -eq 0 ]]; then
    echo "Error: --workloads produced an empty list" >&2
    exit 1
fi

TOTAL=$(( ${#ROUTING_MODES[@]} * ${#WORKLOADS[@]} ))

# ── Header ────────────────────────────────────────────────────────────────────
echo "========================================================"
echo "  run_eval matrix (routing × workload)"
echo "========================================================"
echo "  branch:         $BRANCH"
echo "  base results:   $BASE_RESULTS_DIR"
echo "  nodes:          $NUM_NODES"
echo "  routing modes:  ${ROUTING_MODES[*]}"
echo "  workloads:      ${WORKLOADS[*]}"
echo "  total runs:     $TOTAL"
echo "  rps:            $RPS  |  duration: ${DURATION}s"
echo "  cooldown:       ${COOLDOWN}s between runs"
[[ $DRY_RUN -eq 1 ]] && echo "  *** DRY RUN — no commands will execute ***"
echo "========================================================"
echo ""

# ── Sweep ─────────────────────────────────────────────────────────────────────
run_num=0
for routing in "${ROUTING_MODES[@]}"; do
    for workload in "${WORKLOADS[@]}"; do
        run_num=$(( run_num + 1 ))

        results_dir="${BASE_RESULTS_DIR}/${routing}/${workload}"

        echo "────────────────────────────────────────────────────────"
        echo "  Run $run_num / $TOTAL"
        echo "  routing:   $routing"
        echo "  workload:  $workload"
        echo "  results:   $results_dir"
        echo "────────────────────────────────────────────────────────"

        if [[ $DRY_RUN -eq 1 ]]; then
            echo "  [DRY RUN] bash run_eval.sh \\"
            echo "      --results-dir    $results_dir \\"
            echo "      --branch         $BRANCH \\"
            echo "      --nodes          $NUM_NODES \\"
            echo "      --routing-mode   $routing \\"
            echo "      --workload-mode  $workload \\"
            echo "      --rps            $RPS \\"
            echo "      --duration       $DURATION \\"
            echo "      --mock           $MOCK \\"
            echo "      --s3             $S3 \\"
            echo "      --concurrency    $CONCURRENCY \\"
            echo "      --node-strategy  $NODE_STRATEGY \\"
            echo "      --user           $SSH_USER \\"
            echo "      --port           $SSH_PORT"
            echo ""
            continue
        fi

        bash "$SCRIPT_DIR/run_eval.sh" \
            --results-dir    "$results_dir" \
            --branch         "$BRANCH" \
            --nodes          "$NUM_NODES" \
            --routing-mode   "$routing" \
            --workload-mode  "$workload" \
            --rps            "$RPS" \
            --duration       "$DURATION" \
            --mock           "$MOCK" \
            --s3             "$S3" \
            --concurrency    "$CONCURRENCY" \
            --node-strategy  "$NODE_STRATEGY" \
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

# ── Done ────────────────────────────────────────────────────────────────────────
echo "========================================================"
if [[ $DRY_RUN -eq 1 ]]; then
    echo "  Dry run complete — $TOTAL runs previewed."
else
    echo "  Matrix complete — $TOTAL runs finished."
    echo ""
    echo "  Pull all results:"
    echo "    git pull origin $BRANCH"
    echo ""
    echo "  Results tree (under $BASE_RESULTS_DIR/):"
    for routing in "${ROUTING_MODES[@]}"; do
        echo "    $routing/"
        for workload in "${WORKLOADS[@]}"; do
            echo "      $workload/  → results.jsonl, summary.txt, metrics_*.jsonl"
        done
    done
fi
echo "========================================================"
