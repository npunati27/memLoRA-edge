#!/usr/bin/env bash
# Minimal setup for CPU-only mock nodes (no GPU, no vLLM, no model download).
# Run from the memLoRA-edge repo after clone. Same peer args as scripts/setup.sh.
set -euo pipefail

usage() {
    echo "Usage: $0 NODE_IDX IP0 [IP1 ...]" >&2
    echo "  Writes ~/peers.json and installs a venv with API-only Python deps." >&2
    echo "Example (this machine is the second node): $0 1 10.0.0.1 10.0.0.2 10.0.0.3" >&2
    exit 1
}

SHOULD_SUDO=true
case "${MEMLORA_USE_SUDO:-yes}" in 0|no|false|off) SHOULD_SUDO=false ;;
    1|yes|true|on)  SHOULD_SUDO=true ;;
esac

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-sudo) SHOULD_SUDO=false; shift ;;
        --sudo)    SHOULD_SUDO=true; shift ;;
        -h|--help) usage ;;
        *) break ;;
    esac
done

if [[ "$SHOULD_SUDO" == true ]]; then SUDO=(sudo)
else SUDO=()
fi

[[ $# -lt 2 ]] && usage

NODE_IDX=$1
shift
NODE_IPS=("$@")
NUM_NODES=${#NODE_IPS[@]}

if ! [[ "$NODE_IDX" =~ ^[0-9]+$ ]] || (( NODE_IDX < 0 || NODE_IDX >= NUM_NODES )); then
    echo "Error: NODE_IDX must be in [0, $((NUM_NODES - 1))] (got '$NODE_IDX')" >&2
    exit 1
fi

MY_IP="${NODE_IPS[$NODE_IDX]}"

echo "==> Mock VM setup: node $NODE_IDX / $NUM_NODES (my_ip=$MY_IP)"

install_base_packages() {
    if [[ ! -f /etc/os-release ]]; then
        echo "No /etc/os-release; install python3, pip, and git yourself, then re-run." >&2
        exit 1
    fi
    # shellcheck source=/dev/null
    . /etc/os-release
    local rh=0
    case " ${ID_LIKE:-} " in
        *" rhel "*|*" fedora "*|*" centos "*) rh=1 ;;
    esac
    case "${ID:-}" in
        rhel|fedora|centos|rocky|almalinux|ol|amzn|scientific) rh=1 ;;
    esac
    if [[ "$rh" -eq 1 ]]; then
        echo "==> Detected Red Hat family (ID=$ID); using dnf or yum"
        if command -v dnf >/dev/null 2>&1; then
            "${SUDO[@]}" dnf install -y python3 python3-pip git
        else
            "${SUDO[@]}" yum install -y python3 python3-pip git
        fi
    elif [[ "${ID:-}" == "debian" || "${ID:-}" == "ubuntu" ]] \
        || [[ " ${ID_LIKE:-} " == *" debian "* ]]; then
        echo "==> Detected Debian family (ID=$ID); using apt"
        "${SUDO[@]}" apt-get update -qq
        "${SUDO[@]}" apt-get install -y python3-pip python3-venv git -qq
    else
        echo "Unsupported OS ID='${ID:-}' ID_LIKE='${ID_LIKE:-}'. Install python3, python3-pip, git; then re-run." >&2
        exit 1
    fi
}

install_base_packages

if [[ ! -d ~/venv ]]; then
    python3 -m venv ~/venv
fi
# shellcheck source=/dev/null
source ~/venv/bin/activate

grep -qxF 'source ~/venv/bin/activate' ~/.bashrc 2>/dev/null \
    || echo 'source ~/venv/bin/activate' >> ~/.bashrc

pip install -q --upgrade pip
pip install -q \
    "fastapi>=0.115" \
    "uvicorn[standard]>=0.30" \
    "aiohttp>=3.9" \
    --no-cache-dir

mkdir -p ~/logs ~/adapters

PEERS_JSON=$(printf '"%s",' "${NODE_IPS[@]}")
PEERS_JSON="[${PEERS_JSON%,}]"

cat > ~/peers.json << PEERS
{
    "my_ip":    "$MY_IP",
    "node_idx": $NODE_IDX,
    "peers":    $PEERS_JSON
}
PEERS

echo "==> Done. ~/peers.json and ~/venv ready."
echo "    Next: cd to repo root, then:"
echo "      source ~/venv/bin/activate"
echo "      export MEMLORA_MOCK=1"
echo "      python -m scripts.deploy"
