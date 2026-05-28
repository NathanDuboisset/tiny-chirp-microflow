#!/usr/bin/env bash
# axon compile -> src/generated/nrf_axon_model_cnn_mel_.h
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/.." && pwd)"
axon="$HOME/ncs/edge-ai/tools/axon/compiler"

[ -d "$axon" ] || { echo "no axon compiler at $axon" >&2; exit 1; }
[ -d "$root/.venv" ] || { echo "no .venv, run uv sync at repo root" >&2; exit 1; }

export COMPILER_ROOT_FOLDER="$axon/bin"
export PYTHONPATH="$axon/scripts:$axon/scripts/utility:${PYTHONPATH:-}"

cd "$here"
"$root/.venv/bin/python" "$axon/scripts/axons_ml_nn_compiler_executor.py" cnn_mel.yaml

h=""
for c in outputs/nrf_axon_model_cnn_mel_.h outputs/cnn_mel/nrf_axon_model_cnn_mel_.h; do
    [ -f "$c" ] && h="$c" && break
done
[ -n "$h" ] || { echo "no generated .h found, check scripts/outputs/" >&2; exit 1; }

install -D "$h" "$root/src/generated/nrf_axon_model_cnn_mel_.h"
echo "ok -> src/generated/nrf_axon_model_cnn_mel_.h"
