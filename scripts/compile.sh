#!/usr/bin/env bash
# axon compile <config.yaml>  ->  src/generated/nrf_axon_model_<model_name>_.h
# Defaults to cnn_mel.yaml for backwards compatibility.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/.." && pwd)"
axon="$HOME/ncs/edge-ai/tools/axon/compiler"

config="${1:-cnn_mel.yaml}"

[ -d "$axon" ] || { echo "no axon compiler at $axon" >&2; exit 1; }
[ -d "$root/.venv" ] || { echo "no .venv, run uv sync at repo root" >&2; exit 1; }
[ -f "$here/$config" ] || { echo "no config $here/$config" >&2; exit 1; }

# model_name is declared inside the yaml; extract for header lookup / install.
model_name=$(awk '/^[[:space:]]*model_name:/ {gsub(/^[[:space:]]*model_name:[[:space:]]*/,""); gsub(/[" ]/,""); print; exit}' "$here/$config")
[ -n "$model_name" ] || { echo "couldn't parse model_name from $config" >&2; exit 1; }

export COMPILER_ROOT_FOLDER="$axon/bin"
export PYTHONPATH="$axon/scripts:$axon/scripts/utility:${PYTHONPATH:-}"

cd "$here"
"$root/.venv/bin/python" "$axon/scripts/axons_ml_nn_compiler_executor.py" "$config"

generated="nrf_axon_model_${model_name}_.h"
h=""
for c in "outputs/${generated}" "outputs/${model_name}/${generated}"; do
    [ -f "$c" ] && h="$c" && break
done
[ -n "$h" ] || { echo "no ${generated} found, check scripts/outputs/" >&2; exit 1; }

install -D "$h" "$root/src/nrf/generated/${generated}"
echo "ok -> src/nrf/generated/${generated}"
