#!/usr/bin/env bash
# Run the Axon NPU compiler on cnn_mel.yaml and stage the generated
# header into src/generated/ where main.c #includes it from.
#
# Prereqs: workspace venv at the repo root with the deps in pyproject.toml
# (tensorflow==2.19.0, tflite, pyyaml, cffi, scikit-learn).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPILER_DIR="$HOME/ncs/edge-ai/tools/axon/compiler"
COMPILER_SCRIPTS="$COMPILER_DIR/scripts"

if [ ! -d "$COMPILER_DIR" ]; then
    echo "missing: $COMPILER_DIR (Edge AI add-on not installed?)" >&2
    exit 1
fi
if [ ! -d "$REPO_ROOT/.venv" ]; then
    echo "missing: $REPO_ROOT/.venv (run 'uv sync' at the repo root)" >&2
    exit 1
fi

# Compiler scripts import each other by bare name and ctypes-load the .so
# from $COMPILER_ROOT_FOLDER/Linux. Match the Dockerfile's env exactly.
export COMPILER_ROOT_FOLDER="$COMPILER_DIR/bin"
export PYTHONPATH="$COMPILER_SCRIPTS:$COMPILER_SCRIPTS/utility:${PYTHONPATH:-}"

cd "$SCRIPT_DIR"
"$REPO_ROOT/.venv/bin/python" "$COMPILER_SCRIPTS/axons_ml_nn_compiler_executor.py" cnn_mel.yaml

# The executor writes outputs into ./outputs/ relative to the YAML.
GENERATED=""
for cand in outputs/nrf_axon_model_cnn_mel_.h \
            outputs/cnn_mel/nrf_axon_model_cnn_mel_.h; do
    if [ -f "$cand" ]; then GENERATED="$cand"; break; fi
done
if [ -z "$GENERATED" ]; then
    echo "compile finished but couldn't locate nrf_axon_model_cnn_mel_.h" >&2
    find outputs -name 'nrf_axon_model_*.h' 2>/dev/null >&2
    exit 1
fi

install -D "$GENERATED" "$REPO_ROOT/src/generated/nrf_axon_model_cnn_mel_.h"
echo "staged -> src/generated/nrf_axon_model_cnn_mel_.h"
