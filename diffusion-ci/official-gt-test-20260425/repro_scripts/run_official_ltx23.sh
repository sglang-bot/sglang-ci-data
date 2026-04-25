#!/usr/bin/env bash
set -euo pipefail

# Set HF_TOKEN in the environment before running. Do not hardcode it here.
# Reproduces LTX-2.3 official-repo video key frames with the same CI-sized
# sampling params. Verified on a single H200 with fp8-cast + inference_mode.
OUT_DIR=${1:-/tmp/mmgen-official-ltx23-report}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH=/tmp/mmgen-official-code/LTX-2/packages/ltx-core/src:/tmp/mmgen-official-code/LTX-2/packages/ltx-pipelines/src:/sgl-workspace/sglang/python:$PYTHONPATH
cd /sgl-workspace/sglang/python
python3 "$SCRIPT_DIR/gen_official_ltx23.py" --out-dir "$OUT_DIR"
