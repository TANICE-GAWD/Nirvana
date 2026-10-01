#!/usr/bin/env bash
# CPU-only setup. No GPU needed.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d third_party/Hush ] || git clone --depth 1 https://github.com/pulp-vision/Hush third_party/Hush
[ -d .venv ] || uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install --index-strategy unsafe-best-match -r requirements.txt
