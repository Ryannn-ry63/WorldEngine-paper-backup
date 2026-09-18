#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
PROBE_CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$PROBE_CODE_DIR/../../../miniconda3/envs/algengine/bin/python" "$PROBE_CODE_DIR/migration/owned_resume_probe.py" run --device "${1:-0}"
