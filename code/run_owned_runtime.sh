#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
OWNED_CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$OWNED_CODE_DIR/../../../miniconda3/envs/algengine/bin/python" "$OWNED_CODE_DIR/migration/owned_runtime.py" "$@"
