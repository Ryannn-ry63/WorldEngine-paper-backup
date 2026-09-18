#!/usr/bin/env bash
set -euo pipefail
PAPER_DEBUG_CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$PAPER_DEBUG_CODE/run_paper_study.sh" debug-six --devices 0,1,2,3 "$@"
