#!/usr/bin/env bash
# Run PyTorch official tests against candle and file issues for failures.
# Designed to be called by AI agents (Claude Code, Codex CLI, etc.)
#
# Usage:
#   ./compat/pytorch/test-and-report.sh test_tensor.py  # single file
#   ./compat/pytorch/test-and-report.sh --tier mechanism:1  # tier
#   ./compat/pytorch/test-and-report.sh                   # default: tier mechanism:1
#
# Prerequisites:
#   pip install -e . && pip install -r compat/pytorch/requirements.txt

set -euo pipefail
cd "$(dirname "$0")/../.."

REPORT_DIR="compat/pytorch/_reports"
mkdir -p "$REPORT_DIR"
REPORT_FILE="$REPORT_DIR/latest.json"

# Parse args
ARGS=""
if [ $# -eq 0 ]; then
    ARGS="--tier mechanism:1"
elif [[ "$1" != --* ]]; then
    ARGS="--file $1"
    shift
    ARGS="$ARGS $*"
else
    ARGS="$*"
fi

echo "=== Candle PyTorch Compat Test ==="
echo "Args: $ARGS"
echo ""

# Run tests
USE_CANDLE=1 python compat/pytorch/run.py \
    $ARGS \
    --json-report "$REPORT_FILE" \
    -v --tb=short || true

# Summarize
echo ""
python compat/pytorch/run.py --summarize "$REPORT_FILE"
