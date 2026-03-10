#!/usr/bin/env bash
# Run transformers compatibility tests and file issues for failures.
# Designed to be called by AI agents (Claude Code, Codex CLI, etc.)
#
# Usage:
#   ./compat/transformers/test-and-report.sh bert          # test one model
#   ./compat/transformers/test-and-report.sh --tier 1      # test all tier-1
#   ./compat/transformers/test-and-report.sh               # default: tier 1
#
# Prerequisites:
#   pip install -e . && pip install -r compat/transformers/requirements.txt
#
# Output:
#   - JSON report: compat/transformers/_reports/latest.json
#   - Human summary: stdout

set -euo pipefail
cd "$(dirname "$0")/../.."

REPORT_DIR="compat/transformers/_reports"
mkdir -p "$REPORT_DIR"
REPORT_FILE="$REPORT_DIR/latest.json"

# Parse args
ARGS=""
if [ $# -eq 0 ]; then
    ARGS="--tier 1"
elif [[ "$1" != --* ]]; then
    ARGS="--model $1"
    shift
    ARGS="$ARGS $*"
else
    ARGS="$*"
fi

echo "=== Candle Transformers Compat Test ==="
echo "Args: $ARGS"
echo ""

# Run tests
USE_CANDLE=1 python compat/transformers/run.py \
    $ARGS \
    --json-report "$REPORT_FILE" \
    -v --tb=short || true

# Summarize
echo ""
python compat/transformers/run.py --summarize "$REPORT_FILE"
