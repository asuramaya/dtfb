#!/bin/bash
# Daily inventory sync — pulls new vehicles, skips already-scraped ones,
# flags delisted ones. Meant to run from cron.
#
# Configuration via environment variables (see README.md):
#   DTFB_REPO_DIR          — path to the dtfb checkout
#   DTFB_INVENTORY_URL     — dealer inventory listing URL
#   DTFB_LISTINGS_ROOT     — where to store output listings
#   DTFB_OLLAMA_BIN        — path to ollama binary (optional, seat-vision)
#   DTFB_OLLAMA_MODELS     — ollama models directory (optional)
#
# Export DTFB_DEALER_* variables or DTFB_CONFIG to set dealer info.
set -euo pipefail

REPO_DIR="${DTFB_REPO_DIR:-/home/asuramaya/code/dealer-to-fb}"
OLLAMA_BIN="${DTFB_OLLAMA_BIN:-/home/asuramaya/.local/ollama/bin/ollama}"
LISTINGS_ROOT="${DTFB_LISTINGS_ROOT:-/home/asuramaya/Documents/listings}"
INVENTORY_URL="${DTFB_INVENTORY_URL:-https://www.tomballford.com/inventory/all-vehicles/}"
LOG_DIR="${DTFB_LOG_DIR:-/home/asuramaya/.local/sync-logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/sync-$(date +%Y-%m-%d_%H%M%S).log"

if [ -n "${DTFB_OLLAMA_MODELS:-}" ]; then
    export OLLAMA_MODELS="$DTFB_OLLAMA_MODELS"
fi

# Start ollama if configured and not already running
if [ -n "${DTFB_OLLAMA_BIN:-}" ]; then
    if ! curl -s --max-time 2 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
        nohup "$OLLAMA_BIN" serve >>"$LOG_DIR/ollama-serve.log" 2>&1 &
        disown
        sleep 5
    fi
fi

cd "$REPO_DIR"
source .venv/bin/activate

python3 dtfb.py "$INVENTORY_URL" \
    --out "$LISTINGS_ROOT" \
    --sync --vision-seat-check \
    >>"$LOG_FILE" 2>&1
STATUS=$?

# Keep the last 30 days of logs
find "$LOG_DIR" -name "sync-*.log" -mtime +30 -delete

exit $STATUS

