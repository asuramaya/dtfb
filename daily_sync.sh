#!/bin/bash
# Daily inventory sync — pulls new vehicles, skips already-scraped ones,
# flags delisted ones. Meant to run from cron.
#
# Uses inventory-sync (not `lotstretcher --sync`) -- the purpose-built cron
# entry point: async hero-video rendering overlapped with the next
# vehicle's scrape/CV work, and it self-heals a busy GPU (falls back to
# software encoding rather than losing a video, see render_hero_video()).
#
# Configuration via environment variables (see README.md):
#   LOTSTRETCHER_REPO_DIR          — path to the lotstretcher checkout
#   LOTSTRETCHER_SCOPE             — named inventory scope (see dealer-config.json's
#                             inventory_urls / LOTSTRETCHER_INVENTORY_URL_<SCOPE>),
#                             e.g. "used". Ignored if LOTSTRETCHER_INVENTORY_URL is set.
#   LOTSTRETCHER_INVENTORY_URL     — dealer inventory listing URL (overrides LOTSTRETCHER_SCOPE)
#   LOTSTRETCHER_LISTINGS_ROOT     — where to store output listings
#   LOTSTRETCHER_OLLAMA_BIN        — path to ollama binary (optional, seat-vision)
#   LOTSTRETCHER_OLLAMA_MODELS     — ollama models directory (optional)
#
# Export LOTSTRETCHER_DEALER_* variables or LOTSTRETCHER_CONFIG to set dealer info.
set -euo pipefail

REPO_DIR="${LOTSTRETCHER_REPO_DIR:-/home/asuramaya/code/lotstretcher}"
OLLAMA_BIN="${LOTSTRETCHER_OLLAMA_BIN:-/home/asuramaya/.local/ollama/bin/ollama}"
LISTINGS_ROOT="${LOTSTRETCHER_LISTINGS_ROOT:-/home/asuramaya/Documents/listings}"
# Default scope is "used", not "all" -- the full-inventory crawl is
# currently paused pending an operator-side network fix (repeated
# ERR_NETWORK_CHANGED/timeouts on this machine cause partial crawls at
# that scale); used-vehicles-only completes cleanly every time. Set
# LOTSTRETCHER_INVENTORY_URL directly to bypass scope resolution entirely.
SCOPE="${LOTSTRETCHER_SCOPE:-used}"
LOG_DIR="${LOTSTRETCHER_LOG_DIR:-/home/asuramaya/.local/sync-logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/sync-$(date +%Y-%m-%d_%H%M%S).log"

if [ -n "${LOTSTRETCHER_OLLAMA_MODELS:-}" ]; then
    export OLLAMA_MODELS="$LOTSTRETCHER_OLLAMA_MODELS"
fi

# Start ollama if configured and not already running
if [ -n "${LOTSTRETCHER_OLLAMA_BIN:-}" ]; then
    if ! curl -s --max-time 2 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
        nohup "$OLLAMA_BIN" serve >>"$LOG_DIR/ollama-serve.log" 2>&1 &
        disown
        sleep 5
    fi
fi

cd "$REPO_DIR"
source .venv/bin/activate

if [ -n "${LOTSTRETCHER_INVENTORY_URL:-}" ]; then
    inventory-sync --inventory-url "$LOTSTRETCHER_INVENTORY_URL" \
        --out "$LISTINGS_ROOT" --vision-seat-check \
        >>"$LOG_FILE" 2>&1
else
    inventory-sync --scope "$SCOPE" \
        --out "$LISTINGS_ROOT" --vision-seat-check \
        >>"$LOG_FILE" 2>&1
fi
STATUS=$?

# Keep the last 30 days of logs
find "$LOG_DIR" -name "sync-*.log" -mtime +30 -delete

exit $STATUS
