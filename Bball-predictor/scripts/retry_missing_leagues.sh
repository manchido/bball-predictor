#!/usr/bin/env bash
# Retry scraping for rate-limited leagues (HUNG, ABA 2024-25/2025-26, NBA 2024-25/2025-26)
# Run SERIALLY — one league at a time to avoid SofaScore re-triggering rate limit.
# Each league gets a 120s sleep gap between runs.
#
# Usage:
#   source .venv/bin/activate
#   bash scripts/retry_missing_leagues.sh
#
# Estimated time: ~2-4 hours (mostly waiting on request delays)

set -e
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"

echo "=== Retry Missing Leagues ==="
echo "Start: $(date)"
echo ""

# ----- HUNG (all 3 seasons: 0 files) -----
echo "--- [1/5] HUNG all seasons ---"
$PYTHON scripts/scrape_national_leagues.py \
    --league hung \
    --no-build-gold
echo "HUNG done. Sleeping 120s..."
sleep 120

# ----- ABA 2024-25 (only 6/261 files) -----
echo "--- [2/5] ABA 2024-25 ---"
$PYTHON scripts/scrape_national_leagues.py \
    --league aba \
    --season ABA2024 \
    --no-build-gold
echo "ABA 2024-25 done. Sleeping 120s..."
sleep 120

# ----- ABA 2025-26 (0 files) -----
echo "--- [3/5] ABA 2025-26 ---"
$PYTHON scripts/scrape_national_leagues.py \
    --league aba \
    --season ABA2025 \
    --no-build-gold
echo "ABA 2025-26 done. Sleeping 120s..."
sleep 120

# ----- NBA 2024-25 (~200 missing games) -----
echo "--- [4/5] NBA 2024-25 ---"
$PYTHON scripts/scrape_national_leagues.py \
    --league nba \
    --season NBA2024 \
    --no-build-gold
echo "NBA 2024-25 done. Sleeping 120s..."
sleep 120

# ----- NBA 2025-26 (~775 missing games) -----
echo "--- [5/5] NBA 2025-26 ---"
$PYTHON scripts/scrape_national_leagues.py \
    --league nba \
    --season NBA2025 \
    --no-build-gold
echo "NBA 2025-26 done."
echo ""

# ----- Rebuild gold + retrain -----
echo "=== Rebuilding gold table ==="
$PYTHON cli.py build-gold

echo "=== Retraining model ==="
$PYTHON cli.py train

echo ""
echo "=== Done: $(date) ==="
