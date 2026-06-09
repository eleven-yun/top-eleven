#!/usr/bin/env bash
# run_acceptance_dual_profile.sh — operational + baseline acceptance checks
# Usage: bash scripts/run_acceptance_dual_profile.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

mkdir -p output/backtest

echo "=== Phase 9 Acceptance (Dual Profile) ==="
echo ""

echo "[1/2] Operational gate (blocking): EV>=0.09, conf>=0.45"
conda run -n top-eleven python scripts/generate_acceptance_report.py \
  --ev-threshold 0.09 \
  --min-confidence 0.45 \
  --acceptance-mode supported_universe \
  --acceptance-target 70 \
  --fail-on-overall-fail \
  --output output/backtest/acceptance_report_operational.json

echo ""
echo "[2/2] Baseline monitor (non-blocking): EV>=0.05, conf>=0.55"
set +e
conda run -n top-eleven python scripts/generate_acceptance_report.py \
  --ev-threshold 0.05 \
  --min-confidence 0.55 \
  --acceptance-mode supported_universe \
  --acceptance-target 70 \
  --fail-on-overall-fail \
  --output output/backtest/acceptance_report_baseline.json
baseline_exit=$?
set -e

echo ""
if [[ ${baseline_exit} -eq 0 ]]; then
  echo "Baseline monitor: PASS"
else
  echo "Baseline monitor: FAIL (expected while scope expansion is in progress)"
fi

echo ""
echo "Artifacts:"
echo "  output/backtest/acceptance_report_operational.json"
echo "  output/backtest/acceptance_report_baseline.json"

echo ""
echo "Dual-profile run completed."