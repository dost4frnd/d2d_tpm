#!/usr/bin/env bash
# Convenience wrapper: set a power mode and lock clocks on a Jetson Orin Nano.
# Usage: sudo bash deployment/power/set_power_mode.sh [7w|15w]
set -euo pipefail
MODE="${1:-7w}"
case "$MODE" in
  7w)  ID=1 ;;
  15w) ID=0 ;;
  *) echo "usage: $0 [7w|15w]"; exit 1 ;;
esac
if ! command -v nvpmodel >/dev/null 2>&1; then
  echo "nvpmodel not found — are you on a Jetson?"; exit 1
fi
nvpmodel -m "$ID"
jetson_clocks
echo "Set power mode $MODE (nvpmodel id $ID) and locked clocks."
nvpmodel -q
