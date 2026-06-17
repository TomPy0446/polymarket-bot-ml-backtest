#!/usr/bin/env bash
# Check status of an async-launched command.
# Usage: check_async.sh <name>

set -euo pipefail
NAME="${1:?usage: check_async.sh <name>}"
LOG_DIR="/root/polymarket-bot-ml-backtest/runs/_async"
LOG="${LOG_DIR}/${NAME}.log"
PID_FILE="${LOG_DIR}/${NAME}.pid"

if [[ ! -f "$PID_FILE" ]]; then
    echo "NO PID FILE for name=$NAME (never launched or already cleaned up)"
    exit 0
fi

PID=$(cat "$PID_FILE")
if kill -0 "$PID" 2>/dev/null; then
    echo "RUNNING: name=$NAME pid=$PID"
    echo "--- last 20 lines of $LOG ---"
    tail -20 "$LOG" || true
else
    echo "FINISHED (or dead): name=$NAME pid=$PID"
    echo "--- last 30 lines of $LOG ---"
    tail -30 "$LOG" || true
fi
