#!/usr/bin/env bash
# Launch a long-running command in background with PID + log tracking.
# Usage: run_long.sh <name> "<command>"
#
# Example:
#   run_long.sh ml-train "python -m polymarket_bot.ml.train --folds 5"
#   tail -f runs/_async/ml-train.log
#   kill $(cat runs/_async/ml-train.pid)

set -euo pipefail
NAME="${1:?usage: run_long.sh <name> <command>}"
shift
LOG_DIR="/root/polymarket-bot-ml-backtest/runs/_async"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/${NAME}.log"
PID_FILE="${LOG_DIR}/${NAME}.pid"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
    echo "ALREADY RUNNING: name=$NAME pid=$(cat $PID_FILE) log=$LOG" >&2
    exit 1
fi

cd /root/trading
nohup bash -c "$*" > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"
echo "ASYNC started: name=$NAME pid=$PID log=$LOG"
