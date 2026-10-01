#!/usr/bin/env bash
cd "$(dirname "$0")/.."
until grep -q "QUEUE DONE" /tmp/queue2.log 2>/dev/null; do sleep 30; done
.venv312/Scripts/python.exe -u experiments/linearprobe_full_scale_both_metrics.py 2>&1 | grep --line-buffered -v -i -E "warning|Token indices|it/s\]"
echo "=== QUEUE2 DONE ==="
