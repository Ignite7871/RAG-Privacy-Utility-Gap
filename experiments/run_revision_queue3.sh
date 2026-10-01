#!/usr/bin/env bash
# (1) DP grid extension (eps 25/30/40; retrieval, non-adaptive, adaptive), (2) AdvEnc-v1 at GPU scale + evaluation.
cd "$(dirname "$0")/.."
until grep -q "QUEUE2 DONE" /tmp/queue3.log 2>/dev/null; do sleep 30; done
PY=.venv312/Scripts/python.exe
F='warning|it/s\]'
echo "=== dp_defense_sweep $(date +%H:%M:%S) ==="
$PY -u experiments/dp_defense_sweep.py 2>&1 | grep --line-buffered -v -i -E "$F"
echo "=== dp_adaptive_sweep $(date +%H:%M:%S) ==="
$PY -u experiments/dp_adaptive_sweep.py 2>&1 | grep --line-buffered -v -i -E "$F"
echo "=== DP DONE $(date +%H:%M:%S) ==="
echo "=== train v1 gpu50k $(date +%H:%M:%S) ==="
$PY -u defenses/adv_encoder.py --variant v1 --scale gpu50k --encoder minilm --progress-every 200 2>&1 | grep --line-buffered -v -i -E "$F"
echo "=== sweep $(date +%H:%M:%S) ==="
$PY -u experiments/advenc_adaptive_budget_sweep.py 2>&1 | grep --line-buffered -v -i -E "$F"
echo "=== retrieval $(date +%H:%M:%S) ==="
$PY -u experiments/advenc_retrieval_all_variants.py 2>&1 | grep --line-buffered -v -i -E "$F"
echo "=== nonadaptive $(date +%H:%M:%S) ==="
$PY -u experiments/advenc_nonadaptive_both_metrics.py 2>&1 | grep --line-buffered -v -i -E "$F"
echo "=== QUEUE4 DONE $(date +%H:%M:%S) ==="
