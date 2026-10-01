#!/usr/bin/env bash
# Sequential queue of the revision experiments. Each script is resumable.
# Waits for the Vec2Text run to finish first so the GPU is not shared.
cd "$(dirname "$0")/.."
until [ -f results/vec2text/summary_steps20_beam4.json ]; do sleep 20; done
PY=.venv312/Scripts/python.exe
for s in advenc_adaptive_budget_sweep dp_adaptive_sweep advenc_retrieval_all_variants cross_domain_alignment; do
  echo "=== $s $(date +%H:%M:%S) ==="
  $PY -u experiments/$s.py 2>&1 | grep --line-buffered -v -i -E "warning|Token indices|it/s\]"
done
echo "=== ALGEN n_test=500 $(date +%H:%M:%S) ==="
bash experiments/run_algen_large_test.sh
echo "=== QUEUE DONE $(date +%H:%M:%S) ==="
