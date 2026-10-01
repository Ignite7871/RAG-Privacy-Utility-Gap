#!/usr/bin/env bash
# Second-encoder check: AdvEnc at reduced scale (gpu10k) on MiniLM-L6 and MPNet, then evaluation.
cd "$(dirname "$0")/.."
until grep -q "QUEUE4 DONE" /tmp/queue4.log 2>/dev/null; do sleep 30; done
PY=.venv312/Scripts/python.exe
F='warning|it/s\]'
for enc in minilm mpnet; do
  for v in v2 v1; do
    if [ ! -f checkpoints/advenc_${v}_gpu10k_${enc}.pt ]; then
      echo "=== train $v gpu10k $enc $(date +%H:%M:%S) ==="
      $PY -u defenses/adv_encoder.py --variant $v --scale gpu10k --encoder $enc --progress-every 200 2>&1 | grep --line-buffered -v -i -E "$F" | tail -n 3
    fi
  done
done
echo "=== reduced-scale eval $(date +%H:%M:%S) ==="
$PY -u experiments/advenc_reduced_scale_eval.py 2>&1 | grep --line-buffered -v -i -E "$F"
echo "=== QUEUE5 DONE $(date +%H:%M:%S) ==="
