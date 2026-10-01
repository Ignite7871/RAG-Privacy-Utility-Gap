#!/usr/bin/env bash
# Ablation: separate pair type from lambda_priv (MiniLM, gpu10k): v1s = query-passage pairs with lambda_priv 2.0,
# v2w = self-pairs with lambda_priv 0.3. Then rerun the (resumable) reduced-scale evaluation to include them.
cd "$(dirname "$0")/.."
until grep -q "QUEUE5 DONE" /tmp/queue5.log 2>/dev/null; do sleep 30; done
PY=.venv312/Scripts/python.exe
F='warning|it/s\]'
for v in v1s v2w; do
  if [ ! -f checkpoints/advenc_${v}_gpu10k_minilm.pt ]; then
    echo "=== train $v gpu10k minilm $(date +%H:%M:%S) ==="
    $PY -u defenses/adv_encoder.py --variant $v --scale gpu10k --encoder minilm --progress-every 200 2>&1 | grep --line-buffered -v -i -E "$F" | tail -n 3
  fi
done
echo "=== ablation eval $(date +%H:%M:%S) ==="
$PY -u experiments/advenc_reduced_scale_eval.py 2>&1 | grep --line-buffered -v -i -E "$F"
echo "=== QUEUE6 DONE $(date +%H:%M:%S) ==="
