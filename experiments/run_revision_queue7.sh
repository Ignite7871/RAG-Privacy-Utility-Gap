#!/usr/bin/env bash
# MPNet variants at batch size 8 with an automatic, ordered fallback (data and epochs are never reduced):
#   1) float32, batch 8 (scale gpu10k_b8): v2 first, then v1.
#   2) if v2 does not fit: bf16 autocast, batch 8 (scale gpu10k_b8_bf16), BOTH variants so the comparison stays like for like.
#   3) if that also fails: nothing more is run; the paper reports MPNet v1 (batch 16) only.
# Gradient accumulation is deliberately not used: the InfoNCE loss uses in-batch negatives, so it is not equivalent.
cd "$(dirname "$0")/.."
until grep -q "QUEUE6 DONE" /tmp/queue6.log 2>/dev/null; do sleep 30; done
PY=.venv312/Scripts/python.exe
F='warning|it/s\]'
train() { # variant scale
  [ -f checkpoints/advenc_${1}_${2}_mpnet.pt ] && return 0
  echo "=== train $1 $2 mpnet $(date +%H:%M:%S) ==="
  $PY -u defenses/adv_encoder.py --variant $1 --scale $2 --encoder mpnet --progress-every 200 2>&1 | grep --line-buffered -v -i -E "$F" | tail -n 3
  [ -f checkpoints/advenc_${1}_${2}_mpnet.pt ]
}
if train v2 gpu10k_b8; then
  train v1 gpu10k_b8
else
  echo "=== v2 float32 batch 8 did not fit; falling back to bf16 autocast for both variants $(date +%H:%M:%S) ==="
  train v2 gpu10k_b8_bf16 && train v1 gpu10k_b8_bf16
fi
echo "=== mpnet eval $(date +%H:%M:%S) ==="
$PY -u experiments/advenc_reduced_scale_eval.py 2>&1 | grep --line-buffered -v -i -E "$F"
echo "=== QUEUE7 DONE $(date +%H:%M:%S) ==="
