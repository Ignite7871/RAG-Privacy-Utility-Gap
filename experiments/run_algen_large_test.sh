#!/usr/bin/env bash
# ALGEN with n_test=500 (the earlier sweep used n_test=50) for MS MARCO / NQ x 4 encoders x 4 budgets.
# Runs under .venv-algen-legacy (transformers==4.52.4); see CLAUDE.md dual-venv note.
# Optional first arg: space-separated "dataset:encoder" pairs (default: the 8 clean pairs).
cd "$(dirname "$0")/.."
PY=.venv-algen-legacy/Scripts/python.exe
PAIRS=${1:-"msmarco:minilm msmarco:mpnet msmarco:gtr msmarco:bge nq:minilm nq:mpnet nq:gtr nq:bge"}
NALIGN=${2:-"50 200 1000 2000"}
OUT=${3:-results/algen_n500}
for pair in $PAIRS; do
  ds=${pair%%:*}; enc=${pair##*:}
  for n in $NALIGN; do
    [ -f "$OUT/${ds}_${enc}_n${n}.json" ] && continue
    $PY attackers/run_algen_legacy.py --dataset "$ds" --encoder "$enc" --n_align "$n" --n_test 500 --out_dir "$OUT" 2>&1 | grep -E "ROUGE-L|wrote"
  done
done
