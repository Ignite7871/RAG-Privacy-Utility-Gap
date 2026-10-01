"""Aggregate ALGEN (n_test=500) runs with both metrics from the saved reconstructions.
-> results/algen_n500/summary.csv"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from attackers.metrics import full_scores
rows = []
DIR = REPO_ROOT / "results" / (sys.argv[1] if len(sys.argv) > 1 else "algen_n500")
for f in sorted(DIR.glob("*_n*.json")):
    d = json.load(open(f, encoding="utf-8"))
    rec = d["reconstructions"]
    r = full_scores([x["reconstruction"] for x in rec], [x["source"] for x in rec])
    rows.append({"dataset": d["dataset"], "encoder": d["encoder"], "n_align": d["n_align"], "n_test": d["n_test"], **r})
with (DIR / "summary.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows: print(r["dataset"], r["encoder"], r["n_align"], "P=%.3f C=%.3f Crec=%.3f" % (r["rouge_l_precision"], r["content_precision"], r["content_recall"]))
