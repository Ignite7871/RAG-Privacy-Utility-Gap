"""Recompute the derived quantities (ratios, percentages, ranges) quoted in the manuscript from the result files.

Each line prints the recomputed value next to the value written in the paper and flags any mismatch after rounding.
Usage: python experiments/audit_derived.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"
bad = 0


def check(name, value, quoted, nd=0):
    global bad
    ok = round(value, nd) == round(quoted, nd)
    bad += 0 if ok else 1
    print(f"{'OK ' if ok else 'BAD'} {name}: recomputed {value:.4f} vs quoted {quoted}")


def rows(p):
    return list(csv.DictReader(open(R / p, newline="", encoding="utf-8")))


# --- AdvEnc sweep means
g = defaultdict(list)
for r in rows("advenc/adaptive_budget_sweep.csv"):
    if r["attacker"] == "linearprobe":
        g[(r["encoder"], int(r["n_align"]))].append((float(r["content_precision"]), float(r["rouge_l_precision"])))
mean = lambda e, n, i=0: sum(x[i] for x in g[(e, n)]) / len(g[(e, n)])
van10 = mean("vanilla", 10000)
check("vanilla content precision n=10k", van10, 0.471, 3)
check("content precision 0.144 -> 0.471 saturation: n=200", mean("vanilla", 200), 0.144, 3)
check("vanilla n=40k content", mean("vanilla", 40000), 0.469, 3)
check("vanilla plain n=10k", mean("vanilla", 10000, 1), 0.404, 3)
check("vanilla plain n=40k", mean("vanilla", 40000, 1), 0.421, 3)
check("n=5000/384 = 13d", 5000 / 384, 13.0, 0)
check("n=10000/384 = 26d", 10000 / 384, 26.0, 0)

# --- DP: percentages of undefended (n=10k) and Recall@5 retention
dp = {r["epsilon"]: r for r in rows("dp_defense/dp_sweep_full.csv")}
ad = {(r["epsilon"], r["n_align"]): r for r in rows("dp_defense/dp_adaptive.csv")}
rec_inf = float(dp["inf"]["recall_at_5"])
for eps, pct_of_und, ret in (("50", 47, 81), ("100", 83, 94), ("200", 95, None), ("40", 38, 69)):
    c = float(ad[(eps, "10000")]["content_precision"])
    check(f"eps={eps} adaptive content as % of undefended", 100 * c / van10, pct_of_und)
    if ret:
        check(f"eps={eps} Recall@5 retention %", 100 * float(dp[eps]["recall_at_5"]) / rec_inf, ret)
check("eps=40 reduction %", 100 * (1 - float(ad[("40", "10000")]["content_precision"]) / van10), 62)
check("eps=50 reduction %", 100 * (1 - float(ad[("50", "10000")]["content_precision"]) / van10), 53)
check("eps=100 reduction %", 100 * (1 - float(ad[("100", "10000")]["content_precision"]) / van10), 17)

# --- cross-domain retention at n=5000
cd = defaultdict(list)
for r in rows("cross_domain/cross_domain.csv"):
    cd[(r["encoder"], r["source"], r["target"], int(r["n_align"]))].append(float(r["content_precision"]))
m = lambda *k: sum(cd[k]) / len(cd[k])
ms = [100 * m(e, "nq", "msmarco", 5000) / m(e, "msmarco", "msmarco", 5000) for e in ("minilm", "mpnet", "gtr", "bge")]
nq = [100 * m(e, "msmarco", "nq", 5000) / m(e, "nq", "nq", 5000) for e in ("minilm", "mpnet", "gtr", "bge")]
check("cross-domain MS MARCO victims min %", min(ms), 49)
check("cross-domain MS MARCO victims max %", max(ms), 67)
check("cross-domain NQ victims min %", min(nq), 65)
check("cross-domain NQ victims max %", max(nq), 73)

# --- LinearProbe full scale
lp = {(r["corpus"], r["encoder"]): r for r in rows("linearprobe_full_scale_both_metrics.csv")}
pf = {r["corpus"]: float(r["rouge_l_precision"]) for r in rows("prior_floor.csv") if r["n_align"] == "40000" and r["encoder"] == "minilm"}
check("MS MARCO plain range low over floor", float(lp[("msmarco", "minilm")]["rouge_l_precision"]) - pf["msmarco"], 0.035, 3)
check("MS MARCO plain range high over floor", float(lp[("msmarco", "gtr")]["rouge_l_precision"]) - pf["msmarco"], 0.051, 3)
check("GTR - MiniLM content, MS MARCO", float(lp[("msmarco", "gtr")]["content_precision"]) - float(lp[("msmarco", "minilm")]["content_precision"]), 0.055, 3)
check("BGE - MiniLM content, MS MARCO", float(lp[("msmarco", "bge")]["content_precision"]) - float(lp[("msmarco", "minilm")]["content_precision"]), 0.046, 3)
check("GTR - MiniLM content, NQ", float(lp[("nq", "gtr")]["content_precision"]) - float(lp[("nq", "minilm")]["content_precision"]), 0.035, 3)
check("BGE - MiniLM content, NQ", float(lp[("nq", "bge")]["content_precision"]) - float(lp[("nq", "minilm")]["content_precision"]), 0.015, 3)
vals = [float(lp[("msmarco", e)]["rouge_l_precision"]) for e in ("minilm", "mpnet", "gtr", "bge")]
check("plain-precision relative range MS MARCO %", 100 * (max(vals) - min(vals)) / min(vals), 3.8, 1)

# --- Vec2Text
v = json.load(open(R / "vec2text/summary_steps20_beam4.metrics.json", encoding="utf-8"))
check("Vec2Text content recall (84% in abstract)", 100 * v["vec2text"]["vs_prefix"]["content_recall"], 84)
check("LinearProbe n=5000 content recall (8% in intro)", 100 * v["linearprobe"]["5000"]["vs_prefix"]["content_recall"], 8)

# --- retrieval
ret = json.load(open(R / "advenc/retrieval_all_variants.json", encoding="utf-8"))["variants"]
check("AdvEnc v2 CPU retains about half of Recall@5 (%)", 100 * ret["advenc_v2_cpu"]["recall@5"]["mean"] / ret["vanilla"]["recall@5"]["mean"], 49)
# --- ablation (reduced scale, MiniLM): ranges and comparisons quoted in the second-encoder subsection
red = defaultdict(list)
for r in rows("advenc/reduced_scale_eval.csv"):
    if r["encoder"] == "minilm":
        red[(r["variant"], r["attacker"], int(r["n_align"]))].append(float(r["content_precision"]))
rm = lambda v, a, n: sum(red[(v, a, n)]) / len(red[(v, a, n)])
a40 = [rm(v, "adaptive", 40000) for v in ("v1", "v1s", "v2", "v2w")]
check("ablation adaptive n=40k min", min(a40), 0.450, 3)
check("ablation adaptive n=40k max", max(a40), 0.578, 3)
check("ablation vanilla n=40k", rm("vanilla", "adaptive", 40000), 0.469, 3)
na = [rm(v, "nonadaptive", 40000) for v in ("v1", "v1s", "v2", "v2w")]
check("ablation non-adaptive min", min(na), 0.226, 3)
check("ablation non-adaptive max", max(na), 0.402, 3)
check("v1s adaptive n=5000", rm("v1s", "adaptive", 5000), 0.285, 3)
check("vanilla adaptive n=5000 (reduced eval)", rm("vanilla", "adaptive", 5000), 0.380, 3)
rr = json.load(open(R / "advenc/reduced_scale_retrieval.json", encoding="utf-8"))
for v, q in (("v1", 0.875), ("v1s", 0.838), ("v2", 0.103), ("v2w", 0.058)):
    check(f"Recall@5 {v}", rr[f"minilm/{v}"]["recall@5"]["mean"], q, 3)

# --- MPNet (reduced scale): numbers quoted in the second-encoder paragraph
mp = defaultdict(list)
for r in rows("advenc/reduced_scale_eval.csv"):
    if r["encoder"] == "mpnet":
        mp[(r["variant"], r["attacker"], int(r["n_align"]))].append(float(r["content_precision"]))
mm = lambda v, a, n: sum(mp[(v, a, n)]) / len(mp[(v, a, n)])
check("MPNet v2@b8 adaptive n=40k", mm("v2@b8", "adaptive", 40000), 0.339, 3)
check("MPNet vanilla adaptive n=40k", mm("vanilla", "adaptive", 40000), 0.467, 3)
check("MPNet v2@b8 adaptive n=10k", mm("v2@b8", "adaptive", 10000), 0.241, 3)
check("MPNet vanilla adaptive n=10k", mm("vanilla", "adaptive", 10000), 0.437, 3)
check("MPNet v2@b8 adaptive n=5k", mm("v2@b8", "adaptive", 5000), 0.191, 3)
check("MPNet v1@b8 adaptive n=40k", mm("v1@b8", "adaptive", 40000), 0.462, 3)
check("MPNet v1 adaptive n=40k", mm("v1", "adaptive", 40000), 0.458, 3)
for v, q in (("v2@b8", 0.075), ("v1@b8", 0.216), ("v1", 0.235)):
    check(f"MPNet non-adaptive {v}", mm(v, "nonadaptive", 40000), q, 3)
for v, q in (("vanilla", 0.824), ("v1", 0.846), ("v1@b8", 0.818), ("v2@b8", 0.021)):
    check(f"MPNet Recall@5 {v}", rr[f"mpnet/{v}"]["recall@5"]["mean"], q, 3)

print(f"\n{bad} mismatches")
