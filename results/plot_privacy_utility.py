"""Privacy vs. utility figure: adaptive-attacker content precision against retained Recall@5.

Every point is read from a result file:
  AdvEnc      retrieval: results/advenc/retrieval_all_variants.json   attacker: results/advenc/adaptive_budget_sweep.csv (LinearProbe, mean of seeds)
  Gaussian DP retrieval: results/dp_defense/dp_sweep_full.csv          attacker: results/dp_defense/dp_adaptive.csv
  Key rotation attacker: results/rotation_both_metrics.csv (n_align=1,000; retrieval is unchanged by construction)
x = Recall@5 as a fraction of the undefended value under the same retrieval protocol (AdvEnc: 1,000 queries, symmetric;
DP: 200 clean queries against the noised index). Two panels share axes: n_align = 1,000 (all defenses) and 10,000
(AdvEnc and DP; rotation was not run at 10,000). Shaded band: embedding-free content-word baseline (0.103-0.117).
The same-window rotation attacker coincides with the undefended point by construction (Proposition 3) and is not drawn.
Output: paper/fig_privacy_utility.pdf and .png
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = Path(__file__).resolve().parent
OUT = R.parent / "paper"

# Validated categorical slots 1-3 (scripts/validate_palette.js --pairs all, light): blue, orange, aqua.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e5e4e0", "#fcfcfb"
BASE_LO, BASE_HI = 0.103, 0.117  # embedding-free content-word baseline, MS MARCO (results/content_prior_floor.csv, geia_transfer_both_metrics.json)
EPS_LABELED = (20, 50, 100)


def rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def advenc_points(n):
    ret = json.load(open(R / "advenc/retrieval_all_variants.json", encoding="utf-8"))["variants"]
    base = ret["vanilla"]["recall@5"]["mean"]
    agg = defaultdict(list)
    for r in rows(R / "advenc/adaptive_budget_sweep.csv"):
        if r["attacker"] == "linearprobe" and int(r["n_align"]) == n:
            agg[r["encoder"]].append(float(r["content_precision"]))
    labels = {"advenc_v1_cpu": "v1 CPU", "advenc_v2_cpu": "v2 CPU", "advenc_v2_gpu50k": "v2 GPU", "advenc_v1_gpu50k": "v1 GPU"}
    pts = {labels[k]: (ret[k]["recall@5"]["mean"] / base, sum(v) / len(v)) for k, v in agg.items() if k in ret and k != "vanilla"}
    return pts, sum(agg["vanilla"]) / len(agg["vanilla"])


def dp_points(n):
    rec = {}
    for r in rows(R / "dp_defense/dp_sweep_full.csv"):
        rec[r["epsilon"]] = float(r["recall_at_5"])
    base = rec["inf"]
    pts = []
    for r in rows(R / "dp_defense/dp_adaptive.csv"):
        if int(r["n_align"]) != n:
            continue
        eps = float(r["epsilon"])
        key = str(int(eps)) if str(int(eps)) in rec else r["epsilon"]
        if key in rec:
            pts.append((eps, rec[key] / base, float(r["content_precision"])))
    return sorted(pts)


def rotation_stale():
    v = [float(r["content_precision"]) for r in rows(R / "rotation_both_metrics.csv") if int(r["n_align"]) == 1000 and r["condition"] == "next_epoch"]
    return sum(v) / len(v)


def panel(ax, n, with_rotation, base_label_xy, base_label_ha):
    ax.axhspan(BASE_LO, BASE_HI, color=INK2, alpha=0.18, lw=0)
    ax.text(*base_label_xy, "embedding-free baseline", fontsize=6.5, color=INK2, va="bottom", ha=base_label_ha)
    adv, van = advenc_points(n)
    dp = dp_points(n)
    ax.plot([p[1] for p in dp], [p[2] for p in dp], color=ORANGE, lw=1.2, zorder=2)
    ax.scatter([p[1] for p in dp], [p[2] for p in dp], s=22, marker="s", color=ORANGE, edgecolor=SURFACE, linewidth=0.8, zorder=3)
    for eps, x, y in dp:
        if eps in EPS_LABELED:
            ax.annotate(f"$\\varepsilon$={int(eps)}", (x, y), textcoords="offset points", xytext=((5, 6) if eps == 20 else (-5, 6)), fontsize=6.5, color=INK2, ha=("left" if eps == 20 else "right"))
    offsets = {"v1 CPU": (6, 5), "v2 CPU": (6, -3), "v2 GPU": (6, 7), "v1 GPU": (6, -13)}
    for name, (x, y) in adv.items():
        ax.scatter([x], [y], s=34, marker="o", color=BLUE, edgecolor=SURFACE, linewidth=0.8, zorder=4)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=offsets[name], fontsize=6.5, color=INK2)
    ax.scatter([1.0], [van], s=40, marker="D", color=INK, edgecolor=SURFACE, linewidth=0.8, zorder=5)
    ax.annotate("undefended", (1.0, van), textcoords="offset points", xytext=(-6, 5), fontsize=6.5, color=INK, ha="right")
    if with_rotation:
        y = rotation_stale()
        ax.scatter([1.0], [y], s=34, marker="^", color=AQUA, edgecolor=SURFACE, linewidth=0.8, zorder=4)
        ax.annotate("rotation, stale attacker", (1.0, y), textcoords="offset points", xytext=(-6, -12), fontsize=6.5, color=INK2, ha="right")
    ax.set_xlim(-0.03, 1.10)
    ax.set_ylim(0, 0.52)
    ax.set_title(f"$n_{{\\mathrm{{align}}}}$ = {n:,}", fontsize=8, color=INK, loc="left")
    ax.set_xlabel("Recall@5 retained (fraction of undefended)", fontsize=7.5, color=INK2)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(labelsize=7, colors=INK2, length=2)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
    panel(axes[0], 1000, True, (0.30, BASE_HI + 0.006), "left")
    panel(axes[1], 10000, False, (1.08, BASE_LO - 0.032), "right")
    axes[0].set_ylabel("Adaptive-attacker content precision", fontsize=7.5, color=INK2)
    handles = [
        Line2D([], [], marker="o", ls="", color=BLUE, label="Adversarial training (AdvEnc)", markersize=5),
        Line2D([], [], marker="s", ls="-", color=ORANGE, label="Gaussian DP (increasing $\\varepsilon$ to the right)", markersize=5),
        Line2D([], [], marker="^", ls="", color=AQUA, label="Key rotation", markersize=5),
        Line2D([], [], marker="D", ls="", color=INK, label="Undefended", markersize=5),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "fig_privacy_utility.pdf", facecolor=SURFACE)
    fig.savefig(OUT / "fig_privacy_utility.png", dpi=200, facecolor=SURFACE)
    print("wrote", OUT / "fig_privacy_utility.pdf")


if __name__ == "__main__":
    main()
