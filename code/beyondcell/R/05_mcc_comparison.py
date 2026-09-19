#!/usr/bin/env python3
"""MCC comparison: BeyondCell vs the trained baselines.

Baseline mean MCC values are the per-method 'mean' column of the source-optimized
MCC panel (panel d) of results/result1.png. BeyondCell's mean MCC is computed from
results/classification_metrics.csv over its 27 evaluable pairs.

Bars are coloured on a diverging red/blue MCC scale (red = positive, blue =
negative) matching results/result1.png.

Renders results/figures/mcc_comparison.{pdf,png}.
"""
import csv
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[1]

# --- BeyondCell: mean MCC from our own results --------------------------------
rows = list(csv.DictReader(open(ROOT / "results/classification_metrics.csv")))
bc_mcc = [float(r["mcc"]) for r in rows if r["mcc"]]
bc_mean = st.mean(bc_mcc)

# --- Baselines: mean MCC read from result1.png panel d (source-optimized) -----
data = [
    ("CatBoost (fs)",  0.43),
    ("SSDA4Drug",      0.29),
    ("Log. reg. (fs)", 0.27),
    ("BeyondCell",     round(bc_mean, 2)),
    ("PRECISE",        0.04),
    ("Log. reg.",      0.03),
    ("CatBoost",       0.02),
    ("SCAD",           0.02),
    ("scDEAL",         0.02),
    ("scATD",         -0.01),
]

# Sort by MCC ascending so the largest bar sits at the top of a horizontal axis.
data.sort(key=lambda d: d[1])
labels = [d[0] for d in data]
values = [d[1] for d in data]

# Diverging red(+)/blue(-) scale, symmetric about 0, as in result1.png.
vlim = 0.45
norm = Normalize(vmin=-vlim, vmax=vlim)
cmap = plt.get_cmap("RdBu_r")
bar_colors = [cmap(norm(v)) for v in values]

fig, ax = plt.subplots(figsize=(6.6, 4.0))
y = range(len(labels))
ax.barh(list(y), values, color=bar_colors, height=0.68,
        edgecolor="#4A4A4A", linewidth=0.5, zorder=3)

# Value labels at the end of each bar.
for yi, v in zip(y, values):
    ax.text(v + (0.006 if v >= 0 else -0.006), yi, f"{v:+.2f}",
            va="center", ha="left" if v >= 0 else "right",
            fontsize=8.5, color="#333333")

# Reference line at MCC = 0 (no better than the prevalence-matched guess).
ax.axvline(0, color="#8A8F98", linewidth=1.0, zorder=2)

ax.set_yticks(list(y))
ax.set_yticklabels(labels, fontsize=9)
# Keep our method locatable by weight only (no colour distinction).
for tick, lab in zip(ax.get_yticklabels(), labels):
    if lab == "BeyondCell":
        tick.set_fontweight("bold")

ax.set_xlabel("Mean Matthews correlation coefficient (MCC)", fontsize=9.5)
ax.set_title("Drug-response classification: MCC across methods",
             fontsize=11, pad=10)
ax.set_xlim(-0.08, 0.50)
ax.grid(axis="x", color="#E5E7EB", linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.tick_params(length=0)

# Colorbar keying the red/blue scale to MCC.
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
cbar.set_label("MCC", fontsize=9)
cbar.outline.set_visible(False)

fig.tight_layout()
outdir = ROOT / "results/figures"
outdir.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(outdir / f"mcc_comparison.{ext}", dpi=200, bbox_inches="tight")
print(f"BeyondCell mean MCC = {bc_mean:.3f} (n={len(bc_mcc)})")
print("wrote", outdir / "mcc_comparison.pdf", "and .png")
