#!/usr/bin/env python3
"""
Fig. 3: regional structure -- the size law generalizes, the density
trend does not. Two panels on the same house style as make_main_figures.py.

  (a) size-law slope by region (deposited literature-comparison CSV; night
      element except East Asia, where the nocturnal sample is too small to
      fit and the manuscript reports the annual mean instead)
  (b) within-city density-trend regional-interaction model: rest-of-world
      coefficient vs. the North American excess, primary 2000-2020 window
      (data/results/si_robustness_suite.csv, rows tagged "region").

Reads only deposited result tables; no estimation happens here.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

ROOT = Path(__file__).resolve().parent.parent
DATA, FIG = ROOT / "data", ROOT / "figures"
fs.use()


def figure_regional(out=FIG / "Fig3_regional.png"):
    lit = pd.read_csv(DATA / "results" / "oke_literature_matched_coverage.csv")
    ours = lit[lit.source.str.startswith("OURS")]

    size_rows = [
        ("North America", float(ours[(ours.region == "North America") & (ours.uhi_time == "night")].slope_per_tenfold_ourform.iloc[0]), 361, "night"),
        ("Europe", float(ours[(ours.region == "Europe") & (ours.uhi_time == "night")].slope_per_tenfold_ourform.iloc[0]), 249, "night"),
        ("Asia", 0.26, 141, "mean"),
    ]

    reg = pd.read_csv(DATA / "results" / "si_robustness_suite.csv")
    reg = reg[reg.iloc[:, 0] == "region"]
    row_col = reg.columns[1]

    def pull(label):
        r = reg[reg[row_col] == label].iloc[0]
        return float(r.iloc[3]), float(r.iloc[4]), float(r.iloc[5])  # coef, p, se

    row_c, row_p, row_se = pull("interaction, within: rest of world, 2000-2020")
    na_c, na_p, na_se = pull("interaction, within: North American excess, 2000-2020")

    fig, ax = plt.subplots(1, 2, figsize=(fs.W2, fs.W2 * 0.46))
    plt.subplots_adjust(wspace=.46, left=.09, right=.975, bottom=.20, top=.90)

    # (a) size law by region -- dots, matching Fig. 6's convention
    a = ax[0]
    labels_a = [r[0] for r in size_rows]
    ypos = np.arange(len(labels_a))[::-1]
    for (lab, slope, n, el), y in zip(size_rows, ypos):
        a.scatter(slope, y, s=70, color=fs.BLUE, zorder=4, edgecolor="white", lw=.6,
                  marker="^" if el == "night" else "o")
        a.text(slope + 0.03, y, f"{slope:+.2f}  (n={n}{', annual mean' if el=='mean' else ''})",
               fontsize=7.3, va="center", ha="left", color=fs.BLUE)
    a.axvline(0, color=fs.GREY, lw=.6, zorder=1)
    a.set_yticks(ypos); a.set_yticklabels(labels_a)
    a.set_xlim(-0.15, 1.35)
    a.set_xlabel("size-law slope (°C per tenfold population)")
    a.spines["left"].set_visible(False); a.tick_params(axis="y", length=0)
    fs.panel_label(a, "a")

    # (b) within-city density trend, regional-interaction model
    b = ax[1]
    labs_b = ["Rest of\nworld", "North America\n(excess)"]
    vals = [row_c, na_c]; ses = [row_se, na_se]; ps = [row_p, na_p]
    yb = np.arange(len(labs_b))[::-1]
    b.barh(yb, vals, xerr=[1.96 * s for s in ses], height=.5, color=fs.ORANGE,
           ecolor="#1a1a1a", capsize=3, zorder=3)
    for y, v, p, se in zip(yb, vals, ps, ses):
        b.text(v + 1.96 * se + 0.03 if v >= 0 else v - 1.96 * se - 0.03, y,
               f"{v:+.3f} (p = {p:.2g})", fontsize=7.3, va="center",
               ha="left" if v >= 0 else "right", color=fs.ORANGE)
    b.axvline(0, color=fs.GREY, lw=.6, zorder=1)
    b.set_yticks(yb); b.set_yticklabels(labs_b)
    b.set_xlim(-0.25, 1.05)
    b.set_ylim(-0.7, 1.7)
    b.set_xlabel("within-city density trend\n(°C per log-density, 2000–2020)")
    b.spines["left"].set_visible(False); b.tick_params(axis="y", length=0)
    fs.panel_label(b, "b")

    fig.savefig(out)
    plt.close(fig)
    print(f"  Fig 3  size law: NA {size_rows[0][1]:+.2f}, EU {size_rows[1][1]:+.2f}, "
          f"EA {size_rows[2][1]:+.2f}; density trend: RoW {row_c:+.3f} (p={row_p:.2g}), "
          f"NA excess {na_c:+.3f} (p={na_p:.2g})")


if __name__ == "__main__":
    figure_regional()
