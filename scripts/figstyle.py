"""
House style for the display items.

Journal conventions applied here (Elsevier / Urban Climate):
  * no in-figure titles — the caption carries the message, so titles duplicate it
  * double-column width 190 mm; single-column 90 mm
  * one sans face throughout, 7–9 pt at final size (never below 7 pt after scaling)
  * colour-blind-safe Okabe–Ito pairs, used consistently across figures:
        BLUE  = between-city / city size      ORANGE = within-city / density
  * colour never the sole carrier of meaning — line style and direct labels repeat it
  * spines only where they anchor the data; no grid behind scatter
  * direct labelling in preference to legends
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

MM = 1 / 25.4
W2 = 190 * MM          # double column
W1 = 90 * MM           # single column

BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#9AA0A6"
PURPLE = "#7B4EA0"     # income, a third dimension distinct from size and density
SLATE = "#5A6068"      # the annual-mean element where GREY is too faint to carry a line
INK = "#1a1a1a"

def use():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.linewidth": 0.6, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 2.6, "ytick.major.size": 2.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "legend.frameon": False, "legend.handlelength": 1.6,
        "lines.solid_capstyle": "round",
    })


def panel_label(ax, letter, dx=-0.10, dy=1.04):
    """Bold panel letter, outside the axes, consistent across figures."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", va="top", ha="left")


def annotate(ax, x, y, text, color=INK, ha="left", va="top", size=7.5):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=size, color=color,
            ha=ha, va=va, linespacing=1.35)
