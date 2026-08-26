"""
Figure 6: the within-city effect on instruments that share nothing.

(A) The within-city density coefficient measured three ways: this record, an independent monthly
    archive with different quality control and aggregation, and a satellite surface product with
    different physics entirely. Bars are 95 % intervals.
(B) The two media rank their channels differently. Surface temperature is daytime-dominant and air
    temperature nocturnal, which is why a surface coefficient cannot stand in for an air one.
"""
import os, sys, warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uhi_paths import find
import figstyle
from figstyle import BLUE, ORANGE, GREY, SLATE, INK, W2, MM

# Paths resolve relative to this file, or from the environment, so the scripts run from a clone
# without editing. UHI_AIR_DATA points at the unpacked GHCN-Daily station UHI release; UHI_EXTRA
# at a directory holding inputs neither release redistributes (the GHS GeoPackages, the ring
# panel, the YCEO extract).
import os as _os
from pathlib import Path as _P
_HERE = _P(__file__).resolve().parent
RELEASE = str(_P(_os.environ.get("UHI_AIR_DATA", _HERE.parent.parent / "Paper3_ESSD" / "data")))
INPUTS  = str(_HERE.parent / "data" / "inputs")
EXTRA   = str(_P(_os.environ.get("UHI_EXTRA", _HERE.parent / "data" / "external")))


figstyle.use()
HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "..", "data", "inputs") + os.sep
OUT = os.path.join(HERE, "..", "figures", "FigReplication.png")

def mundlak(s, y, ent, grp):
    s = s.dropna(subset=[y, "ln_popdensity"]).copy()
    for v in [y, "ln_popdensity"]:
        s["t_" + v] = s[v] - s.groupby("year")[v].transform("mean")
    bar = s.groupby(ent)["t_ln_popdensity"].transform("mean")
    s["b"] = bar; s["w"] = s["t_ln_popdensity"] - bar
    f = smf.ols(f"t_{y} ~ w + b", data=s).fit(cov_type="cluster",
                                              cov_kwds={"groups": s[grp]})
    return f.params["w"], f.bse["w"], s[ent].nunique()

est = []
p = pd.read_csv(IN + "hist_stage3_panel_daynight.csv")
w, se, n = mundlak(p, "uhi_night", "CityID", "CityID")
est.append(("This record\nGHCN-Daily air", w, se, n, BLUE))

q = pd.read_csv(find("ghcnm_qcu_density_epoch_panel.csv")).rename(
    columns={"city_id": "CityID", "epoch": "year"})
q = q.dropna(subset=["ghcnm_uhi_C", "ghcnd_uhi_C", "ln_popdensity"])
w, se, n = mundlak(q, "ghcnm_uhi_C", "CityID", "CityID")
est.append(("Independent archive\nGHCN-M v4 QCU", w, se, n, SLATE))

y = pd.read_csv(IN + "yceo_city_panel.csv")
y = y[(y.npix >= 20) & y.suhi.notna() & (y.variant == "annual") & (y.band == "night")]
y["epoch"] = (y.year / 5).round() * 5
pred = pd.read_csv(find("hist_predictors.csv", air=RELEASE))
m = y.merge(pred.rename(columns={"year": "epoch"}), left_on=["city_id", "epoch"],
            right_on=["CityID", "epoch"], how="inner")
w, se, n = mundlak(m, "suhi", "city_id", "city_id")
est.append(("Independent instrument\nYCEO v4 surface", w, se, n, ORANGE))

fig, ax = plt.subplots(1, 2, figsize=(W2, 66 * MM),
                       gridspec_kw={"width_ratios": [1.25, 1]})

a = ax[0]
a.axvline(0, lw=0.6, color=INK, zorder=1)
for i, (lab, b, s, n, col) in enumerate(est):
    a.errorbar(b, i, xerr=1.96 * s, fmt="o", ms=5, color=col, lw=1.4,
               capsize=2.6, mec="white", mew=0.6, zorder=3)
    a.text(b, i + 0.24, f"{b:+.3f}", ha="center", fontsize=7.5, color=col)
    a.text(0.985, i - 0.26, f"{n:,} cities", transform=a.get_yaxis_transform(),
           ha="right", fontsize=6.8, color=GREY)
a.set_yticks(range(len(est)))
a.set_yticklabels([e[0] for e in est])
a.set_ylim(-0.6, len(est) - 0.35)
a.invert_yaxis()
a.set_xlabel("within-city density coefficient, night (°C per log-unit)")
a.set_title("A   the same coefficient, three measurements", loc="left", fontweight="bold")

b = ax[1]
lev = pd.DataFrame({"medium": ["Air\n(this record)", "Surface\n(YCEO v4)"],
                    "day": [0.28, 1.26], "night": [0.50, 0.55]})
x = np.arange(2)
b.bar(x - 0.19, lev.day, width=0.36, color=GREY, label="day")
b.bar(x + 0.19, lev.night, width=0.36, color=BLUE, label="night")
for i in range(2):
    b.text(i - 0.19, lev.day[i] + 0.04, f"{lev.day[i]:.2f}", ha="center", fontsize=7, color=INK)
    b.text(i + 0.19, lev.night[i] + 0.04, f"{lev.night[i]:.2f}", ha="center", fontsize=7, color=BLUE)
b.set_xticks(x); b.set_xticklabels(lev.medium)
b.set_ylabel("median heat island (°C)")
b.set_ylim(0, 1.55)
b.legend(loc="upper left", fontsize=7)
b.set_title("B   which channel dominates the level", loc="left", fontweight="bold")

fig.tight_layout(w_pad=2.2)
fig.savefig(OUT)
for lab, bb, s, n, _ in est:
    print(f"  {lab.replace(chr(10),' '):<34}{bb:+.3f} ± {1.96*s:.3f}  n={n:,}")
print(f"wrote {os.path.normpath(OUT)}")
