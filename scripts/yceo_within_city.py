"""yceo_within_city.py — the cross-instrument row of Supplementary Table S1(b).

Fits the joint (Mundlak) model of the within-city population term on the YCEO v4 surface urban
heat island (2003–2018, annual composites, night and day) against the same GHS-UCDB population
density used for the station panel, so the station-air result can be checked on an independent
instrument. Reads yceo_city_panel.csv and hist_predictors.csv from the companion record.

Run after make_inputs.py.
"""
import os, warnings, numpy as np, pandas as pd, statsmodels.api as sm
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
import sys; sys.path.insert(0, HERE)
from uhi_paths import find

y = pd.read_csv(find("yceo_city_panel.csv"))
y = y[(y.npix >= 20) & y.suhi.notna() & (y.variant == "annual")].copy()
y["epoch"] = (y.year / 5).round().astype(int) * 5          # 2003–2018 composites to the nearest GHS epoch
pred = pd.read_csv(find("hist_predictors.csv")).rename(columns={"year": "epoch", "CityID": "city_id"})
m = y.merge(pred[["city_id", "epoch", "ln_popdensity"]], on=["city_id", "epoch"], how="inner")


def mundlak(d, yvar):
    s = d.dropna(subset=[yvar, "ln_popdensity"]).copy()
    for v in (yvar, "ln_popdensity"):
        s["t_" + v] = s[v] - s.groupby("year")[v].transform("mean")
    bar = s.groupby("city_id")["t_ln_popdensity"].transform("mean")
    s["between"], s["within"] = bar, s["t_ln_popdensity"] - bar
    f = sm.OLS(s["t_" + yvar], sm.add_constant(s[["within", "between"]])).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(s.city_id)[0]})
    ci = f.conf_int()
    return (f"within {f.params['within']:+.3f} ({ci.loc['within', 0]:+.2f} to {ci.loc['within', 1]:+.2f})  "
            f"between {f.params['between']:+.3f} ({ci.loc['between', 0]:+.2f} to {ci.loc['between', 1]:+.2f})  "
            f"n = {s.city_id.nunique():,} cities, {len(s):,} city-years")


print("YCEO v4 surface UHI on GHS-UCDB population density, joint model, city-clustered 95% CI")
for band in ("night", "day"):
    print(f"  {band:6} {mundlak(m[m.band == band], 'suhi')}")
