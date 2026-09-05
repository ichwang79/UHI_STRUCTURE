"""Land-cover tests of S1.6 and Supplementary Table S6 with 95% CIs (built-up share and MODIS NDVI on the fixed footprint)."""
import os, re, warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
SP = os.path.dirname(os.path.abspath(__file__))
exec(open(SP + "/building_volume.py").read().split('print("\\n== A.')[0])
def ci(e, s): return f"{e:+.3f} ({e-1.96*s:+.2f} to {e+1.96*s:+.2f})"
def fe(dd, rhs, y):
    dd = dd.dropna(subset=[y] + rhs.split(" + ")).drop_duplicates(["CityID", "year"]).copy()
    m = smf.ols(f"{y} ~ {rhs} + C(CityID) + C(year)", data=dd).fit(cov_type="cluster", cov_kwds={"groups": dd.CityID})
    return " ; ".join(f"{q} {ci(m.params[q], m.bse[q])}" for q in rhs.split(" + ")) + f"  n={dd.CityID.nunique()}"
from supplement_paths import EXTRA
nd = pd.read_csv(EXTRA + "gee_ndvi_extraction_5km.csv"); print("zones:", nd.zone.unique().tolist())
core_z = [z for z in nd.zone.unique() if "urban" in z.lower() or "core" in z.lower()]
def upper(z):
    m = re.findall(r"(\d+)\s*-\s*(\d+)", z); return int(m[0][1]) if m else 999
ring_z = [z for z in nd.zone.unique() if z.lower().startswith("rural") and upper(z) <= 10]
print("core zones:", core_z, "rural rings <=10 km:", ring_z)
w = nd.pivot_table(index=["CityID", "year"], columns="zone", values="NDVI_mean").reset_index()
w["ndvi_core"] = w[core_z].mean(axis=1); w["ndvi_contrast"] = w.ndvi_core - w[ring_z].mean(axis=1)
w = w[["CityID", "year", "ndvi_core", "ndvi_contrast"]]
base = P[["CityID", "year", "ln_popdensity", "frac_built", "uhi_obs"]].drop_duplicates(["CityID", "year"])
N = DN[["CityID", "year", "uhi_night", "uhi_day"]].merge(base, on=["CityID", "year"]).merge(w, on=["CityID", "year"], how="left")
N["continent"] = N.CityID.map(cont)
H = N[N.year >= 2000]
full = H.dropna(subset=["uhi_night", "ln_popdensity", "frac_built", "ndvi_core", "ndvi_contrast"])
keep = full.groupby("CityID").year.nunique(); full = full[full.CityID.isin(keep[keep >= 2].index)]
print("== night, 2000-2020, constant sample of cities carrying all series:", full.CityID.nunique())
for rhs in ["ln_popdensity", "ln_popdensity + frac_built", "ln_popdensity + ndvi_contrast", "ln_popdensity + ndvi_core", "ln_popdensity + frac_built + ndvi_contrast"]:
    print("  ", fe(full, rhs, "uhi_night"))
for reg in ["North America", "Europe"]:
    sub = full[full.continent == reg]; print(f"  {reg} alone   ", fe(sub, "ln_popdensity", "uhi_night")); print(f"  {reg} + both  ", fe(sub, "ln_popdensity + frac_built + ndvi_contrast", "uhi_night"))
print("== night, 1975-2020 (built-up share only; NDVI starts 2000)")
L = N.dropna(subset=["uhi_night", "ln_popdensity", "frac_built"])
print("  ", fe(L, "ln_popdensity", "uhi_night")); print("  ", fe(L, "ln_popdensity + frac_built", "uhi_night"))
print("== daily mean, 2000-2020")
Dm = H.dropna(subset=["uhi_obs", "ln_popdensity", "frac_built"])
print("  ", fe(Dm, "ln_popdensity", "uhi_obs")); print("  ", fe(Dm, "ln_popdensity + frac_built", "uhi_obs"))
Dn = H.dropna(subset=["uhi_obs", "ln_popdensity", "ndvi_core", "ndvi_contrast"])
print("  ", fe(Dn, "ln_popdensity + ndvi_core", "uhi_obs")); print("  ", fe(Dn, "ln_popdensity + ndvi_contrast", "uhi_obs"))
g = full.groupby("CityID"); print("== within-city SD: share", round(float((full.frac_built - g.frac_built.transform("mean")).std()), 3), "core NDVI", round(float((full.ndvi_core - g.ndvi_core.transform("mean")).std()), 3))
