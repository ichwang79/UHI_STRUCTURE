"""Intervals for Supplementary Table S1 (Driscoll-Kraay standard errors, the GHCN-M within-city
crosscheck) and Supplementary Table S4 (balanced-panel cross-city slope by epoch, income-spline
partial effects at the 90th and 99th percentiles)."""
import os, sys, warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
SP = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SP)
from supplement_paths import COMP, CODE_IN, CODE_SCRIPTS
IN = CODE_IN
def ci(e, s): return f"{e:+.3f} (se {s:.3f}, 95% CI {e-1.96*s:+.2f} to {e+1.96*s:+.2f})"

print("== Driscoll-Kraay, two-way FE, 1975-2020, density only")
from linearmodels.panel import PanelOLS
P = pd.read_csv(IN + "within_city_panel.csv")
s = P.dropna(subset=["uhi_obs", "ln_popdensity"]).drop_duplicates(["CityID", "year"]).set_index(["CityID", "year"])
for cov, kw in [("kernel", dict(kernel="bartlett")), ("clustered", dict(cluster_entity=True))]:
    f = PanelOLS(s["uhi_obs"], s[["ln_popdensity"]], entity_effects=True, time_effects=True, drop_absorbed=True).fit(cov_type=cov, **kw)
    print(f"  {cov:9s} {ci(f.params['ln_popdensity'], f.std_errors['ln_popdensity'])} n_cities={s.index.get_level_values(0).nunique()}")

print("== GHCN-M v4 QCU within-city crosscheck (Mundlak, city-clustered)")
q = pd.read_csv(COMP + "ghcnm_qcu_density_epoch_panel.csv").rename(columns={"city_id": "CityID", "epoch": "year"})
q = q.dropna(subset=["ghcnm_uhi_C", "ghcnd_uhi_C", "ln_popdensity"])
def mund(s, y):
    s = s.copy()
    for v in [y, "ln_popdensity"]: s["t_" + v] = s[v] - s.groupby("year")[v].transform("mean")
    bar = s.groupby("CityID")["t_ln_popdensity"].transform("mean"); s["b"], s["w"] = bar, s["t_ln_popdensity"] - bar
    f = smf.ols(f"t_{y} ~ w + b", data=s).fit(cov_type="cluster", cov_kwds={"groups": s.CityID}); return ci(f.params["w"], f.bse["w"])
for y in ["ghcnm_uhi_C", "ghcnd_uhi_C"]: print(f"  {y:12s} {mund(q, y)} n_obs={len(q)} cities={q.CityID.nunique()}")

print("== balanced panel (all 10 epochs, 1975-2020): cross-city slope by epoch; population reconstructed on the frozen footprint")
rep = pd.read_csv(IN + "representativeness.csv")[["CityID", "pop", "country"]].drop_duplicates("CityID")
pn = P.dropna(subset=["uhi_obs", "ln_popdensity"]).drop_duplicates(["CityID", "year"]).merge(rep, on="CityID", how="inner"); pn = pn[pn["pop"] > 0].copy(); pn["country"] = pn.country.fillna("NA")
d20 = pn[pn.year == 2020].set_index("CityID").ln_popdensity; pn = pn[pn.CityID.isin(d20.index)].copy()
pn["lp"] = np.log10(pn["pop"]) + (pn.ln_popdensity - pn.CityID.map(d20)) / np.log(10)
cnt = pn.groupby("CityID").year.nunique(); bal = pn[pn.CityID.isin(cnt[cnt == 10].index)].copy(); print("  balanced cities:", bal.CityID.nunique())
for yr in sorted(bal.year.unique()):
    sub = bal[bal.year == yr]; m = smf.ols("uhi_obs ~ lp", data=sub).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(sub.country)[0]})
    print(f"  {yr}: {ci(m.params['lp'], m.bse['lp'])} n={len(sub)}")
bal["t"] = (bal.year - 1975) / 45; m = smf.ols("uhi_obs ~ lp * t + C(year)", data=bal).fit(cov_type="cluster", cov_kwds={"groups": bal.CityID})
print(f"  size x time interaction (change over 1975->2020): {ci(m.params['lp:t'], m.bse['lp:t'])} p={m.pvalues['lp:t']:.2f}")
bal["lp0"] = bal.CityID.map(bal[bal.year == 1975].set_index("CityID").lp)
m = smf.ols("uhi_obs ~ lp0 * t + C(year)", data=bal).fit(cov_type="cluster", cov_kwds={"groups": bal.CityID}); print(f"  fixed (1975) size x time: {ci(m.params['lp0:t'], m.bse['lp0:t'])} p={m.pvalues['lp0:t']:.2f}")

print("== income spline: partial effect at the 90th and 99th income percentiles relative to mean income")
src = open(CODE_SCRIPTS + "gdp_rcs.py").read(); exec(src.split("def main()")[0])
raw = pd.read_csv(IN + "uhi_level_model_cities.csv"); d, knots = prepare(raw); rng = np.random.default_rng(20260818)
for dv, label in DVS:
    s = d.dropna(subset=[dv, "ln_pop", "g"]); pts = np.nanpercentile(s.g, [90, 99])
    fit = smf.ols(f"{dv} ~ ln_pop + C(kop) + g + g_rcs2 + g_rcs3", data=s).fit(cov_type="HC1"); point = partial_effect(fit.params, pts, knots)
    draws = []; idx = np.arange(len(s))
    for b in range(1000):
        bs = s.iloc[rng.choice(idx, len(s), replace=True)]
        try: draws.append(partial_effect(smf.ols(f"{dv} ~ ln_pop + C(kop) + g + g_rcs2 + g_rcs3", data=bs).fit().params, pts, knots))
        except Exception: pass
    lo, hi = np.nanpercentile(np.array(draws), [2.5, 97.5], axis=0)
    print(f"  {label}: n={len(s)} top decile {point[0]:+.2f} ({lo[0]:+.2f} to {hi[0]:+.2f}); top 1% {point[1]:+.2f} ({lo[1]:+.2f} to {hi[1]:+.2f})")
