"""size_law_diagnostics.py — regression diagnostics for the night-time population-size law.

Reproduces the four diagnostic rows of Supplementary Table S1(a) and the matching blocks of
Supplementary Note 2 (S2.5): the RESET functional-form test, the Breusch–Pagan test with the
HC3-robust slope and its interval, and the slope after dropping the 1% of cities with the
largest Cook's distance, with its country-clustered interval.

Run after make_inputs.py; reads the same master city table as oke_analysis.py.
"""
import os, warnings, numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "oke_analysis.py"), encoding="utf-8").read().split("# ---------- B.")[0]
ns = {"__file__": os.path.join(HERE, "oke_analysis.py")}
exec(src, ns)                      # builds the master city table `d` exactly as oke_analysis.py does
d = ns["d"]

S = d.dropna(subset=["uhi_tmin", "lp"]).reset_index(drop=True)
y = S.uhi_tmin.values
X = sm.add_constant(S[["lp"]].astype(float)).values
g = pd.factorize(S.country)[0]

def fit_cluster(y, X, groups):
    return sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})

def show(label, m, n):
    lo, hi = m.conf_int()[1]
    print(f"  {label:48} {m.params[1]:+.3f} °C per tenfold population (95% CI {lo:+.2f} to {hi:+.2f}), n = {n}")

print(f"night-time size law, {len(S)} cities")
base = fit_cluster(y, X, g); show("country-clustered (reference)", base, len(S))

ols = sm.OLS(y, X).fit()
rs = linear_reset(ols, power=2, use_f=True)
print(f"  RESET functional-form test: F = {rs.statistic:.2f}, p = {rs.pvalue:.2f}")
bp = het_breuschpagan(ols.resid, X)
print(f"  Breusch–Pagan test: p = {bp[1]:.2f}")
show("HC3-robust slope", sm.OLS(y, X).fit(cov_type="HC3"), len(S))

cd = ols.get_influence().cooks_distance[0]
k = int(0.01 * len(S))                                   # top 1% of cities by Cook's distance
keep = np.argsort(cd)[:-k]
S2 = S.iloc[keep]
m2 = fit_cluster(S2.uhi_tmin.values, sm.add_constant(S2[["lp"]].astype(float)).values, pd.factorize(S2.country)[0])
show(f"Cook's distance, top {k} cities dropped", m2, len(S2))

# ---- the remaining rows of Supplementary Table S1(a): sample and specification checks ----
from iso_cont import ISO_EXTRA
print("\nsample and specification checks, night-time size law")
if "continent" not in S.columns:
    S["continent"] = np.nan
S["continent"] = S.continent.where(S.continent.notna(), S.country.map(ISO_EXTRA))

# mixed-effects model with a random country intercept
import statsmodels.formula.api as smf
mm = smf.mixedlm("uhi_tmin ~ lp", S, groups=S["country"]).fit(reml=True)
ci = mm.conf_int().loc["lp"]
print(f"  {'mixed effects, random country intercept':48} {mm.params['lp']:+.3f} (95% CI {ci[0]:+.2f} to {ci[1]:+.2f}), n = {len(S)}")

# excluding one region at a time
for label, mask in [("excluding the United States", S.country != "USA"),
                    ("excluding Europe", S.continent != "Europe"),
                    ("excluding Asia", S.continent != "Asia")]:
    s = S[mask]
    show(label, fit_cluster(s.uhi_tmin.values, sm.add_constant(s[["lp"]].astype(float)).values, pd.factorize(s.country)[0]), len(s))

# city size sampled across the full range: share of the GHS-UCDB centres in each size class
uni = pd.read_csv(ns["IN"] + "representativeness.csv")
uni["size_class"] = pd.cut(uni["pop"], [0, 3e5, 1e6, 5e6, np.inf], labels=["<0.3M", "0.3-1M", "1-5M", ">5M"])
share = uni.groupby("size_class", observed=True).sampled.mean().mul(100).round(1)
print("  share of GHS-UCDB centres in the level sample, by size class (%):", share.to_dict())

# income-control form
inc = pd.get_dummies(S.income.astype(str), prefix="inc", drop_first=True).astype(float)
for label, X in [("size + income group effects", pd.concat([S[["lp"]].astype(float), inc], axis=1)),
                 ("size + linear ln GDP per capita", S[["lp", "ln_gdp_c"]].astype(float)),
                 ("size + quadratic ln GDP per capita", S[["lp", "ln_gdp_c"]].astype(float).assign(g2=lambda t: t.ln_gdp_c ** 2))]:
    ok = X.notna().all(axis=1)
    m = fit_cluster(S.uhi_tmin.values[ok], sm.add_constant(X[ok]).values, pd.factorize(S.country[ok])[0])
    print(f"  {label:48} {m.params[1]:+.3f}, n = {int(ok.sum())}")

# permutation of population across cities, 500 draws
rng = np.random.default_rng(20260818)
perm = np.array([sm.OLS(y, sm.add_constant(rng.permutation(S.lp.values))).fit().params[1] for _ in range(500)])
print(f"  {'permutation of population (500 draws)':48} null {perm.mean():+.2f} ± {perm.std():.2f} against {base.params[1]:+.2f} observed")

# ---- level quantities quoted in the text and in Table 1 / Fig. 1 ----
print("\nlevel quantities on the night sample")
Sz = S.dropna(subset=["ln_popdensity"]).copy()
Sz["z_size"] = (Sz.lp - Sz.lp.mean()) / Sz.lp.std(); Sz["z_dens"] = (Sz.ln_popdensity - Sz.ln_popdensity.mean()) / Sz.ln_popdensity.std()
gz = pd.factorize(Sz.country)[0]
for label, cols in [("density alone, per SD", ["z_dens"]), ("size and density together, per SD", ["z_size", "z_dens"])]:
    m = fit_cluster(Sz.uhi_tmin.values, sm.add_constant(Sz[cols].astype(float)).values, gz); ci = m.conf_int()
    print("  " + label + ": " + "; ".join(f"{c} {m.params[i+1]:+.3f} ({ci[i+1][0]:+.2f} to {ci[i+1][1]:+.2f})" for i, c in enumerate(cols)) + f", n = {len(Sz)}")
m = fit_cluster(Sz.uhi_tmin.values, sm.add_constant(Sz[["lp", "ln_popdensity"]].astype(float)).values, gz); ci = m.conf_int()
print(f"  density with size held, per log unit: {m.params[2]:+.3f} ({ci[2][0]:+.2f} to {ci[2][1]:+.2f}); size {m.params[1]:+.3f} ({ci[1][0]:+.2f} to {ci[1][1]:+.2f}), n = {len(Sz)}")
rng = np.random.default_rng(20260818)
v, w = S.uhi_tmin.values, S["pop"].values
def wmean(idx): return np.average(v[idx], weights=w[idx])
bs = np.array([wmean(rng.integers(0, len(v), len(v))) for _ in range(2000)])
bm = np.array([np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(2000)])
print(f"  night level: city median {np.median(v):+.2f} ({np.percentile(bm, 2.5):+.2f} to {np.percentile(bm, 97.5):+.2f}); population-weighted mean {wmean(np.arange(len(v))):+.2f} ({np.percentile(bs, 2.5):+.2f} to {np.percentile(bs, 97.5):+.2f}); share of cities with a negative night level {100*(v<0).mean():.0f}%")
d2 = d.dropna(subset=["uhi_tmin", "uhi_tavg"]).copy(); d2["uhi_day"] = 2 * d2.uhi_tavg - d2.uhi_tmin
for reg in ["North America", "Europe", "Asia"]:
    s = d2[d2.continent == reg]
    if len(s) < 20: continue
    out = []
    for col, lab in [("uhi_tmin", "night"), ("uhi_day", "day"), ("uhi_tavg", "daily mean")]:
        x = s[col].values; se = x.std(ddof=1) / np.sqrt(len(x)); out.append(f"{lab} {x.mean():+.2f} ({x.mean()-1.96*se:+.2f} to {x.mean()+1.96*se:+.2f})")
    print(f"  {reg} mean level, n = {len(s)}: " + "; ".join(out))
print("  size law by region, country-clustered:")
for reg, col in [("North America", "uhi_tmin"), ("Europe", "uhi_tmin"), ("Asia", "uhi_tavg"), ("North America", "uhi_tavg"), ("Europe", "uhi_tavg")]:
    s = d[(d.continent == reg)].dropna(subset=[col, "lp"])
    m = fit_cluster(s[col].values, sm.add_constant(s[["lp"]].astype(float)).values, pd.factorize(s.country)[0]); ci = m.conf_int()
    print(f"    {reg:14} {col}: {m.params[1]:+.3f} ({ci[1][0]:+.2f} to {ci[1][1]:+.2f}), n = {len(s)}")
