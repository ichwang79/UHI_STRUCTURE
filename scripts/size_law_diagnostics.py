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
