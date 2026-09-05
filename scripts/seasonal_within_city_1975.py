"""Seasonal within-city test for the regional gap (S1.7, Table 2).

Builds the composition-free station UHI panel (within_city_panel.py construction, unchanged: 12 km
clean-rural screen, five-year window with >=3 valid years, >=3 references, alpha_s + tau_t
least-squares reference, 6.5 C/km elevation correction) separately for each season (JJA, DJF)
and element (TMIN, TMAX, TAVG) from seasonal_by_elem.csv, then estimates the within-city
population term (city + year FE, city-clustered SE) by region.

Prediction of the cooling-waste-heat reading: the North American term concentrates in JJA TMIN
and weakens in DJF; Europe null in both. A DJF term points to heating waste heat; a term of
similar size in both seasons points to traffic/industry or land cover.

Run:  python3 seasonal_within_city.py        (from any directory)
Out:  seasonal_uhi_panel_1975.csv next to this file, and the printed tables.
"""
import os, numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
from scipy.spatial import cKDTree
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
from supplement_paths import AIR, COMP, EXTRA
REL = AIR.rstrip("/"); COMP = COMP.rstrip("/")
EP = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020]
L = 6.5 / 1000

sea = pd.read_csv(f"{EXTRA}seasonal_by_elem.csv", dtype={"id": str}).drop_duplicates(["id", "year"])
print(f"seasonal station-years: {len(sea):,}, stations {sea.id.nunique():,}")
meta = pd.read_csv(f"{REL}/need_broad_meta.csv", dtype={"id": str}); S = {r.id: (r.lat, r.lon, r.elev) for r in meta.itertuples()}
match = pd.read_csv(f"{REL}/city_station_match_broad.csv", dtype={"urban": str, "rural": str})
allc = pd.read_csv(f"{REL}/city_centroids.csv")[["lon", "lat"]].drop_duplicates()
def proj(la, lo): return np.c_[la * 111.0, lo * 111.0 * np.cos(np.radians(la))]
ct = cKDTree(proj(allc.lat.values, allc.lon.values)); cc = {}
def clean(s):
    if s not in cc:
        la, lo, _ = S[s]; d, _ = ct.query(proj(np.array([la]), np.array([lo]))); cc[s] = d[0] >= 12.0
    return cc[s]
pred = pd.read_csv(f"{COMP}/hist_predictors.csv")
grp = pd.read_csv(f"{COMP}/city_groupings.csv").dropna(subset=["CityID"]).drop_duplicates("CityID")
grp["CityID"] = grp.CityID.astype(int); cont = grp.set_index("CityID")["continent"].to_dict(); kop = grp.set_index("CityID")["koppen"].to_dict()

def build(col):
    T = {}
    for r in sea[["id", "year", col]].dropna().itertuples():
        T.setdefault(r.id, {})[r.year] = getattr(r, col)
    def wm(sid, a, b, mn=3):
        d = T.get(sid, {}); x = [d[y] for y in range(a, b + 1) if y in d]; return np.mean(x) if len(x) >= mn else np.nan
    def tau(stations, yrs):
        obs = [(si, ti, wm(st, y - 2, y + 2)) for si, st in enumerate(stations) for ti, y in enumerate(yrs) if not np.isnan(wm(st, y - 2, y + 2))]
        if not obs: return None
        ns, nt = len(stations), len(yrs); X = np.zeros((len(obs), ns + nt - 1)); v = np.zeros(len(obs))
        for i, (si, ti, val) in enumerate(obs):
            X[i, si] = 1.0
            if ti > 0: X[i, ns + ti - 1] = 1.0
            v[i] = val
        beta, *_ = np.linalg.lstsq(X, v, rcond=None); alpha = beta[:ns]; tu = np.r_[0.0, beta[ns:]]
        cnt = {}
        for _, ti, _ in obs: cnt[ti] = cnt.get(ti, 0) + 1
        base = float(np.median(alpha)); return {yrs[ti]: base + tu[ti] for ti in range(nt) if cnt.get(ti, 0) >= 3}
    rows = []
    for r in match.itertuples():
        u = r.urban
        if u not in T or u not in S: continue
        cl = [s for s in (r.rural.split(";") if isinstance(r.rural, str) else []) if s in T and s in S and clean(s)]
        if len(cl) < 3: continue
        yrs = [y for y in EP if not np.isnan(wm(u, y - 2, y + 2))]
        if not yrs: continue
        R = tau(cl, yrs)
        if not R: continue
        ue = S[u][2]; re0 = np.nanmedian([S[s][2] for s in cl]); ec = L * (ue - re0) if np.isfinite(ue) and np.isfinite(re0) else 0.0
        for y in yrs:
            if y in R: rows.append({"CityID": r.city_id, "year": y, "n_ref": len(cl), "uhi": wm(u, y - 2, y + 2) - R[y] + ec})
    return pd.DataFrame(rows)

panels = []
for col in ["jja_tmin", "djf_tmin", "jja_tmax", "djf_tmax", "jja_tavg", "djf_tavg"]:
    d = build(col); d["series"] = col; panels.append(d)
    print(f"  {col}: {d.CityID.nunique():,} cities, {len(d):,} city-epochs")
P = pd.concat(panels).merge(pred[["CityID", "year", "ln_popdensity", "frac_built"]], on=["CityID", "year"], how="inner")
P["continent"] = P.CityID.map(cont); P["koppen"] = P.CityID.map(kop)
P.to_csv(f"{EXTRA}seasonal_uhi_panel_1975.csv", index=False)

def fe(d, rhs="ln_popdensity", y="uhi"):
    d = d.dropna(subset=[y, rhs]).copy()
    if d.CityID.nunique() < 40: return "too few"
    from linearmodels.panel import PanelOLS   # two-way effects without dummies (the dummy design fails on the full record)
    dd = d.set_index(["CityID", "year"]); pm = PanelOLS(dd[y], dd[[rhs]], entity_effects=True, time_effects=True, drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
    class _M: pass
    m = _M(); m.params = pm.params; m.bse = pm.std_errors; m.pvalues = pm.pvalues
    return f"{m.params[rhs]:+.3f} (se {m.bse[rhs]:.3f}, p {m.pvalues[rhs]:.3f}) n={d.CityID.nunique()}"

print("\n== within-city population term by season x element x region, 2000-2020, city + year FE")
print(f"{'series':10s} {'pooled':38s} {'North America':38s} {'Europe':38s} {'ex-NA':38s}")
for col in ["jja_tmin", "djf_tmin", "jja_tmax", "djf_tmax", "jja_tavg", "djf_tavg"]:
    s = P[P.series == col]
    print(f"{col:10s} {fe(s):38s} {fe(s[s.continent=='North America']):38s} {fe(s[s.continent=='Europe']):38s} {fe(s[s.continent.notna()&(s.continent!='North America')]):38s}")

print("\n== summer minus winter, same cities (difference of the two seasonal UHIs as outcome)")
for el in ["tmin", "tmax", "tavg"]:
    a = P[P.series == f"jja_{el}"][["CityID", "year", "uhi", "ln_popdensity", "continent"]].rename(columns={"uhi": "s"})
    b = P[P.series == f"djf_{el}"][["CityID", "year", "uhi"]].rename(columns={"uhi": "w"})
    d = a.merge(b, on=["CityID", "year"]); d["uhi"] = d.s - d.w
    print(f"  {el}: pooled {fe(d)} | NA {fe(d[d.continent=='North America'])} | Europe {fe(d[d.continent=='Europe'])}")

print("\n== North America by Köppen group, JJA vs DJF TMIN")
for k in ["B", "C", "D"]:
    s = P[(P.continent == "North America") & (P.koppen.astype(str).str.startswith(k))]
    print(f"  {k}: JJA {fe(s[s.series=='jja_tmin'])} | DJF {fe(s[s.series=='djf_tmin'])}")
print("\n== level check: mean seasonal UHI by region (2000-2020), TMIN")
print(P[P.series.isin(["jja_tmin", "djf_tmin"])].groupby(["continent", "series"]).uhi.mean().round(2).unstack().to_string())
