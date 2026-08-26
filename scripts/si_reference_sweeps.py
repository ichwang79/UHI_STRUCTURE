"""Measurement sweeps for the Supplementary Information, rebuilt under the anomaly rural reference.

The checks in `si_robustness_battery.py` all run on the deposited epoch panel and so hold the
measurement fixed. These three do the opposite: they rebuild the heat island from the station
record with one measurement choice moved at a time, and re-estimate. The choices are the lapse rate
applied to the urban-minus-rural height difference, the radius of the clean-rural contamination
screen, and the rural annulus itself.

The estimator is the corrected Mundlak form of Section 4.3, as in the battery, so the numbers here
are comparable with those there. The reference is the anomaly construction of Section 4.1
throughout: a least-squares fit of T[s,t] = alpha_s + tau_t over all of a city's clean-rural
stations, with R(t) = median_s(alpha_s) + tau_t.

Output: data/results/si_reference_sweeps.log
"""
import numpy as np, pandas as pd, warnings, time, math
from scipy.spatial import cKDTree
import statsmodels.formula.api as smf

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

warnings.filterwarnings("ignore")
D = f"{RELEASE}/"
EP = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020]
LAPSE, MINY, MINR = 6.5 / 1000.0, 3, 3

meta = pd.read_csv(D + "need_broad_meta.csv", dtype={"id": str})
S = {r.id: r.elev for r in meta.itertuples()}
LAT = {r.id: r.lat for r in meta.itertuples()}
LON = {r.id: r.lon for r in meta.itertuples()}
# The contamination screen is built against every GHS-UCDB urban centre, not only the ones this
# paper's cities are matched to. The v2 companion release screens against the full 11,422-centre
# database rather than the 11,161-centre working list its earlier version carried; the difference
# affects five of 7,657 rural stations (Paper 3 Sect. 2.4). CENTROIDS resolves to the rebuilt
# data_rebuilt/ copy if present, so this reproduces the deposited v2 screen; it falls back to the
# release's own city_centroids.csv otherwise.
_centroids_v2 = _P(RELEASE).parent / "data_rebuilt" / "city_centroids.csv"
CENTROIDS = str(_centroids_v2) if _centroids_v2.exists() else D + "city_centroids.csv"
cen = pd.read_csv(CENTROIDS)[["lon", "lat"]].drop_duplicates()
proj = lambda la, lo: np.c_[np.asarray(la) * 111.0,
                            np.asarray(lo) * 111.0 * np.cos(np.radians(np.asarray(la)))]
tree = cKDTree(proj(cen.lat.values, cen.lon.values))
DIST, _ = tree.query(proj(meta.lat.values, meta.lon.values))
SCREEN_D = dict(zip(meta.id, DIST))
adf = pd.read_csv(D + "annual_by_elem.csv", dtype={"id": str})
TA = {}
for r in adf.itertuples():
    if pd.notna(r.tavg):
        TA.setdefault(r.id, {})[r.year] = r.tavg
match = pd.read_csv(D + "city_station_match_broad.csv", dtype={"urban": str, "rural": str})
_COMPANION = str(_P(_os.environ.get("UHI_AIR_COMPANION", _HERE.parent / "data" / "reproduction_extras" / "companion")))
pred = pd.read_csv(f"{_COMPANION}/hist_predictors.csv")

_c = {}
def wm(sid, y):
    k = (sid, y)
    if k not in _c:
        d = TA.get(sid, {})
        x = [d[t] for t in range(y - 2, y + 3) if t in d]
        _c[k] = float(np.mean(x)) if len(x) >= MINY else np.nan
    return _c[k]


def build(screen_km, ring=None, lapse=LAPSE):
    """Rebuild the epoch panel from the station record with one measurement choice moved."""
    rows = []
    for r in match.itertuples():
        u = r.urban
        if u not in TA or u not in S: continue
        cand = [s for s in (r.rural.split(";") if isinstance(r.rural, str) else [])
                if s in TA and s in S]
        cl = [s for s in cand if SCREEN_D.get(s, 0) >= screen_km]
        if ring is not None:
            lo, hi = ring
            cl = [s for s in cl if lo <= math.hypot((LAT[s] - r.lat) * 111.0,
                  (LON[s] - r.lon) * 111.0 * math.cos(math.radians(r.lat))) <= hi]
        if len(cl) < MINR: continue
        yrs = [y for y in EP if np.isfinite(wm(u, y))]
        if not yrs: continue
        obs = [(i, j, wm(s, y)) for i, s in enumerate(cl) for j, y in enumerate(yrs)
               if np.isfinite(wm(s, y))]
        if not obs: continue
        ns, nt = len(cl), len(yrs)
        X = np.zeros((len(obs), ns + nt - 1)); v = np.zeros(len(obs))
        for k, (i, j, val) in enumerate(obs):
            X[k, i] = 1.0
            if j > 0: X[k, ns + j - 1] = 1.0
            v[k] = val
        b, *_ = np.linalg.lstsq(X, v, rcond=None)
        alpha, tau = b[:ns], np.r_[0.0, b[ns:]]
        cnt = {}
        for _, j, _x in obs: cnt[j] = cnt.get(j, 0) + 1
        base = float(np.median(alpha)); zr = np.nanmedian([S[s] for s in cl])
        ec = lapse * (S[u] - zr) if np.isfinite(S[u]) and np.isfinite(zr) else 0.0
        for j, y in enumerate(yrs):
            if cnt.get(j, 0) >= MINR:
                rows.append({"CityID": r.city_id, "year": y,
                             "uhi_obs": wm(u, y) - (base + tau[j]) + ec})
    return pd.DataFrame(rows).merge(pred, on=["CityID", "year"], how="inner")


def mundlak(d):
    s = d.dropna(subset=["uhi_obs", "ln_popdensity"]).drop_duplicates(["CityID", "year"]).copy()
    if s.CityID.nunique() < 50: return None
    for v in ["uhi_obs", "ln_popdensity"]:
        s["t_" + v] = s[v] - s.groupby("year")[v].transform("mean")
    bar = s.groupby("CityID").t_ln_popdensity.transform("mean")
    s["b"] = bar; s["w"] = s.t_ln_popdensity - bar
    f = smf.ols("t_uhi_obs ~ w + b", data=s).fit(cov_type="cluster",
                                                 cov_kwds={"groups": s.CityID})
    return (f.params["w"], f.pvalues["w"], f.params["b"],
            float(f.f_test("w - b = 0").pvalue), len(s), s.CityID.nunique())


def build_urban_mean(screen_km=12.0, radius_km=25.0, lapse=LAPSE, fixed_set=False):
    """Same construction, but the city value is the mean of every archive station within
    `radius_km` of the centroid rather than the single nearest one. Tests whether the choice of
    which in-city thermometer represents the city carries the trend. With `fixed_set`, the urban
    stations are restricted to those reporting at every epoch the city contributes, which applies
    the composition-free principle of Section 4.1 to the urban side as well as the rural one."""
    from scipy.spatial import cKDTree
    ok = [s for s in meta.id if s in TA]
    sub = meta[meta.id.isin(ok)]
    tree = cKDTree(proj(sub.lat.values, sub.lon.values))
    ids = sub.id.values
    rows = []
    for r in match.itertuples():
        near = [ids[i] for i in tree.query_ball_point(proj([r.lat], [r.lon])[0], radius_km)]
        if not near: continue
        cand = [s for s in (r.rural.split(";") if isinstance(r.rural, str) else [])
                if s in TA and s in S]
        cl = [s for s in cand if SCREEN_D.get(s, 0) >= screen_km]
        if len(cl) < MINR: continue
        yrs = [y for y in EP if any(np.isfinite(wm(u, y)) for u in near)]
        if not yrs: continue
        if fixed_set:
            near = [u for u in near if all(np.isfinite(wm(u, y)) for y in yrs)]
            if not near: continue
        obs = [(i, j, wm(s, y)) for i, s in enumerate(cl) for j, y in enumerate(yrs)
               if np.isfinite(wm(s, y))]
        if not obs: continue
        ns, nt = len(cl), len(yrs)
        X = np.zeros((len(obs), ns + nt - 1)); v = np.zeros(len(obs))
        for k, (i, j, val) in enumerate(obs):
            X[k, i] = 1.0
            if j > 0: X[k, ns + j - 1] = 1.0
            v[k] = val
        b, *_ = np.linalg.lstsq(X, v, rcond=None)
        alpha, tau = b[:ns], np.r_[0.0, b[ns:]]
        cnt = {}
        for _, j, _x in obs: cnt[j] = cnt.get(j, 0) + 1
        base = float(np.median(alpha)); zr = np.nanmedian([S[s] for s in cl])
        zu = np.nanmedian([S[s] for s in near if s in S])
        ec = lapse * (zu - zr) if np.isfinite(zu) and np.isfinite(zr) else 0.0
        for j, y in enumerate(yrs):
            if cnt.get(j, 0) < MINR: continue
            vals = [wm(u, y) for u in near if np.isfinite(wm(u, y))]
            if not vals: continue
            rows.append({"CityID": r.city_id, "year": y, "n_urban": len(vals),
                         "uhi_obs": float(np.mean(vals)) - (base + tau[j]) + ec})
    return pd.DataFrame(rows).merge(pred, on=["CityID", "year"], how="inner")


if __name__ == "__main__":
    print("=== in-city station selection: single nearest against the mean within 25 km", flush=True)
    a = build(12.0); b = build_urban_mean()
    ra, rb = mundlak(a), mundlak(b)
    j = a[["CityID", "year", "uhi_obs"]].merge(b[["CityID", "year", "uhi_obs", "n_urban"]],
                                               on=["CityID", "year"], suffixes=("_near", "_mean"))
    multi = j[j.n_urban > 1]
    c = build_urban_mean(fixed_set=True)
    rc = mundlak(c)
    multi_ids = set(b.loc[b.n_urban > 1, "CityID"])
    one_ids = set(b.loc[b.n_urban == 1, "CityID"]) - multi_ids
    def line(lab, d):
        r = mundlak(d)
        print(f"  {lab:46} {r[0]:>+7.3f} (p={r[1]:.4f})  {r[5]} cities" if r
              else f"  {lab:46} sample too small", flush=True)
    line("single nearest, all cities", a)
    line("mean within 25 km, all cities", b)
    line("mean within 25 km, urban set fixed across epochs", c)
    line("single nearest, one-station cities", a[a.CityID.isin(one_ids)])
    line("mean, one-station cities (identical by construction)", b[b.CityID.isin(one_ids)])
    line("single nearest, multi-station cities", a[a.CityID.isin(multi_ids)])
    line("mean, multi-station cities", b[b.CityID.isin(multi_ids)])
    line("mean fixed set, multi-station cities", c[c.CityID.isin(multi_ids)])
    j = a[["CityID", "year", "uhi_obs"]].merge(b[["CityID", "year", "uhi_obs", "n_urban"]],
                                               on=["CityID", "year"], suffixes=("_near", "_mean"))
    mm = j[j.n_urban > 1]; dm = mm.uhi_obs_mean - mm.uhi_obs_near
    print(f"  agreement on the {mm.CityID.nunique()} multi-station cities: r = "
          f"{mm.uhi_obs_near.corr(mm.uhi_obs_mean):.3f}, median difference {np.median(dm):+.3f} C, "
          f"mean |difference| {np.abs(dm).mean():.3f} C", flush=True)

    # The paper headlines the 2000-2020 window (population-back-cast credibility, Section 3), with
    # the full 1975-2020 record carried as a robustness check. These three sweeps therefore run on
    # both: mundlak(d) below is the full-record estimate, mundlak(d[d.year >= 2000]) the headline.
    def mundlak2000(d):
        return mundlak(d[d.year >= 2000]) if d is not None else None

    print("\n=== lapse rate applied to the urban-minus-rural height difference", flush=True)
    for g in (0.0, 5.0 / 1000, 6.5 / 1000, 8.0 / 1000, 9.8 / 1000):
        built = build(12.0, lapse=g)
        r = mundlak(built)
        print(f"  {g*1000:4.1f} C/km: within {r[0]:>+7.3f} (p={r[1]:.4f})  between {r[2]:>+7.3f}"
              f"  eq p={r[3]:.2e}", flush=True)
        r2 = mundlak2000(built)
        if r2: print(f"  {g*1000:4.1f} C/km (2000-2020): within {r2[0]:>+7.3f} (p={r2[1]:.4f})"
                     f"  between {r2[2]:>+7.3f}  eq p={r2[3]:.2e}  {r2[5]} cities", flush=True)

    print("\n=== contamination screen radius (annulus held at 40-150 km)", flush=True)
    for km in (0, 3, 6, 9, 12, 15, 20, 25, 30):
        t0 = time.time(); built = build(float(km)); r = mundlak(built)
        if r: print(f"  {km:>2} km: within {r[0]:>+7.3f} (p={r[1]:.4f})  between {r[2]:>+7.3f}"
                    f"  eq p={r[3]:.2e}  {r[5]} cities  [{time.time()-t0:.0f}s]", flush=True)
        r2 = mundlak2000(built)
        if r2: print(f"  {km:>2} km (2000-2020): within {r2[0]:>+7.3f} (p={r2[1]:.4f})"
                     f"  between {r2[2]:>+7.3f}  eq p={r2[3]:.2e}  {r2[5]} cities", flush=True)

    print("\n=== rural annulus (screen held at 12 km)", flush=True)
    for ring in [(25, 75), (30, 100), (40, 150), (25, 80), (60, 200), (100, 300)]:
        t0 = time.time(); built = build(12.0, ring=ring); r = mundlak(built)
        if r: print(f"  {ring[0]:>3}-{ring[1]:<3} km: within {r[0]:>+7.3f} (p={r[1]:.4f})"
                    f"  between {r[2]:>+7.3f}  eq p={r[3]:.2e}  {r[5]} cities"
                    f"  [{time.time()-t0:.0f}s]", flush=True)
        else: print(f"  {ring[0]}-{ring[1]} km: sample too small", flush=True)
        r2 = mundlak2000(built)
        if r2: print(f"  {ring[0]:>3}-{ring[1]:<3} km (2000-2020): within {r2[0]:>+7.3f}"
                     f" (p={r2[1]:.4f})  between {r2[2]:>+7.3f}  eq p={r2[3]:.2e}  {r2[5]} cities",
                     flush=True)
